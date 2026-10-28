"""
    Precompute the frozen AMap teacher's current-frame outputs on the training set, so
    that the distillation runs at the student's speed (AMapDistiller(teacher_cache=...)).
    Each training sample is cached once: one draw of its future clip and of the
    photometric distortion, instead of a new draw at every epoch.
        torchrun --nproc_per_node=8 tools/amap_cache_teacher.py CONFIG OUT_DIR
    CONFIG is a live-teacher KD config (multi-frame train data). Existing files are
    skipped, so an interrupted run can be restarted.
"""
import argparse
import os
import os.path as osp
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import get_dist_info, init_dist

sys.path.append(os.path.abspath('.'))
import plugin  # noqa: F401, registers the plugin modules
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def worker_init_fn(worker_id, num_workers, rank):
    np.random.seed(num_workers * rank + worker_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('out_dir')
    parser.add_argument('--samples-per-gpu', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    assert cfg.data.train.multi_frame, 'the teacher needs the multi-frame clips'
    init_dist('pytorch', **cfg.dist_params)
    rank, world_size = get_dist_info()
    device = torch.cuda.current_device()
    os.makedirs(args.out_dir, exist_ok=True)

    distiller = build_model(cfg.model)
    distiller.init_weights()
    teacher = distiller.teacher.cuda()
    teacher.eval()

    dataset = build_dataset(cfg.data.train)
    indices = [i for i in range(rank, len(dataset), world_size)
               if not osp.exists(osp.join(args.out_dir, f'{dataset.idx2token[i]}.pt'))]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.samples_per_gpu,
        num_workers=args.workers,
        collate_fn=partial(collate, samples_per_gpu=args.samples_per_gpu),
        worker_init_fn=partial(worker_init_fn, num_workers=args.workers, rank=rank))

    start = time.time()
    for it, data in enumerate(loader):
        data = scatter(data, [device])[0]
        img_metas = data['img_metas']
        teacher_prev_data = [dict(prev_data, img=distiller._renorm_for_teacher(prev_data['img']))
                             for prev_data in data['all_prev_data']]
        with torch.no_grad():
            outs = teacher.forward_kd_teacher(distiller._renorm_for_teacher(data['img']), img_metas,
                                              teacher_prev_data)
        for b_i, meta in enumerate(img_metas):
            path = osp.join(args.out_dir, f"{meta['token']}.pt")
            # clone: a saved view would carry the whole batch storage
            torch.save({
                'bev_basic': outs['bev_basic'][b_i].half().cpu().clone(),
                'bev_refined': outs['bev_refined'][b_i].half().cpu().clone(),
                'lines': outs['lines'][b_i].float().cpu().clone(),
                'scores': outs['scores'][b_i].float().cpu().clone(),
            }, path + '.tmp')
            os.replace(path + '.tmp', path)
        if rank == 0 and (it + 1) % 50 == 0:
            elapsed = time.time() - start
            eta = elapsed / (it + 1) * (len(loader) - it - 1)
            print(f'[rank 0] {it + 1}/{len(loader)} batches, {elapsed / 60:.1f} min, '
                  f'ETA {eta / 60:.1f} min', flush=True)

    torch.distributed.barrier()
    if rank == 0:
        print(f'done: {len(dataset)} samples, {len(os.listdir(args.out_dir))} files in {args.out_dir}',
              flush=True)


if __name__ == '__main__':
    main()
