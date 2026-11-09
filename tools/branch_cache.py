"""
    Cache for the branched current/future KD (one pass, same augmented images for all):
      - the future teacher on its clip (same keys as tools/amap_cache_teacher.py, so the
        folder also serves the AMap / FTKD distillers): bev_basic, bev_refined, lines,
        scores, hs_embeds
      - the same teacher on the current frame only (empty memory): cur_bev_refined,
        cur_lines, cur_scores
      - a frozen current-frame student (the KD-free baseline): f0_bev (its neck BEV)
      - gt_fg: G.T. semantic mask on the BEV grid (uint8, 1 x H x W)
        torchrun --nproc_per_node=8 tools/branch_cache.py KD_CONFIG OUT_DIR \
            --student-config CFG --student-ckpt CKPT
    KD_CONFIG is a live-teacher KD config (multi-frame train data).
"""
import argparse
import os
import os.path as osp
import sys
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

sys.path.append(os.path.abspath('.'))
import plugin  # noqa: F401, registers the plugin modules
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def worker_init_fn(worker_id, num_workers, rank):
    import numpy as np
    np.random.seed(num_workers * rank + worker_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('out_dir')
    parser.add_argument('--student-config', required=True)
    parser.add_argument('--student-ckpt', required=True)
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

    student = build_model(Config.fromfile(args.student_config).model)
    load_checkpoint(student, args.student_ckpt, map_location='cpu', strict=True)
    student = student.cuda()
    student.eval()
    assert not student.temporal

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
        img_t = distiller._renorm_for_teacher(data['img'])
        teacher_prev_data = [dict(prev_data, img=distiller._renorm_for_teacher(prev_data['img']))
                             for prev_data in data['all_prev_data']]
        with torch.no_grad():
            outs = teacher.forward_kd_teacher(img_t, img_metas, teacher_prev_data)
            cur = teacher.forward_kd_teacher(img_t, img_metas, [])
            f0 = [student._test_step(data['img'][b:b+1], [meta], meta['scene_name'], 0)['bev_refined'][0]
                  for b, meta in enumerate(img_metas)]
            fg = torch.flip(data['semantic_mask'], [2,]).float().amax(1, keepdim=True)
            fg = F.adaptive_max_pool2d(fg, outs['bev_refined'].shape[-2:])
        for b_i, meta in enumerate(img_metas):
            path = osp.join(args.out_dir, f"{meta['token']}.pt")
            # clone: a saved view would carry the whole batch storage
            torch.save({
                'bev_basic': outs['bev_basic'][b_i].half().cpu().clone(),
                'bev_refined': outs['bev_refined'][b_i].half().cpu().clone(),
                'lines': outs['lines'][b_i].float().cpu().clone(),
                'scores': outs['scores'][b_i].float().cpu().clone(),
                'hs_embeds': outs['hs_embeds'][b_i].half().cpu().clone(),
                'cur_bev_refined': cur['bev_refined'][b_i].half().cpu().clone(),
                'cur_lines': cur['lines'][b_i].float().cpu().clone(),
                'cur_scores': cur['scores'][b_i].float().cpu().clone(),
                'f0_bev': f0[b_i].half().cpu().clone(),
                'gt_fg': fg[b_i].to(torch.uint8).cpu().clone(),
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
