"""
    Stage 0 of the branched KD: predictability probe F_0 -> Delta (= F_T - F_Tcur) on the
    joint cache (tools/branch_cache.py), heteroscedastic Gaussian NLL (beta-NLL), plus the
    P0 / P1 analysis on held-out scenes:
      - Delta energy per region (ahead/rear x near/far, G.T. fg/bg)
      - R^2 of the probe mean per region, mean sigma^2 per region
      - corr(sigma^2, squared error), corr(sigma^2, fg), corr(sigma^2, |Delta|^2)
        torchrun --nproc_per_node=8 tools/train_probe.py CACHE OUT_DIR
"""
import argparse
import json
import os
import os.path as osp
import random
import sys
import time

import mmcv
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.abspath('.'))
from plugin.models.mapers.branch_kd import BEVProbe  # noqa: E402

INFO = 'datasets/nuscenes/taeu_pkls/maptracker/nuscenes_map_infos_train.pkl'


class CacheSet(Dataset):

    def __init__(self, cache, tokens):
        self.cache, self.tokens = cache, tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        d = torch.load(osp.join(self.cache, f'{self.tokens[i]}.pt'), map_location='cpu')
        return d['f0_bev'], d['bev_refined'], d['cur_bev_refined'], d['gt_fg']


def region_masks(H, W, device):
    """columns follow x from -30 m to 30 m (MapTracker.plane), x > 0 is ahead"""
    x = (torch.arange(W, device=device).float() + 0.5) / W * 60 - 30
    x = x.view(1, 1, W).expand(1, H, W)
    return {
        'all': torch.ones_like(x, dtype=torch.bool),
        'ahead': x > 0, 'rear': x < 0,
        'ahead_near': (x > 0) & (x < 15), 'ahead_far': x >= 15,
        'rear_near': (x < 0) & (x > -15), 'rear_far': x <= -15,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('cache')
    p.add_argument('out_dir')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch', type=int, default=16, help='per GPU')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--beta', type=float, default=0.5, help='beta-NLL, 0 = plain NLL')
    p.add_argument('--holdout', type=float, default=0.1, help='fraction of scenes held out')
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()

    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dev = torch.device('cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    have = {f[:-3] for f in os.listdir(args.cache) if f.endswith('.pt')}
    infos = [x for x in mmcv.load(INFO) if x['token'] in have]
    scenes = sorted({x['scene_name'] for x in infos})
    random.Random(0).shuffle(scenes)
    held = set(scenes[:int(len(scenes) * args.holdout)])
    train_tok = [x['token'] for x in infos if x['scene_name'] not in held]
    val_tok = [x['token'] for x in infos if x['scene_name'] in held]

    # per-channel standardization of the input F_0 and of the target Delta
    stats_tok = random.Random(1).sample(train_tok, min(512, len(train_tok)))
    acc = torch.zeros(4, 256, dtype=torch.float64)
    n = 0
    for tok in stats_tok[rank::world]:
        f0, ft, fc, _ = CacheSet(args.cache, [tok])[0]
        f0, dl = f0.double().flatten(1), (ft.double() - fc.double()).flatten(1)
        acc += torch.stack([f0.sum(1), (f0 ** 2).sum(1), dl.sum(1), (dl ** 2).sum(1)])
        n += f0.shape[1]
    acc = acc.to(dev)
    n = torch.tensor(float(n), device=dev, dtype=torch.float64)
    dist.all_reduce(acc)
    dist.all_reduce(n)
    m = acc / n
    f0_mean, f0_std = m[0].float(), (m[1] - m[0] ** 2).clamp(min=1e-8).sqrt().float()
    tgt_mean, tgt_std = m[2].float(), (m[3] - m[2] ** 2).clamp(min=1e-8).sqrt().float()
    view = lambda v: v.view(1, -1, 1, 1)

    probe_args = dict(in_channels=256, out_channels=256, hidden=256, dilations=(1, 2, 4, 8))
    probe = BEVProbe(**probe_args).to(dev)
    model = DDP(probe, device_ids=[dev.index])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_set = CacheSet(args.cache, train_tok)
    sampler = DistributedSampler(train_set, shuffle=True)
    loader = DataLoader(train_set, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                        pin_memory=True, drop_last=True)
    total = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)

    def prep(batch):
        f0, ft, fc, fg = [b.to(dev, non_blocking=True).float() for b in batch]
        x = (f0 - view(f0_mean)) / view(f0_std)
        delta = ft - fc
        y = (delta - view(tgt_mean)) / view(tgt_std)
        return x, y, delta, ft, fg

    log = []
    start = time.time()
    for ep in range(args.epochs):
        sampler.set_epoch(ep)
        model.train()
        for it, batch in enumerate(loader):
            x, y, _, _, _ = prep(batch)
            mu, logvar = model(x)
            sq = ((y - mu) ** 2).mean(1, keepdim=True)
            nll = 0.5 * (sq * torch.exp(-logvar) + logvar)
            if args.beta > 0:
                nll = nll * logvar.detach().exp() ** args.beta
            loss = nll.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            if rank == 0 and it % 50 == 0:
                msg = (f'ep {ep} it {it}/{len(loader)} loss {loss.item():.4f} mse {sq.mean().item():.4f} '
                       f'logvar {logvar.mean().item():.3f} {(time.time() - start) / 60:.1f} min')
                print(msg, flush=True)
                log.append(msg)

    # ---- held-out analysis
    model.eval()
    val_set = CacheSet(args.cache, val_tok)
    vloader = DataLoader(val_set, batch_size=args.batch, sampler=DistributedSampler(val_set, shuffle=False),
                         num_workers=args.workers)
    regs = None
    corr_n = 0
    keys = ['n', 'sse', 'sst', 'sig2', 'dE', 'tE', 'w']
    for batch in vloader:
        x, y, delta, ft, fg = prep(batch)
        with torch.no_grad():
            mu, logvar = probe(x)
        H, W = y.shape[-2:]
        if regs is None:
            base = region_masks(H, W, dev)
            names = []
            for r in base:
                names += [r, r + '/fg', r + '/bg']
            sums = torch.zeros(len(names), len(keys), device=dev, dtype=torch.float64)
            # pairs for correlations: (sig2, sq), (sig2, fg), (sig2, |delta|^2)
            corr = torch.zeros(3, 5, device=dev, dtype=torch.float64)
            regs = True
        sq = ((y - mu) ** 2).mean(1)
        sst = (y ** 2).mean(1)
        sig2 = logvar.exp()[:, 0]
        w = (1.0 / sig2).clamp(0.1, 10.0)
        dE = (delta ** 2).mean(1)
        tE = (ft ** 2).mean(1)
        fgb = fg[:, 0] > 0
        i = 0
        for r, mask in base.items():
            mask = mask.expand_as(sq)
            for sub in (mask, mask & fgb, mask & ~fgb):
                sub = sub.double()
                vals = [sub.sum()] + [(v.double() * sub).sum() for v in (sq, sst, sig2, dE, tE, w)]
                sums[i] += torch.stack(vals)
                i += 1
        for j, other in enumerate((sq, fg[:, 0], dE)):
            a, b = sig2.double().flatten(), other.double().flatten()
            corr[j] += torch.stack([a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()])
        corr_n += sig2.numel()
    dist.all_reduce(sums)
    dist.all_reduce(corr)
    cn = torch.tensor(float(corr_n), device=dev, dtype=torch.float64)
    dist.all_reduce(cn)

    if rank == 0:
        res = {'regions': {}, 'corr': {}}
        w_all = (sums[0, 6] / sums[0, 0]).item()
        for i, name in enumerate(names):
            s = sums[i]
            cnt = s[0].item()
            if cnt == 0:
                continue
            res['regions'][name] = {
                'frac_cells': cnt / sums[0, 0].item(),
                'R2': 1 - (s[1] / s[2]).item(),
                'sigma2_mean': (s[3] / cnt).item(),
                'delta_energy_ratio': (s[4] / s[5]).item(),  # |Delta|^2 / |F_T|^2
                'w_inv_var_mean_norm': (s[6] / cnt).item() / w_all,
            }
        for j, name in enumerate(('sigma2~sq_err', 'sigma2~gt_fg', 'sigma2~delta_energy')):
            sa, sb, saa, sbb, sab = corr[j].tolist()
            N = cn.item()
            cov = sab / N - sa * sb / N ** 2
            va, vb = saa / N - (sa / N) ** 2, sbb / N - (sb / N) ** 2
            res['corr'][name] = cov / max((va * vb) ** 0.5, 1e-12)
        res['args'] = vars(args)
        res['num_train'], res['num_val'] = len(train_tok), len(val_tok)
        json.dump(res, open(osp.join(args.out_dir, 'analysis.json'), 'w'), indent=1)
        torch.save({'state_dict': probe.state_dict(), 'probe_args': probe_args,
                    'f0_mean': f0_mean.cpu(), 'f0_std': f0_std.cpu(),
                    'tgt_mean': tgt_mean.cpu(), 'tgt_std': tgt_std.cpu()},
                   osp.join(args.out_dir, 'probe.pth'))
        open(osp.join(args.out_dir, 'train_log.txt'), 'w').write('\n'.join(log))
        print(json.dumps(res, indent=1), flush=True)
    dist.barrier()


if __name__ == '__main__':
    main()
