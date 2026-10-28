"""Per-frame video of one scene: the predicted map elements projected onto the 6 cameras
(2x3 grid), the predictions in BEV and the G.T. in BEV, side by side.

    python tools/visualization/vis_cam_bev.py CONFIG --result work_dir/submission_vector.json \
        --scene scene-0345 --out vis/scene-0345.mp4

The config's eval_config gives the G.T. vectors in ego meters; its ann_file must contain
the scene (e.g. a single-scene val pkl). Predictions are the denormalized vectors of
submission_vector.json (ego meters).
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import cv2
import imageio
import mmcv
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset

from tracking.cmap_utils.utils import import_plugin

# BGR, same class colors as vis_per_frame.py (ped_crossing blue, divider red, boundary green)
COLORS = {0: (255, 90, 0), 1: (40, 40, 230), 2: (40, 170, 40)}
NAMES = {0: 'ped_crossing', 1: 'divider', 2: 'boundary'}
CAM_GRID = [['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
            ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']]


def parse_args():
    parser = argparse.ArgumentParser(description='Camera + BEV prediction / G.T. video')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--result', required=True, help='submission_vector.json')
    parser.add_argument('--scene', required=True, help='scene name, e.g. scene-0345')
    parser.add_argument('--out', required=True, help='output .mp4 path')
    parser.add_argument('--score-thr', type=float, default=0.4)
    parser.add_argument('--cam-w', type=int, default=480, help='width of one camera tile')
    parser.add_argument('--px-per-m', type=float, default=9.0, help='BEV scale')
    parser.add_argument('--fps', type=float, default=2.0, help='nuScenes key frames are 2Hz')
    return parser.parse_args()


def densify(line, step=0.5):
    """Resample a polyline every `step` meters so that its projection bends correctly."""
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    dist = np.concatenate([[0], np.cumsum(seg)])
    if dist[-1] < 1e-6:
        return line
    t = np.linspace(0, dist[-1], max(int(dist[-1] / step), 2))
    return np.stack([np.interp(t, dist, line[:, 0]), np.interp(t, dist, line[:, 1])], axis=1)


def draw_on_cam(img, vectors, ego2img, scale):
    """vectors: list of (label, (N, 2) ego meters). The map is on the ground plane (z=0)."""
    for label, line in vectors:
        pts = densify(line)
        pts_h = np.concatenate([pts, np.zeros((len(pts), 1)), np.ones((len(pts), 1))], axis=1)
        proj = pts_h @ ego2img.T
        depth = proj[:, 2]
        uv = proj[:, :2] / np.clip(depth[:, None], 1e-5, None) * scale
        valid = depth > 0.5
        # draw each run of points in front of the camera, cv2 clips at the image border
        start = None
        for i in range(len(pts) + 1):
            if i < len(pts) and valid[i]:
                start = i if start is None else start
            elif start is not None:
                if i - start >= 2:
                    run = np.round(uv[start:i]).astype(np.int32)
                    cv2.polylines(img, [run], False, COLORS[label], 3, cv2.LINE_AA)
                start = None
    return img


def bev_panel(vectors, roi_size, px_per_m, title):
    """Forward is up, left is left. roi_size: (x range, y range) in meters."""
    h, w = int(roi_size[0] * px_per_m), int(roi_size[1] * px_per_m)
    canvas = np.full((h, w, 3), 255, np.uint8)
    for d in range(-int(roi_size[0] // 2), int(roi_size[0] // 2) + 1, 10):
        y = int(h / 2 - d * px_per_m)
        cv2.line(canvas, (0, y), (w, y), (230, 230, 230), 1)

    def to_px(p):
        # ego x (forward) -> up, ego y (left) -> left
        return np.stack([w / 2 - p[:, 1] * px_per_m, h / 2 - p[:, 0] * px_per_m], axis=1)

    for label, line in vectors:
        cv2.polylines(canvas, [np.round(to_px(line)).astype(np.int32)], False, COLORS[label], 2,
                      cv2.LINE_AA)
    # ego car (4.4m x 1.8m)
    car = np.array([[2.2, 0.9], [2.2, -0.9], [-2.2, -0.9], [-2.2, 0.9]])
    cv2.fillPoly(canvas, [np.round(to_px(car)).astype(np.int32)], (0, 140, 255))
    cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (120, 120, 120), 1)
    return with_title(canvas, title)


def with_title(img, title, bar=34):
    out = np.full((img.shape[0] + bar, img.shape[1], 3), 255, np.uint8)
    out[bar:] = img
    cv2.putText(out, title, (8, bar - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    return out


def legend(width, height=34):
    bar = np.full((height, width, 3), 255, np.uint8)
    x = 10
    for label in range(3):
        cv2.line(bar, (x, height // 2), (x + 30, height // 2), COLORS[label], 3, cv2.LINE_AA)
        cv2.putText(bar, NAMES[label], (x + 38, height // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (30, 30, 30), 1, cv2.LINE_AA)
        x += 200
    return bar


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)
    dataset = build_dataset(cfg.eval_config)
    results = mmcv.load(args.result)['results']
    roi_size = cfg.roi_size

    idxs = [i for i, s in enumerate(dataset.samples) if s['scene_name'] == args.scene]
    assert idxs, f'{args.scene} is not in {cfg.eval_config.ann_file}'

    frames = []
    for n, idx in enumerate(idxs):
        sample = dataset.samples[idx]
        token = sample['token']
        input_dict = dataset.get_sample(idx)
        cam_names = list(sample['cams'].keys())

        pred = results[token]
        pred_vectors = [(int(l), np.array(v, dtype=np.float64).reshape(-1, 2))
                        for v, s, l in zip(pred['vectors'], pred['scores'], pred['labels'])
                        if s >= args.score_thr]
        gt = dataset[idx]['vectors'].data
        gt_vectors = [(int(l), np.array(v, dtype=np.float64)) for l, vs in gt.items() for v in vs]

        tiles = {}
        for c, cam in enumerate(cam_names):
            img = cv2.imread(sample['cams'][cam]['img_fpath'])
            scale = args.cam_w / img.shape[1]
            img = cv2.resize(img, (args.cam_w, int(round(img.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
            img = draw_on_cam(img, pred_vectors, np.array(input_dict['ego2img'][c]), scale)
            cv2.putText(img, cam, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            tiles[cam] = img
        cams = np.concatenate([np.concatenate([tiles[c] for c in row], axis=1) for row in CAM_GRID],
                              axis=0)
        cams = with_title(cams, f'{args.scene}  frame {n + 1}/{len(idxs)}  '
                                f'prediction projected (score >= {args.score_thr})')

        pred_bev = bev_panel(pred_vectors, roi_size, args.px_per_m, 'BEV prediction')
        gt_bev = bev_panel(gt_vectors, roi_size, args.px_per_m, 'BEV ground truth')
        h = max(cams.shape[0], pred_bev.shape[0])
        pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 0), (0, 0)), constant_values=255)
        frame = np.concatenate([pad(cams), pad(pred_bev), pad(gt_bev)], axis=1)
        frame = np.concatenate([frame, legend(frame.shape[1])], axis=0)
        # even size for the h264 encoder
        frame = frame[:frame.shape[0] // 2 * 2, :frame.shape[1] // 2 * 2]
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    imageio.mimsave(args.out, frames, format='MP4', fps=args.fps)
    cv2.imwrite(args.out.replace('.mp4', '_frame0.jpg'), cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
    print(f'saved {len(frames)} frames to {args.out}')


if __name__ == '__main__':
    main()
