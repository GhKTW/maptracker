"""
Re-evaluate saved submissions (work_dirs/<run>/submission_vector.json) on more regions:
the whole ROI, the ahead / rear halves and their near (|x| < 15 m) / far distance bands,
with the per-threshold APs. CPU only, no inference.

usage: python tools/eval_regions.py CONFIG OUT_JSON RUN [RUN ...]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmcv import Config

REGIONS = [None, 'ahead', 'rear', 'ahead_near', 'ahead_far', 'rear_near', 'rear_far']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='a config with the test dataset (data.test.eval_config)')
    parser.add_argument('out', help='output json, results keyed by run name (merged if it exists)')
    parser.add_argument('runs', nargs='+', help='run names under work_dirs/ or paths to submission files')
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    import plugin  # noqa: F401, registers the datasets
    from plugin.datasets.evaluation.vector_eval import VectorEvaluate
    evaluator = VectorEvaluate(cfg.data.test.eval_config, n_workers=args.workers, regions=REGIONS)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for run in args.runs:
        path = run if run.endswith('.json') else f'work_dirs/{run}/submission_vector.json'
        name = os.path.basename(os.path.dirname(path)) if run.endswith('.json') else run
        print(f'==== {name}: {path}', flush=True)
        results[name] = {k: float(v) for k, v in evaluator.evaluate(path).items()}
        json.dump(results, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
