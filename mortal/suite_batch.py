"""Batch decision-point benchmark scoring across multiple checkpoints.

Scores a list of checkpoints on the same held-out decision points
(suite.score()) and writes the results to a CSV (and optionally
TensorBoard), one row per checkpoint. The main use case is decomposing
suite/sp metrics across the offline-to-online transition period
(roadmap Phase A step 3): scoring the full run of best/top-k/pre- and
post-online checkpoints in one pass to see where behavior shifts.

Requires the same config.toml keys as suite.py: [control] version,
[dataset] holdout_files / player_names_files, [grp] (for reward
labels via FileDatasetsIter), and [suite] (batch_size / file_batch_size
/ sp_metrics / max_files).

Usage:
    python suite_batch.py ckpt1.pth ckpt2.pth --glob 'runs/**/*.pth' --out results.csv
"""
import prelude

import argparse
import csv
import glob as globmod
import logging
import os
import torch
import suite
from engine import build_engine_from_state
from checkpoint import load_checkpoint
from config import config

def collect_states(states, pattern):
    paths = list(states)
    if pattern:
        paths += globmod.glob(pattern, recursive=True)
    seen = set()
    result = []
    for p in paths:
        key = os.path.normcase(os.path.normpath(p))
        if key in seen:
            continue
        seen.add(key)
        result.append(p)
    return result

def score_one(path, device, max_files):
    state = load_checkpoint(path)
    steps = state.get('steps', 0)
    timestamp = state.get('timestamp', '')
    logging.info(f'scoring {path} (steps={steps})')

    engine, info = build_engine_from_state(state, device=device, name='suite-batch')
    try:
        metrics, counts = suite.score(engine.brain, engine.dqn, device, max_files=max_files)
    finally:
        del engine
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return steps, timestamp, metrics, counts

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('states', nargs='*')
    parser.add_argument('--glob', default='')
    parser.add_argument('--out', default='suite_batch_results.csv')
    parser.add_argument('--tb', default='')
    parser.add_argument('--files', type=int, default=0)
    parser.add_argument('--device', default='')
    parser.add_argument('--sp', dest='sp', action='store_true', default=None)
    parser.add_argument('--no-sp', dest='sp', action='store_false')
    args = parser.parse_args()

    if args.sp is not None:
        config.setdefault('suite', {})['sp_metrics'] = args.sp

    device = torch.device(args.device) if args.device else torch.device(config['control']['device'])
    paths = collect_states(args.states, args.glob)

    results = []
    for path in paths:
        try:
            steps, timestamp, metrics, counts = score_one(path, device, args.files)
        except Exception:
            logging.warning(f'failed to score {path}, skipping', exc_info=True)
            continue

        nan = float('nan')
        msg = (
            'suite-batch: '
            f"{path} steps={steps} "
            f"shanten_regression_rate={metrics.get('suite/efficiency/shanten_regression_rate', nan):.4f}, "
            f"advance_miss_rate={metrics.get('suite/efficiency/advance_miss_rate', nan):.4f}, "
            f"human_top1={metrics.get('suite/agreement/human_top1', nan):.4f}, "
            "none_rate_given_call_opp_vs_riichi="
            f"{metrics.get('suite/behavior/none_rate_given_call_opp_vs_riichi', nan):.4f}"
        )
        if 'suite/sp/model_match_rate' in metrics:
            msg += (
                f", sp_match={metrics['suite/sp/model_match_rate']:.4f}"
                f", sp_ev_loss={metrics.get('suite/sp/model_ev_loss', nan):.1f}"
            )
        logging.info(msg)

        results.append({
            'path': path,
            'steps': steps,
            'timestamp': timestamp,
            'metrics': metrics,
        })

    if not results:
        logging.warning('no checkpoints scored successfully')
        return

    results.sort(key=lambda r: r['steps'])

    all_tags = sorted({tag for r in results for tag in r['metrics']})

    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'steps', 'timestamp'] + all_tags)
        for r in results:
            row = [r['path'], r['steps'], r['timestamp']]
            row += [r['metrics'].get(tag, '') for tag in all_tags]
            writer.writerow(row)
    logging.info(f'wrote {args.out} ({len(results)} checkpoints)')

    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tb)
        for r in results:
            for tag, value in r['metrics'].items():
                writer.add_scalar(tag, value, r['steps'])
        writer.flush()

    header = 'metric'.ljust(55) + ''.join(f'{r["steps"] / 1000:>10.0f}k' for r in results)
    print(header)
    for tag in all_tags:
        line = tag.ljust(55)
        for r in results:
            value = r['metrics'].get(tag)
            line += f'{value:>11.4f}' if value is not None else f'{"-":>11}'
        print(line)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
