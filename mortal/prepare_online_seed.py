"""Prepare an immutable offline checkpoint for a fresh online run.

The output intentionally remains a legacy single-head checkpoint.  On first
load, train.py replicates that DQN into every bootstrap head and samples the
fixed randomized priors.  The original file is never modified.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_state(state, source_name, source_digest):
    required = {'mortal', 'current_dqn', 'aux_net', 'config'}
    missing = required - state.keys()
    if missing:
        raise ValueError(f'checkpoint is missing keys: {sorted(missing)}')
    if state['config']['control'].get('online', False):
        raise ValueError('source is already an online checkpoint; use an offline/main model')
    state['steps'] = 0
    state['timestamp'] = datetime.now().timestamp()
    state['best_perf'] = {'avg_rank': 4.0, 'avg_pt': -135.0}
    state['top_checkpoints'] = []
    state['online_seed'] = {
        'source_name': source_name,
        'source_sha256': source_digest,
    }
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path, help='offline/main checkpoint')
    parser.add_argument('output', type=Path, help='working checkpoint for the online run')
    parser.add_argument('--force', action='store_true', help='replace an existing output')
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        parser.error('source and output must be different; the seed is immutable')
    if not source.is_file():
        parser.error(f'source checkpoint does not exist: {source}')
    if output.exists() and not args.force:
        parser.error(f'output already exists: {output} (use --force to replace it)')

    state = torch.load(source, weights_only=True, map_location=torch.device('cpu'))
    source_digest = sha256(source)
    try:
        prepare_state(state, source.name, source_digest)
    except ValueError as e:
        parser.error(str(e))

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, output)

    manifest = {
        'created_at': datetime.now().astimezone().isoformat(),
        'source': str(source),
        'source_sha256': source_digest,
        'output': str(output),
        'output_sha256': sha256(output),
    }
    manifest_path = output.with_suffix(output.suffix + '.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
