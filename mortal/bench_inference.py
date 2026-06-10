"""Micro-benchmark for MortalEngine inference throughput.

Measures react_batch samples/sec across batch sizes for combinations of
{autocast / bf16 weights} x {cudnn benchmark on/off} x {CUDA graphs on/off},
and reports the action (argmax) agreement rate of each variant against an
fp32 reference. Use this to decide which [inference] options to enable in
config.toml without running full games.

Usage:
    python bench_inference.py [--state PATH] [--batches 16,64,128,256,512]
"""
import prelude

import argparse
import logging
import time
import numpy as np
import torch
from engine import build_engine_from_state
from libriichi.consts import obs_shape, ACTION_SPACE
from config import config

WARMUP = 3
REPS = 10
AGREEMENT_SAMPLES = 2048

def make_inputs(rng, count, channels):
    # random binary planes are not real game states, but conv cost is
    # shape-bound so throughput numbers carry over; agreement is indicative
    obs = [(rng.random((channels, 34)) < 0.1).astype(np.float32) for _ in range(count)]
    masks = []
    for _ in range(count):
        mask = np.zeros(ACTION_SPACE, dtype=bool)
        mask[rng.choice(37, size=14, replace=False)] = True
        mask[45] = True
        masks.append(mask)
    return obs, masks

def build_variant(state, device, name, **kwargs):
    engine, _ = build_engine_from_state(state, device=device, name=name, **kwargs)
    return engine

def bench_engine(engine, obs, masks, batch_sizes):
    results = {}
    for bs in batch_sizes:
        obs_b, masks_b = obs[:bs], masks[:bs]
        for _ in range(WARMUP):
            engine.react_batch(obs_b, masks_b, None)
        t0 = time.perf_counter()
        for _ in range(REPS):
            engine.react_batch(obs_b, masks_b, None)
        dt = time.perf_counter() - t0
        results[bs] = bs * REPS / dt
    return results

def actions_of(engine, obs, masks, chunk=256):
    actions = []
    for i in range(0, len(obs), chunk):
        a, _, _, _ = engine.react_batch(obs[i:i+chunk], masks[i:i+chunk], None)
        actions.extend(a)
    return np.array(actions)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', default=config['control']['best_state_file'])
    parser.add_argument('--batches', default='16,64,128,256,512')
    args = parser.parse_args()
    batch_sizes = [int(b) for b in args.batches.split(',')]

    device = torch.device(config['control']['device'])
    assert device.type == 'cuda', 'this benchmark targets CUDA devices'
    state = torch.load(args.state, weights_only=False, map_location=torch.device('cpu'))
    version = state['config']['control'].get('version', 1)
    channels = obs_shape(version)[0]
    logging.info(f'state: {args.state} (v{version}, {channels} channels)')
    logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')

    rng = np.random.default_rng(1)
    obs, masks = make_inputs(rng, max(max(batch_sizes), AGREEMENT_SAMPLES), channels)

    infer_cfg = config.get('inference', {})
    buckets = infer_cfg.get('buckets', [64, 128, 256, 512, 1024, 2048])
    graph_max_batch = infer_cfg.get('graph_max_batch', 512)
    variants = [
        # (label, cudnn_benchmark, engine kwargs)
        ('fp32 reference', False, dict(enable_amp=False)),
        ('autocast (current)', False, dict(enable_amp=True)),
        ('autocast +benchmark +bucket', True, dict(enable_amp=True, bucket_batch=True, buckets=buckets)),
        ('autocast +benchmark +bucket +graphs', True, dict(
            enable_amp=True, bucket_batch=True, buckets=buckets,
            use_cuda_graphs=True, graph_max_batch=graph_max_batch,
        )),
        ('fp32 +benchmark +bucket +graphs', True, dict(
            enable_amp=False, bucket_batch=True, buckets=buckets,
            use_cuda_graphs=True, graph_max_batch=graph_max_batch,
        )),
        ('bf16 weights +benchmark +bucket', True, dict(weights_dtype='bfloat16', bucket_batch=True, buckets=buckets)),
        ('bf16 weights +benchmark +bucket +graphs', True, dict(
            weights_dtype='bfloat16', bucket_batch=True, buckets=buckets,
            use_cuda_graphs=True, graph_max_batch=graph_max_batch,
        )),
    ]

    ref_actions = None
    rows = []
    for label, benchmark, kwargs in variants:
        torch.backends.cudnn.benchmark = benchmark
        engine = build_variant(state, device, label, **kwargs)
        throughput = bench_engine(engine, obs, masks, batch_sizes)
        actions = actions_of(engine, obs[:AGREEMENT_SAMPLES], masks[:AGREEMENT_SAMPLES])
        if ref_actions is None:
            ref_actions = actions
            agreement = 1.0
        else:
            agreement = float((actions == ref_actions).mean())
        rows.append((label, throughput, agreement))
        del engine
        torch.cuda.empty_cache()

    header = f'{"variant":<42}' + ''.join(f'{f"bs={bs}":>12}' for bs in batch_sizes) + f'{"agree":>9}'
    print()
    print(header)
    print('-' * len(header))
    for label, throughput, agreement in rows:
        cells = ''.join(f'{throughput[bs]:>12,.0f}' for bs in batch_sizes)
        print(f'{label:<42}{cells}{agreement:>9.4f}')
    print()
    print('throughput is react_batch samples/sec; agree is argmax agreement vs fp32')
    print('(>= 0.995 is the suggested bar for enabling a variant in config.toml)')

if __name__ == '__main__':
    main()
