import prelude

import numpy as np
import torch
import secrets
import os
from engine import build_engine_from_state
from libriichi.arena import OneVsThree
from config import config

def main():
    cfg = config['1v3']
    games_per_iter = cfg['games_per_iter']
    seeds_per_iter = games_per_iter // 4
    iters = cfg['iters']
    log_dir = cfg['log_dir']
    use_akochan = cfg['akochan']['enabled']

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    if use_akochan:
        os.environ['AKOCHAN_DIR'] = cfg['akochan']['dir']
        os.environ['AKOCHAN_TACTICS'] = cfg['akochan']['tactics']
    else:
        state = torch.load(cfg['champion']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        engine_cham, cham_info = build_engine_from_state(
            state,
            device=torch.device(cfg['champion']['device']),
            enable_compile=cfg['champion']['enable_compile'],
            enable_amp=cfg['champion']['enable_amp'],
            enable_rule_based_agari_guard=cfg['champion']['enable_rule_based_agari_guard'],
            name=cfg['champion']['name'],
        )
        print(f"loaded champion ({cham_info['head_kind']}, norm={cham_info['norm']})")

    state = torch.load(cfg['challenger']['state_file'], weights_only=True, map_location=torch.device('cpu'))
    engine_chal, chal_info = build_engine_from_state(
        state,
        device=torch.device(cfg['challenger']['device']),
        enable_compile=cfg['challenger']['enable_compile'],
        enable_amp=cfg['challenger']['enable_amp'],
        enable_rule_based_agari_guard=cfg['challenger']['enable_rule_based_agari_guard'],
        name=cfg['challenger']['name'],
    )
    print(f"loaded challenger ({chal_info['head_kind']}, norm={chal_info['norm']})")

    seed_start = 10000
    for i, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
        print('-' * 50)
        print('#', i)
        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = log_dir,
        )
        if use_akochan:
            rankings = env.ako_vs_py(
                engine = engine_chal,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        else:
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = engine_cham,
                seed_start = (seed, key),
                seed_count = seeds_per_iter,
            )
        rankings = np.array(rankings)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
        print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
