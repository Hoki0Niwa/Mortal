import torch
import numpy as np
import os
import shutil
import secrets
import logging
import random
from os import path
from engine import MortalEngine, resolve_amp_dtype, build_engine_from_state
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=False, map_location=torch.device('cpu'))
        self.baseline_engine, info = build_engine_from_state(
            state,
            device=device,
            enable_compile=baseline_cfg['enable_compile'],
            name='baseline',
        )
        logging.info(f"loaded test baseline ({info['head_kind']}, norm={info['norm']})")

        self.chal_version = config['control']['version']
        self.chal_amp_dtype = resolve_amp_dtype(config['control'])
        self.log_dir = path.abspath(config['test_play']['log_dir'])
        self.aggregate_runs = max(1, int(config['test_play'].get('aggregate_runs', 1)))

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        try:
            engine_chal = MortalEngine(
                mortal,
                dqn,
                is_oracle = False,
                version = self.chal_version,
                device = device,
                enable_amp = True,
                amp_dtype = self.chal_amp_dtype,
                name = 'mortal',
            )

            if path.isdir(self.log_dir):
                shutil.rmtree(self.log_dir)

            if self.aggregate_runs == 1:
                run_log_dirs = [self.log_dir]
                stat_dir = self.log_dir
            else:
                os.makedirs(self.log_dir, exist_ok=True)
                run_log_dirs = [
                    path.join(self.log_dir, f'run_{idx:04d}')
                    for idx in range(self.aggregate_runs)
                ]
                stat_dir = self.log_dir

            for idx, run_log_dir in enumerate(run_log_dirs):
                env = OneVsThree(
                    disable_progress_bar = False,
                    log_dir = run_log_dir,
                )
                seed_start = 10000 + idx * seed_count
                env.py_vs_py(
                    challenger = engine_chal,
                    champion = self.baseline_engine,
                    seed_start = (seed_start, 0x2000),
                    seed_count = seed_count,
                )

            stat = Stat.from_dir(stat_dir, 'mortal')
            if self.aggregate_runs > 1:
                logging.info(
                    f'aggregated {self.aggregate_runs} test-play runs '
                    f'({stat.game:,} games)'
                )
            return stat
        finally:
            torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        device = torch.device(baseline_cfg['device'])

        state = torch.load(baseline_cfg['state_file'], weights_only=False, map_location=torch.device('cpu'))
        self.baseline_engine, info = build_engine_from_state(
            state,
            device=device,
            enable_compile=baseline_cfg['enable_compile'],
            name='baseline',
        )
        logging.info(f"loaded train baseline ({info['head_kind']}, norm={info['norm']})")

        profile = os.environ.get('TRAIN_PLAY_PROFILE', 'default')
        logging.info(f'using profile {profile}')
        cfg = config['train_play'][profile]
        self.chal_version = config['control']['version']
        self.chal_amp_dtype = resolve_amp_dtype(config['control'])
        self.log_dir = path.abspath(cfg['log_dir'])
        self.train_key = secrets.randbits(64)
        self.train_seed = 10000

        self.seed_count = cfg['games'] // 4
        self.boltzmann_epsilon = cfg['boltzmann_epsilon']
        self.boltzmann_temp = cfg['boltzmann_temp']
        self.top_p = cfg['top_p']

        self.repeats = cfg['repeats']
        self.repeat_counter = 0

        self.opponent_pool_dir = cfg.get('opponent_pool_dir', '')
        self.opponent_pool_prob = cfg.get('opponent_pool_prob', 0.0)

    def _load_pool_opponent(self, device):
        pool_dir = self.opponent_pool_dir
        if not pool_dir or not path.isdir(pool_dir):
            return None
        candidates = [f for f in os.listdir(pool_dir) if f.endswith('.pth')]
        if not candidates:
            return None
        chosen = path.join(pool_dir, random.choice(candidates))
        try:
            state = torch.load(chosen, weights_only=False, map_location=torch.device('cpu'))
            engine, info = build_engine_from_state(state, device=device, name='pool')
            engine.name = f"pool-{info['head_kind']}-{info['norm']}"
            logging.info(
                f"using pool opponent ({info['head_kind']}, norm={info['norm']}): "
                f"{path.basename(chosen)}"
            )
            return engine
        except Exception as e:
            logging.warning(f'failed to load pool opponent {chosen}: {e}')
            return None

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = False
        engine_chal = MortalEngine(
            mortal,
            dqn,
            is_oracle = False,
            version = self.chal_version,
            boltzmann_epsilon = self.boltzmann_epsilon,
            boltzmann_temp = self.boltzmann_temp,
            top_p = self.top_p,
            device = device,
            enable_amp = True,
            amp_dtype = self.chal_amp_dtype,
            name = 'trainee',
        )

        opponent = self.baseline_engine
        if self.opponent_pool_prob > 0 and random.random() < self.opponent_pool_prob:
            pool_opponent = self._load_pool_opponent(device)
            if pool_opponent is not None:
                opponent = pool_opponent

        if path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        env = OneVsThree(
            disable_progress_bar = False,
            log_dir = self.log_dir,
        )
        rankings = env.py_vs_py(
            challenger = engine_chal,
            champion = opponent,
            seed_start = (self.train_seed, self.train_key),
            seed_count = self.seed_count,
        )
        self.repeat_counter += 1
        if self.repeat_counter == self.repeats:
            self.train_seed += self.seed_count
            self.repeat_counter = 0

        rankings = np.array(rankings)
        file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))

        torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
        return rankings, file_list
