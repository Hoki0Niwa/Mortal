import torch
import numpy as np
import os
import shutil
import secrets
import logging
import random
import time
from os import path
from engine import MortalEngine, resolve_amp_dtype, build_engine_from_state
from checkpoint import load_checkpoint
from libriichi.stat import Stat
from libriichi.arena import OneVsThree
from config import config

INFER_CFG = config.get('inference', {})

def perf_engine_kwargs(allow_weights_cast):
    # weights_dtype casts the modules in place, so it must stay disabled for
    # engines that wrap live training models
    kwargs = {
        'bucket_batch': INFER_CFG.get('bucket_batch', False),
        'buckets': INFER_CFG.get('buckets'),
        'use_cuda_graphs': INFER_CFG.get('use_cuda_graphs', False),
        'graph_max_batch': INFER_CFG.get('graph_max_batch', 512),
    }
    if allow_weights_cast:
        kwargs['weights_dtype'] = INFER_CFG.get('weights_dtype')
    return kwargs

def play_cudnn_benchmark():
    return INFER_CFG.get('cudnn_benchmark_in_play', False)

def log_react_stats(*engines):
    for engine in engines:
        stats = engine.take_react_stats()
        if stats['calls'] == 0:
            continue
        logging.info(
            f"react stats [{engine.name}]: calls={stats['calls']:,}, "
            f"samples={stats['samples']:,}, avg_batch={stats['avg_batch']:.1f}, "
            f"max_batch={stats['max_batch']}"
        )

class TestPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['test']
        device = torch.device(baseline_cfg['device'])

        state = load_checkpoint(baseline_cfg['state_file'])
        self.baseline_engine, info = build_engine_from_state(
            state,
            device=device,
            enable_compile=baseline_cfg['enable_compile'],
            name='baseline',
            **perf_engine_kwargs(allow_weights_cast=True),
        )
        logging.info(f"loaded test baseline ({info['head_kind']}, norm={info['norm']})")

        self.chal_version = config['control']['version']
        self.chal_amp_dtype = resolve_amp_dtype(config['control'])
        self.log_dir = path.abspath(config['test_play']['log_dir'])
        self.aggregate_runs = max(1, int(config['test_play'].get('aggregate_runs', 1)))

    def test_play(self, seed_count, mortal, dqn, device):
        torch.backends.cudnn.benchmark = play_cudnn_benchmark()
        t_start = time.monotonic()
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
                # the challenger wraps the live training models, so weight
                # casting must stay off here
                **perf_engine_kwargs(allow_weights_cast=False),
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
            test_secs = time.monotonic() - t_start
            total_games = seed_count * 4 * self.aggregate_runs
            logging.info(
                f'test_play: {total_games:,} games in {test_secs:.1f}s '
                f'({total_games / test_secs * 3600:,.0f} games/h)'
            )
            log_react_stats(engine_chal, self.baseline_engine)
            return stat
        finally:
            torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']

class TrainPlayer:
    def __init__(self):
        baseline_cfg = config['baseline']['train']
        device = torch.device(baseline_cfg['device'])

        state = load_checkpoint(baseline_cfg['state_file'])
        self.baseline_engine, info = build_engine_from_state(
            state,
            device=device,
            enable_compile=baseline_cfg['enable_compile'],
            name='baseline',
            **perf_engine_kwargs(allow_weights_cast=True),
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
            t_load = time.monotonic()
            state = load_checkpoint(chosen)
            engine, info = build_engine_from_state(
                state,
                device=device,
                name='pool',
                **perf_engine_kwargs(allow_weights_cast=True),
            )
            engine.name = f"pool-{info['head_kind']}-{info['norm']}"
            logging.info(
                f"using pool opponent ({info['head_kind']}, norm={info['norm']}): "
                f"{path.basename(chosen)} (loaded in {time.monotonic() - t_load:.1f}s)"
            )
            return engine
        except Exception as e:
            logging.warning(f'failed to load pool opponent {chosen}: {e}')
            return None

    def train_play(self, mortal, dqn, device):
        torch.backends.cudnn.benchmark = play_cudnn_benchmark()
        try:
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
                # the client's models are local inference copies, so weight
                # casting is safe here (load_state_dict re-casts on each update)
                **perf_engine_kwargs(allow_weights_cast=True),
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
            t_play = time.monotonic()
            rankings = env.py_vs_py(
                challenger = engine_chal,
                champion = opponent,
                seed_start = (self.train_seed, self.train_key),
                seed_count = self.seed_count,
            )
            play_secs = time.monotonic() - t_play
            games = self.seed_count * 4
            logging.info(
                f'train_play: {games:,} games in {play_secs:.1f}s '
                f'({games / play_secs * 3600:,.0f} games/h)'
            )
            log_react_stats(engine_chal, opponent)
            self.repeat_counter += 1
            if self.repeat_counter == self.repeats:
                self.train_seed += self.seed_count
                self.repeat_counter = 0

            rankings = np.array(rankings)
            file_list = list(map(lambda p: path.join(self.log_dir, p), os.listdir(self.log_dir)))
            return rankings, file_list
        finally:
            torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
