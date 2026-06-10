def train():
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import time
    import threading
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libriichi.consts import obs_shape
    from config import config

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    test_games = config['test_play']['games']
    min_q_weight = config['cql']['min_q_weight']
    next_rank_weight = config['aux']['next_rank_weight']
    metrics_every = config['control'].get('metrics_every', 1)
    async_save = config['control'].get('async_save', False)
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    amp_dtype_str = config['control'].get('amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    gamma = config['env']['gamma']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    # GradScaler is not needed for bfloat16 (wider dynamic range)
    scaler = GradScaler(device.type, enabled=enable_amp and amp_dtype == torch.float16)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    def state_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to('cpu', copy=True)
        if isinstance(obj, dict):
            return {k: state_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(state_to_cpu(v) for v in obj)
        return obj

    save_thread = None
    def queue_save(state_cpu, filepath):
        # single in-flight write; the snapshot is taken on the main thread so
        # the background thread only does serialization and disk I/O
        nonlocal save_thread
        join_save()
        def _save():
            tmp = filepath + '.tmp'
            torch.save(state_cpu, tmp)
            os.replace(tmp, filepath)
        save_thread = threading.Thread(target=_save, daemon=True)
        save_thread.start()

    def join_save():
        nonlocal save_thread
        if save_thread is not None:
            save_thread.join()
            save_thread = None

    perf = {'block_t0': time.monotonic()}

    steps = 0
    state_file = config['control']['state_file']
    best_state_file = config['control']['best_state_file']
    top_k = config['control'].get('top_k', 5)
    top_k_dir = path.join(path.dirname(best_state_file), 'candidates')
    top_checkpoints = []  # list of {'avg_rank', 'avg_pt', 'steps', 'filepath'}
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']
        if 'top_checkpoints' in state:
            top_checkpoints = [c for c in state['top_checkpoints'] if path.exists(c['filepath'])]

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    metrics = {
        'train/reward_mean': 0,
        'train/reward_std': 0,
        'train/reward_positive_rate': 0,
        'train/reward_negative_rate': 0,
        'train/steps_to_done_mean': 0,
        'train/turn_mean': 0,
        'train/shanten/tenpai_or_better_rate': 0,
        'train/shanten/one_shanten_rate': 0,
        'train/shanten/two_shanten_rate': 0,
        'train/shanten/three_plus_shanten_rate': 0,
        'train/legal_action_count_mean': 0,
        'train/legal_action_count_max': 0,
        'train/q_best_mean': 0,
        'train/q_legal_mean': 0,
        'train/q_legal_std': 0,
        'train/q_legal_min': 0,
        'train/q_legal_max': 0,
        'train/q_logged_gap_to_best': 0,
        'train/q_top1_top2_margin': 0,
        'train/logged_action_argmax_rate': 0,
        'train/policy_entropy': 0,
        'train/policy_entropy_norm': 0,
        'train/action_rate/discard': 0,
        'train/action_rate/riichi': 0,
        'train/action_rate/chi': 0,
        'train/action_rate/pon': 0,
        'train/action_rate/kan': 0,
        'train/action_rate/agari': 0,
        'train/action_rate/ryukyoku': 0,
        'train/action_rate/none': 0,
        'train/agari/agari_opportunity_rate': 0,
        'train/agari/agari_taken_given_opportunity': 0,
        'train/agari/agari_argmax_given_opportunity': 0,
        'train/agari/q_agari_minus_best_non_agari': 0,
        'train/threat/opp_riichi_rate': 0,
        'train/threat/opp_fuuro_rate': 0,
        'train/threat/opp_fuuro_no_riichi_rate': 0,
        'train/threat/opp_riichi_and_fuuro_rate': 0,
        'train/threat/any_threat_rate': 0,
        'train/threat/none_rate_vs_no_threat': 0,
        'train/threat/none_rate_vs_riichi': 0,
        'train/threat/none_rate_vs_fuuro_no_riichi': 0,
        'train/threat/call_rate_vs_no_threat': 0,
        'train/threat/call_rate_vs_riichi': 0,
        'train/threat/call_rate_vs_fuuro_no_riichi': 0,
        'train/threat/riichi_rate_vs_no_threat': 0,
        'train/threat/riichi_rate_vs_riichi': 0,
        'train/threat/riichi_rate_vs_fuuro_no_riichi': 0,
        'train/threat/agari_rate_vs_no_threat': 0,
        'train/threat/agari_rate_vs_riichi': 0,
        'train/threat/agari_rate_vs_fuuro_no_riichi': 0,
        'train/threat/one_shanten_none_rate_vs_riichi': 0,
        'train/threat/one_shanten_call_rate_vs_riichi': 0,
        'train/threat/one_shanten_none_rate_vs_fuuro_no_riichi': 0,
        'train/threat/one_shanten_call_rate_vs_fuuro_no_riichi': 0,
        'train/threat/tenpai_none_rate_vs_riichi': 0,
        'train/threat/tenpai_riichi_rate_vs_riichi': 0,
        'train/threat/tenpai_none_rate_vs_fuuro_no_riichi': 0,
        'train/threat/tenpai_riichi_rate_vs_fuuro_no_riichi': 0,
        'train/threat/call_taken_given_riichi': 0,
        'train/threat/call_taken_given_fuuro_no_riichi': 0,
        'train/threat/none_argmax_given_call_opp_vs_riichi': 0,
        'train/threat/none_argmax_given_call_opp_vs_fuuro_no_riichi': 0,
        'train/threat/q_best_call_minus_none_vs_riichi': 0,
        'train/threat/q_best_call_minus_none_vs_fuuro_no_riichi': 0,
        'train/threat/riichi_taken_given_riichi': 0,
        'train/threat/riichi_taken_given_fuuro_no_riichi': 0,
        'train/threat/q_riichi_minus_best_non_riichi_vs_riichi': 0,
        'train/threat/q_riichi_minus_best_non_riichi_vs_fuuro_no_riichi': 0,
        'train/call/call_opportunity_rate': 0,
        'train/call/chi_opportunity_rate': 0,
        'train/call/pon_opportunity_rate': 0,
        'train/call/kan_opportunity_rate': 0,
        'train/call/call_taken_given_opportunity': 0,
        'train/call/call_argmax_given_opportunity': 0,
        'train/call/none_argmax_given_opportunity': 0,
        'train/call/q_best_call_minus_none': 0,
        'train/call/q_chi_best_minus_none': 0,
        'train/call/q_pon_minus_none': 0,
        'train/call/q_kan_minus_none': 0,
        'train/by_shanten/tenpai_or_better/call_rate': 0,
        'train/by_shanten/tenpai_or_better/riichi_rate': 0,
        'train/by_shanten/tenpai_or_better/none_rate': 0,
        'train/by_shanten/one_shanten/call_rate': 0,
        'train/by_shanten/one_shanten/none_rate': 0,
        'train/by_shanten/two_plus_shanten/call_rate': 0,
        'train/by_shanten/two_plus_shanten/none_rate': 0,
        'train/rank_rate/1st': 0,
        'train/rank_rate/2nd': 0,
        'train/rank_rate/3rd': 0,
        'train/rank_rate/4th': 0,
    }
    opt_metrics = {
        'count': 0,
        'grad/total_norm': 0,
        'grad/mortal_norm': 0,
        'grad/dqn_norm': 0,
        'grad/aux_norm': 0,
        'grad/post_clip_total_norm': 0,
    }
    rank_metric_tags = (
        'train/rank_rate/1st',
        'train/rank_rate/2nd',
        'train/rank_rate/3rd',
        'train/rank_rate/4th',
    )
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0
    metrics_batches = 0

    def to_scalar(value):
        if torch.is_tensor(value):
            return value.detach().float().cpu().item()
        return float(value)

    def module_param_norm(module):
        total = torch.zeros((), device=device)
        with torch.inference_mode():
            for param in module.parameters():
                total += param.detach().float().square().sum()
        return total.sqrt()

    def module_grad_norm(module):
        total = torch.zeros((), device=device)
        for param in module.parameters():
            if param.grad is not None:
                total += param.grad.detach().float().square().sum()
        return total.sqrt()

    def extract_threats(obs):
        if version != 4:
            return None
        # v4 obs channel layout from libriichi/src/state/obs_repr.rs.
        opp1_fuuro_start = 771
        opp_fuuro_rows = obs[:, opp1_fuuro_start:opp1_fuuro_start + 60].reshape(-1, 3, 4, 5, 34)
        opp_fuuro_players = opp_fuuro_rows.amax(dim=(-1, -2, -3)) > 0
        opp_fuuro = opp_fuuro_players.any(-1)

        opp_riichi_accepted_start = 857
        opp_riichi_rows = obs[:, opp_riichi_accepted_start:opp_riichi_accepted_start + 3]
        opp_riichi_players = opp_riichi_rows.amax(-1) > 0
        opp_riichi = opp_riichi_players.any(-1)

        return {
            'riichi': opp_riichi,
            'fuuro': opp_fuuro,
            'fuuro_no_riichi': opp_fuuro & ~opp_riichi,
            'riichi_and_fuuro': opp_riichi & opp_fuuro,
            'any': opp_riichi | opp_fuuro,
        }

    def train_epoch():
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            player_names = ['trainee']
            t_drain = time.monotonic()
            dirname = drain()
            logging.info(f'drain wait: {time.monotonic() - t_drain:.1f}s')
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        ))

        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_at_turns = []
        remaining_shantens = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, at_turns, shantens):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal metrics_batches

            # non_blocking pairs with the DataLoader's pin_memory to overlap
            # H2D transfers with compute; results are identical
            obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
            actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
            masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device, non_blocking=True)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)
            at_turns = at_turns.to(dtype=torch.float32, device=device, non_blocking=True)
            shantens = shantens.to(dtype=torch.int64, device=device, non_blocking=True)
            detailed_metrics = metrics_every <= 1 or steps % metrics_every == 0
            if detailed_metrics:
                assert masks[range(batch_size), actions].all()

            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, dtype=amp_dtype, enabled=enable_amp):
                phi = mortal(obs)
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                dqn_loss = 0.5 * mse(q, q_target_mc)
                cql_loss = 0
                if not online:
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks)

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                ))
            scaler.scale(loss / opt_step_every).backward()

            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

                if detailed_metrics:
                    metrics_batches += 1
                    q_metric = q.float()
                    q_out_metric = q_out.float()
                    reward = kyoku_rewards.to(torch.float32)
                    legal_action_count = masks.sum(-1).to(torch.float32)
                    q_best, q_best_action = q_out_metric.max(-1)
                    legal_q = q_out_metric.masked_select(masks)
                    q_top2 = q_out_metric.topk(k=2, dim=-1).values
                    q_margin = q_top2[:, 0] - q_top2[:, 1]
                    q_margin = q_margin[torch.isfinite(q_margin)]

                    probs = q_out_metric.softmax(-1)
                    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1)
                    max_entropy = legal_action_count.log()
                    entropy_norm = torch.where(
                        legal_action_count > 1,
                        entropy / max_entropy.clamp_min(1e-8),
                        torch.zeros_like(entropy),
                    )

                    metrics['train/reward_mean'] += reward.mean()
                    metrics['train/reward_std'] += reward.std(unbiased=False)
                    metrics['train/reward_positive_rate'] += (reward > 0).to(torch.float32).mean()
                    metrics['train/reward_negative_rate'] += (reward < 0).to(torch.float32).mean()
                    metrics['train/steps_to_done_mean'] += steps_to_done.to(torch.float32).mean()
                    metrics['train/turn_mean'] += at_turns.mean()
                    metrics['train/shanten/tenpai_or_better_rate'] += (shantens <= 0).to(torch.float32).mean()
                    metrics['train/shanten/one_shanten_rate'] += (shantens == 1).to(torch.float32).mean()
                    metrics['train/shanten/two_shanten_rate'] += (shantens == 2).to(torch.float32).mean()
                    metrics['train/shanten/three_plus_shanten_rate'] += (shantens >= 3).to(torch.float32).mean()
                    metrics['train/legal_action_count_mean'] += legal_action_count.mean()
                    metrics['train/legal_action_count_max'] += legal_action_count.max()
                    metrics['train/q_best_mean'] += q_best.mean()
                    metrics['train/q_legal_mean'] += legal_q.mean()
                    metrics['train/q_legal_std'] += legal_q.std(unbiased=False)
                    metrics['train/q_legal_min'] += legal_q.min()
                    metrics['train/q_legal_max'] += legal_q.max()
                    metrics['train/q_logged_gap_to_best'] += (q_best - q_metric).mean()
                    if q_margin.numel() > 0:
                        metrics['train/q_top1_top2_margin'] += q_margin.mean()
                    metrics['train/logged_action_argmax_rate'] += (q_best_action == actions).to(torch.float32).mean()
                    metrics['train/policy_entropy'] += entropy.mean()
                    metrics['train/policy_entropy_norm'] += entropy_norm.mean()

                    metrics['train/action_rate/discard'] += (actions <= 36).to(torch.float32).mean()
                    metrics['train/action_rate/riichi'] += (actions == 37).to(torch.float32).mean()
                    metrics['train/action_rate/chi'] += ((actions >= 38) & (actions <= 40)).to(torch.float32).mean()
                    metrics['train/action_rate/pon'] += (actions == 41).to(torch.float32).mean()
                    metrics['train/action_rate/kan'] += (actions == 42).to(torch.float32).mean()
                    metrics['train/action_rate/agari'] += (actions == 43).to(torch.float32).mean()
                    metrics['train/action_rate/ryukyoku'] += (actions == 44).to(torch.float32).mean()
                    metrics['train/action_rate/none'] += (actions == 45).to(torch.float32).mean()

                    agari_opp = masks[:, 43]
                    metrics['train/agari/agari_opportunity_rate'] += agari_opp.to(torch.float32).mean()
                    if agari_opp.any():
                        non_agari_q = q_out_metric.masked_fill(~masks, -torch.inf)
                        non_agari_q[:, 43] = -torch.inf
                        best_non_agari_q = non_agari_q.max(-1).values
                        metrics['train/agari/agari_taken_given_opportunity'] += (actions[agari_opp] == 43).to(torch.float32).mean()
                        metrics['train/agari/agari_argmax_given_opportunity'] += (q_best_action[agari_opp] == 43).to(torch.float32).mean()
                        metrics['train/agari/q_agari_minus_best_non_agari'] += (
                            q_out_metric[agari_opp, 43] - best_non_agari_q[agari_opp]
                        ).mean()

                    threats = extract_threats(obs)
                    is_call_action = (actions >= 38) & (actions <= 42)
                    is_none_action = actions == 45
                    is_riichi_action = actions == 37
                    is_agari_action = actions == 43
                    tenpai_or_better = shantens <= 0
                    one_shanten = shantens == 1
                    if threats is not None:
                        opp_riichi = threats['riichi']
                        opp_fuuro = threats['fuuro']
                        opp_fuuro_no_riichi = threats['fuuro_no_riichi']
                        any_threat = threats['any']
                        no_threat = ~any_threat
                        metrics['train/threat/opp_riichi_rate'] += opp_riichi.to(torch.float32).mean()
                        metrics['train/threat/opp_fuuro_rate'] += opp_fuuro.to(torch.float32).mean()
                        metrics['train/threat/opp_fuuro_no_riichi_rate'] += opp_fuuro_no_riichi.to(torch.float32).mean()
                        metrics['train/threat/opp_riichi_and_fuuro_rate'] += threats['riichi_and_fuuro'].to(torch.float32).mean()
                        metrics['train/threat/any_threat_rate'] += any_threat.to(torch.float32).mean()

                        threat_groups = (
                            ('no_threat', no_threat),
                            ('riichi', opp_riichi),
                            ('fuuro_no_riichi', opp_fuuro_no_riichi),
                        )
                        for label, cond in threat_groups:
                            if not cond.any():
                                continue
                            metrics[f'train/threat/none_rate_vs_{label}'] += is_none_action[cond].to(torch.float32).mean()
                            metrics[f'train/threat/call_rate_vs_{label}'] += is_call_action[cond].to(torch.float32).mean()
                            metrics[f'train/threat/riichi_rate_vs_{label}'] += is_riichi_action[cond].to(torch.float32).mean()
                            metrics[f'train/threat/agari_rate_vs_{label}'] += is_agari_action[cond].to(torch.float32).mean()

                        for label, cond in (('riichi', opp_riichi), ('fuuro_no_riichi', opp_fuuro_no_riichi)):
                            one = cond & one_shanten
                            if one.any():
                                metrics[f'train/threat/one_shanten_none_rate_vs_{label}'] += is_none_action[one].to(torch.float32).mean()
                                metrics[f'train/threat/one_shanten_call_rate_vs_{label}'] += is_call_action[one].to(torch.float32).mean()
                            tenpai = cond & tenpai_or_better
                            if tenpai.any():
                                metrics[f'train/threat/tenpai_none_rate_vs_{label}'] += is_none_action[tenpai].to(torch.float32).mean()
                                metrics[f'train/threat/tenpai_riichi_rate_vs_{label}'] += is_riichi_action[tenpai].to(torch.float32).mean()

                    call_legal = masks[:, 38:43].any(-1)
                    call_opp = call_legal & masks[:, 45]
                    metrics['train/call/call_opportunity_rate'] += call_opp.to(torch.float32).mean()
                    if call_opp.any():
                        call_actions = (actions >= 38) & (actions <= 42)
                        call_argmax = (q_best_action >= 38) & (q_best_action <= 42)
                        none_argmax = q_best_action == 45
                        q_none = q_out_metric[:, 45]
                        q_chi = q_out_metric[:, 38:41].masked_fill(~masks[:, 38:41], -torch.inf).max(-1).values
                        q_pon = q_out_metric[:, 41].masked_fill(~masks[:, 41], -torch.inf)
                        q_kan = q_out_metric[:, 42].masked_fill(~masks[:, 42], -torch.inf)
                        q_best_call = torch.stack((q_chi, q_pon, q_kan), dim=-1).max(-1).values
                        chi_opp = call_opp & torch.isfinite(q_chi)
                        pon_opp = call_opp & torch.isfinite(q_pon)
                        kan_opp = call_opp & torch.isfinite(q_kan)
                        metrics['train/call/chi_opportunity_rate'] += chi_opp.to(torch.float32).mean()
                        metrics['train/call/pon_opportunity_rate'] += pon_opp.to(torch.float32).mean()
                        metrics['train/call/kan_opportunity_rate'] += kan_opp.to(torch.float32).mean()
                        metrics['train/call/call_taken_given_opportunity'] += call_actions[call_opp].to(torch.float32).mean()
                        metrics['train/call/call_argmax_given_opportunity'] += call_argmax[call_opp].to(torch.float32).mean()
                        metrics['train/call/none_argmax_given_opportunity'] += none_argmax[call_opp].to(torch.float32).mean()
                        metrics['train/call/q_best_call_minus_none'] += (q_best_call[call_opp] - q_none[call_opp]).mean()
                        if chi_opp.any():
                            metrics['train/call/q_chi_best_minus_none'] += (q_chi[chi_opp] - q_none[chi_opp]).mean()
                        if pon_opp.any():
                            metrics['train/call/q_pon_minus_none'] += (q_pon[pon_opp] - q_none[pon_opp]).mean()
                        if kan_opp.any():
                            metrics['train/call/q_kan_minus_none'] += (q_kan[kan_opp] - q_none[kan_opp]).mean()

                        if threats is not None:
                            for label, cond in (('riichi', opp_riichi), ('fuuro_no_riichi', opp_fuuro_no_riichi)):
                                threat_call_opp = call_opp & cond
                                if not threat_call_opp.any():
                                    continue
                                metrics[f'train/threat/call_taken_given_{label}'] += call_actions[threat_call_opp].to(torch.float32).mean()
                                metrics[f'train/threat/none_argmax_given_call_opp_vs_{label}'] += none_argmax[threat_call_opp].to(torch.float32).mean()
                                metrics[f'train/threat/q_best_call_minus_none_vs_{label}'] += (
                                    q_best_call[threat_call_opp] - q_none[threat_call_opp]
                                ).mean()

                    riichi_opp = masks[:, 37]
                    if threats is not None and riichi_opp.any():
                        non_riichi_q = q_out_metric.masked_fill(~masks, -torch.inf)
                        non_riichi_q[:, 37] = -torch.inf
                        best_non_riichi_q = non_riichi_q.max(-1).values
                        for label, cond in (('riichi', opp_riichi), ('fuuro_no_riichi', opp_fuuro_no_riichi)):
                            threat_riichi_opp = riichi_opp & cond
                            if not threat_riichi_opp.any():
                                continue
                            metrics[f'train/threat/riichi_taken_given_{label}'] += (actions[threat_riichi_opp] == 37).to(torch.float32).mean()
                            metrics[f'train/threat/q_riichi_minus_best_non_riichi_vs_{label}'] += (
                                q_out_metric[threat_riichi_opp, 37] - best_non_riichi_q[threat_riichi_opp]
                            ).mean()

                    is_call_action = (actions >= 38) & (actions <= 42)
                    is_none_action = actions == 45
                    is_riichi_action = actions == 37
                    tenpai_or_better = shantens <= 0
                    one_shanten = shantens == 1
                    two_plus_shanten = shantens >= 2
                    if tenpai_or_better.any():
                        metrics['train/by_shanten/tenpai_or_better/call_rate'] += is_call_action[tenpai_or_better].to(torch.float32).mean()
                        metrics['train/by_shanten/tenpai_or_better/riichi_rate'] += is_riichi_action[tenpai_or_better].to(torch.float32).mean()
                        metrics['train/by_shanten/tenpai_or_better/none_rate'] += is_none_action[tenpai_or_better].to(torch.float32).mean()
                    if one_shanten.any():
                        metrics['train/by_shanten/one_shanten/call_rate'] += is_call_action[one_shanten].to(torch.float32).mean()
                        metrics['train/by_shanten/one_shanten/none_rate'] += is_none_action[one_shanten].to(torch.float32).mean()
                    if two_plus_shanten.any():
                        metrics['train/by_shanten/two_plus_shanten/call_rate'] += is_call_action[two_plus_shanten].to(torch.float32).mean()
                        metrics['train/by_shanten/two_plus_shanten/none_rate'] += is_none_action[two_plus_shanten].to(torch.float32).mean()
                    for rank, tag in enumerate(rank_metric_tags):
                        metrics[tag] += (player_ranks == rank).to(torch.float32).mean()

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if detailed_metrics:
                    grad_norm_mortal = module_grad_norm(mortal)
                    grad_norm_dqn = module_grad_norm(dqn)
                    grad_norm_aux = module_grad_norm(aux_net)
                    grad_norm_total = (
                        grad_norm_mortal.square()
                        + grad_norm_dqn.square()
                        + grad_norm_aux.square()
                    ).sqrt()
                    opt_metrics['count'] += 1
                    opt_metrics['grad/total_norm'] += grad_norm_total
                    opt_metrics['grad/mortal_norm'] += grad_norm_mortal
                    opt_metrics['grad/dqn_norm'] += grad_norm_dqn
                    opt_metrics['grad/aux_norm'] += grad_norm_aux

                    if max_grad_norm > 0:
                        params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                        clip_grad_norm_(params, max_grad_norm)
                        post_clip_total = (
                            module_grad_norm(mortal).square()
                            + module_grad_norm(dqn).square()
                            + module_grad_norm(aux_net).square()
                        ).sqrt()
                    else:
                        post_clip_total = grad_norm_total
                    opt_metrics['grad/post_clip_total_norm'] += post_clip_total
                elif max_grad_norm > 0:
                    # clipping is part of training and must run every opt step;
                    # only the extra norm bookkeeping is skipped here
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                block_secs = time.monotonic() - perf['block_t0']
                writer.add_scalar('perf/steps_per_sec', save_every / max(1e-9, block_secs), steps)
                if device.type == 'cuda':
                    writer.add_scalar(
                        'perf/gpu_mem_peak_gb',
                        torch.cuda.max_memory_allocated(device) / 2**30,
                        steps,
                    )
                    torch.cuda.reset_peak_memory_stats(device)

                # downsample to reduce tensorboard event size
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]
                all_td = all_q - all_q_target
                all_td_abs = all_td.abs()
                all_td_1d = all_td.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', to_scalar(stats['dqn_loss'] / save_every), steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', to_scalar(stats['cql_loss'] / save_every), steps)
                writer.add_scalar('loss/next_rank_loss', to_scalar(stats['next_rank_loss'] / save_every), steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_scalar('q/selected_mean', to_scalar(all_q.mean()), steps)
                writer.add_scalar('q/selected_std', to_scalar(all_q.std(unbiased=False)), steps)
                writer.add_scalar('q/selected_min', to_scalar(all_q.min()), steps)
                writer.add_scalar('q/selected_max', to_scalar(all_q.max()), steps)
                writer.add_scalar('target/mean', to_scalar(all_q_target.mean()), steps)
                writer.add_scalar('target/std', to_scalar(all_q_target.std(unbiased=False)), steps)
                writer.add_scalar('target/min', to_scalar(all_q_target.min()), steps)
                writer.add_scalar('target/max', to_scalar(all_q_target.max()), steps)
                writer.add_scalar('td/error_mean', to_scalar(all_td.mean()), steps)
                writer.add_scalar('td/abs_mean', to_scalar(all_td_abs.mean()), steps)
                writer.add_scalar('td/rmse', to_scalar(all_td.square().mean().sqrt()), steps)
                writer.add_scalar('td/abs_p95', to_scalar(torch.quantile(all_td_abs.flatten(), 0.95)), steps)
                for tag, value in metrics.items():
                    if version != 4 and tag.startswith('train/threat/'):
                        continue
                    writer.add_scalar(tag, to_scalar(value / max(1, metrics_batches)), steps)

                opt_count = opt_metrics['count']
                if opt_count > 0:
                    for tag, value in opt_metrics.items():
                        if tag == 'count':
                            continue
                        writer.add_scalar(tag, to_scalar(value / opt_count), steps)

                    param_norm_mortal = module_param_norm(mortal)
                    param_norm_dqn = module_param_norm(dqn)
                    param_norm_aux = module_param_norm(aux_net)
                    writer.add_scalar('param/mortal_norm', to_scalar(param_norm_mortal), steps)
                    writer.add_scalar('param/dqn_norm', to_scalar(param_norm_dqn), steps)
                    writer.add_scalar('param/aux_norm', to_scalar(param_norm_aux), steps)
                    writer.add_scalar(
                        'grad_to_param/mortal',
                        to_scalar((opt_metrics['grad/mortal_norm'] / opt_count) / param_norm_mortal.clamp_min(1e-12)),
                        steps,
                    )
                    writer.add_scalar(
                        'grad_to_param/dqn',
                        to_scalar((opt_metrics['grad/dqn_norm'] / opt_count) / param_norm_dqn.clamp_min(1e-12)),
                        steps,
                    )
                    writer.add_scalar(
                        'grad_to_param/aux',
                        to_scalar((opt_metrics['grad/aux_norm'] / opt_count) / param_norm_aux.clamp_min(1e-12)),
                        steps,
                    )
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.add_histogram('td_error', all_td_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                for k in metrics:
                    metrics[k] = 0
                for k in opt_metrics:
                    opt_metrics[k] = 0
                idx = 0
                metrics_batches = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'top_checkpoints': top_checkpoints,
                    'config': config,
                }
                t_save = time.monotonic()
                if async_save:
                    queue_save(state_to_cpu(state), state_file)
                else:
                    torch.save(state, state_file)
                writer.add_scalar('perf/checkpoint_save_sec', time.monotonic() - t_save, steps)

                if online and steps % submit_every != 0:
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    # make sure the checkpoint just queued is fully on disk
                    # before it gets copied below
                    join_save()
                    t_test = time.monotonic()
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
                    writer.add_scalar('perf/test_play_sec', time.monotonic() - t_test, steps)
                    mortal.train()
                    dqn.train()

                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        if not async_save:
                            # with async_save the same state is already fully
                            # written to state_file (joined above)
                            torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)

                    # top-k candidate checkpoint management
                    if top_k > 0:
                        worst_rank = max(c['avg_rank'] for c in top_checkpoints) if top_checkpoints else float('inf')
                        if len(top_checkpoints) < top_k or stat.avg_rank < worst_rank:
                            os.makedirs(top_k_dir, exist_ok=True)
                            ckpt_path = path.join(top_k_dir, f'candidate_{steps}.pth')
                            shutil.copy(state_file, ckpt_path)
                            top_checkpoints.append({
                                'avg_rank': stat.avg_rank,
                                'avg_pt': avg_pt,
                                'steps': steps,
                                'filepath': ckpt_path,
                            })
                            if len(top_checkpoints) > top_k:
                                worst = max(top_checkpoints, key=lambda c: c['avg_rank'])
                                top_checkpoints.remove(worst)
                                if path.exists(worst['filepath']):
                                    os.remove(worst['filepath'])
                            ranked = sorted(top_checkpoints, key=lambda c: c['avg_rank'])
                            logging.info(f'top-{top_k} candidates:')
                            for i, c in enumerate(ranked):
                                logging.info(f'  #{i+1}: rank={c["avg_rank"]:.6}, pt={c["avg_pt"]:.6}, step={c["steps"]}')
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                perf['block_t0'] = time.monotonic()
                pb = tqdm(total=save_every, desc='TRAIN')

        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, at_turns, shantens in data_loader:
            bs = obs.shape[0]
            if bs != batch_size:
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_at_turns.append(at_turns)
                remaining_shantens.append(shantens)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, at_turns, shantens)

        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            at_turns = torch.cat(remaining_at_turns, dim=0)
            shantens = torch.cat(remaining_shantens, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                    at_turns[start:end],
                    shantens[start:end],
                )
                start = end
                end += batch_size
        pb.close()

        if online:
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break
    join_save()

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
