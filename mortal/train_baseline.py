"""Train the oracle luck baseline b(state, hidden info) used by offline AWBC
when `offline_loss.awbc_baseline` is enabled.

Offline AWBC weights the BC term by exp(advantage / beta), where advantage is
normally `MC return - dueling V`. That advantage is dominated by mahjong's
luck noise (wall draws, opponents' hands), so it ends up over-weighting
lucky trajectories instead of good decisions. This script pre-trains a
network b(s, z) conditioned on the hidden information z (the invisible_obs:
wall order and opponents' hands) that is only observable during offline
training, not during play. Because the wall is fixed at the start of the
kyoku and independent of the trainee's actions, conditioning on it does not
leak action-dependent information into the baseline, so
`advantage = MC return - b(s, z)` remains a valid (lower-variance, luck-
corrected) advantage estimate for AWBC.

Usage:
    python train_baseline.py

The resulting checkpoint is consumed by train.py when
`offline_loss.awbc_baseline = true`.

Gate: check the reported `unexplained_var` (validation
`(target - pred).var() / target.var()`) before enabling awbc_baseline in
training. Calibration on this dataset: a 9-feature linear probe (own
shanten, opponents' hidden shanten/waits/riichi, turn) reaches ~0.95, so a
useful baseline must land clearly below ~0.9 to carry more signal than
trivial features. Note the kyoku outcome has large irreducible entropy from
human policy choices even given the full hidden state, so values near 0.5
are not attainable; what matters for AWBC is that Var(return - b) is
meaningfully smaller than Var(return - V), i.e. that b explains luck the
state alone cannot.
"""
import prelude

import logging
import os
import gzip
import json
import random
from os import path
from glob import glob
from datetime import datetime
from itertools import chain
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from common import filtered_trimmed_lines, tqdm
from dataloader import FileDatasetsIter, worker_init_fn
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from model import Brain
from config import config

def main():
    version = config['control']['version']
    device = torch.device(config['control']['device'])

    gamma = config['env']['gamma']
    pts = config['env']['pts']

    file_batch_size = config['dataset']['file_batch_size']
    num_workers = config['dataset']['num_workers']

    b_cfg = config['awbc_baseline']
    state_file = b_cfg['state_file']
    net_cfg = b_cfg['network']
    conv_channels = net_cfg['conv_channels']
    num_blocks = net_cfg['num_blocks']

    t_cfg = b_cfg.get('train', {})
    batch_size = t_cfg.get('batch_size', 256)
    lr = t_cfg.get('lr', 3e-4)
    warmup_steps = t_cfg.get('warmup_steps', 500)
    init_state_file = t_cfg.get('init_state_file', '')
    weight_decay = t_cfg.get('weight_decay', 0.01)
    huber_delta = t_cfg.get('huber_delta', 1.0)
    valid_files = t_cfg.get('valid_files', 64)
    eval_every = t_cfg.get('eval_every', 2000)
    patience = t_cfg.get('patience', 5)
    max_steps = t_cfg.get('max_steps', 200_000)
    tensorboard_dir = t_cfg.get('tensorboard_dir', '')

    # build the file list (same logic as the offline branch in train.py)
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

    # deterministic train/valid split, independent of the per-epoch shuffling
    # FileDatasetsIter does internally
    rng = random.Random(0xba5e)
    shuffled = file_list.copy()
    rng.shuffle(shuffled)
    valid_file_list = shuffled[:valid_files]
    train_file_list = shuffled[valid_files:]
    logging.info(f'train files: {len(train_file_list):,}, valid files: {len(valid_file_list):,}')

    logging.info(f'version: {version}')
    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    brain = Brain(version=version, is_oracle=True, conv_channels=conv_channels, num_blocks=num_blocks).to(device)
    head = nn.Linear(1024, 1).to(device)

    if init_state_file:
        # warm start from a trained (non-oracle) main checkpoint; learning the
        # oracle increment on top of a trunk that already understands the game
        # is far easier than training b(s, z) from scratch. The network config
        # must match the source checkpoint's [resnet] for the transfer to work.
        src_brain = torch.load(init_state_file, weights_only=True, map_location='cpu')['mortal']
        dst = brain.state_dict()
        transferred = 0
        for key, value in src_brain.items():
            if key not in dst:
                continue
            if dst[key].shape == value.shape:
                dst[key] = value
                transferred += 1
            elif key == 'encoder.net.0.weight':
                # first conv: copy the obs slice and zero-init the extra oracle
                # input channels, so at step 0 the model behaves exactly like
                # the source trunk on the visible obs alone
                weight = torch.zeros_like(dst[key])
                weight[:, :value.shape[1]] = value
                dst[key] = weight
                transferred += 1
        brain.load_state_dict(dst)
        logging.info(f'warm start from {init_state_file}: {transferred}/{len(dst)} tensors transferred')

    optimizer = optim.AdamW(chain(brain.parameters(), head.parameters()), lr=1, weight_decay=weight_decay)
    scheduler = LinearWarmUpCosineAnnealingLR(
        optimizer,
        peak = lr,
        final = lr,
        warm_up_steps = warmup_steps,
        max_steps = max_steps,
    )

    if huber_delta > 0:
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = lambda input, target: 0.5 * nn.functional.mse_loss(input, target)

    train_data = FileDatasetsIter(
        version = version,
        file_list = train_file_list,
        pts = pts,
        oracle = True,
        file_batch_size = file_batch_size,
        num_epochs = 1000,
        enable_augmentation = False,
    )
    train_loader = DataLoader(
        dataset = train_data,
        batch_size = batch_size,
        drop_last = True,
        num_workers = num_workers,
        pin_memory = True,
        worker_init_fn = worker_init_fn,
    )

    def evaluate():
        # a fresh FileDatasetsIter/DataLoader is built each time so the full
        # valid set is swept exactly once, regardless of when this is called
        valid_data = FileDatasetsIter(
            version = version,
            file_list = valid_file_list,
            pts = pts,
            oracle = True,
            file_batch_size = file_batch_size,
            num_epochs = 1,
            enable_augmentation = False,
        )
        valid_loader = DataLoader(
            dataset = valid_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = 2,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        )
        brain.eval()
        head.eval()
        preds = []
        targets = []
        with torch.inference_mode():
            for batch in tqdm(valid_loader, desc='VALID'):
                obs, invisible_obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, at_turns, shantens = batch
                obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
                invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
                steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
                kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device, non_blocking=True)
                target = (gamma ** steps_to_done * kyoku_rewards).to(torch.float32)
                pred = head(brain(obs, invisible_obs)).squeeze(-1)
                preds.append(pred)
                targets.append(target)
        brain.train()
        head.train()
        pred = torch.cat(preds)
        target = torch.cat(targets)
        valid_loss = criterion(pred, target).item()
        unexplained_var = ((target - pred).var() / target.var()).item()
        return valid_loss, unexplained_var

    writer = SummaryWriter(tensorboard_dir) if tensorboard_dir else None

    steps = 0
    best_valid_loss = float('inf')
    best_unexplained_var = float('inf')
    patience_left = patience
    train_loss_sum = 0.
    train_loss_count = 0

    brain.train()
    head.train()

    pb = tqdm(total=max_steps, desc='TRAIN')
    stop = False
    for batch in train_loader:
        if stop:
            break
        obs, invisible_obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, at_turns, shantens = batch

        # non_blocking pairs with the DataLoader's pin_memory to overlap
        # H2D transfers with compute; results are identical
        obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
        invisible_obs = invisible_obs.to(dtype=torch.float32, device=device, non_blocking=True)
        steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
        kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device, non_blocking=True)
        target = (gamma ** steps_to_done * kyoku_rewards).to(torch.float32)

        pred = head(brain(obs, invisible_obs)).squeeze(-1)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        steps += 1
        pb.update(1)
        train_loss_sum += loss.item()
        train_loss_count += 1

        if steps % eval_every == 0 or steps >= max_steps:
            valid_loss, unexplained_var = evaluate()
            train_loss_mean = train_loss_sum / max(1, train_loss_count)
            logging.info(
                f'steps={steps:,} train_loss={train_loss_mean:.4f} '
                f'valid_loss={valid_loss:.4f} unexplained_var={unexplained_var:.3f}'
            )
            if writer is not None:
                writer.add_scalar('loss/train', train_loss_mean, steps)
                writer.add_scalar('loss/valid', valid_loss, steps)
                writer.add_scalar('valid/unexplained_var', unexplained_var, steps)
                writer.flush()
            train_loss_sum = 0.
            train_loss_count = 0

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_unexplained_var = unexplained_var
                patience_left = patience
                state = {
                    'brain': brain.state_dict(),
                    'head': head.state_dict(),
                    'version': version,
                    'network': {'conv_channels': conv_channels, 'num_blocks': num_blocks},
                    'steps': steps,
                    'valid_loss': valid_loss,
                    'unexplained_var': unexplained_var,
                    'timestamp': datetime.now().timestamp(),
                }
                tmp = state_file + '.tmp'
                torch.save(state, tmp)
                os.replace(tmp, state_file)
                logging.info(f'new best checkpoint saved to {state_file}')
            else:
                patience_left -= 1
                logging.info(f'no improvement, patience left: {patience_left}')
                if patience_left <= 0:
                    logging.info('early stopping: patience exhausted')
                    stop = True

        if steps >= max_steps:
            logging.info('max_steps reached')
            stop = True
    pb.close()

    if writer is not None:
        writer.close()

    logging.info(f'training finished at step {steps:,}')
    logging.info(f'best valid_loss={best_valid_loss:.4f}, best unexplained_var={best_unexplained_var:.3f}')
    # a 9-feature linear probe reaches ~0.95 on this target, so the baseline
    # must land clearly below that to carry more signal than trivial features
    if best_unexplained_var < 0.9:
        logging.info(
            'gate passed: unexplained_var is clearly below the ~0.95 linear-probe '
            'floor, the baseline explains luck beyond trivial state features'
        )
    else:
        logging.info(
            'gate failed: unexplained_var does not clearly beat the ~0.95 linear-probe '
            'floor; enabling offline_loss.awbc_baseline is unlikely to help'
        )

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
