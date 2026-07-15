"""Decision-point benchmark suite (v1).

Scores a checkpoint on decision points from held-out game logs
(dataset.holdout_files, excluded from training), giving low-noise
per-checkpoint curves for efficiency and push/fold behavior that avg-rank
test play (SE ~0.02 rank at 10k games) cannot resolve.

The v4 obs channel constants below were located and validated empirically
against the action mask on real data; validate_channels() re-checks the
invariants on every run and fails hard if the obs encoding ever changes.

Usage:
    python suite.py --state /path/to/checkpoint.pth [--files N] [--tb DIR]

train.py also calls run_suite() at every test_every boundary when
[suite].enabled is true.
"""
import prelude

import argparse
import logging
import time
import numpy as np
import torch
from engine import build_engine_from_state, resolve_amp_dtype
from checkpoint import load_checkpoint
from common import filtered_trimmed_lines, tqdm
from dataloader import FileDatasetsIter
from config import config

V4_CAND_CH = 874        # discard candidates (deaka'd), == mask-derived set
V4_KEEP_CH = 875        # keep-shanten discards
V4_NEXT_CH = 876        # next-shanten discards
V4_OWN_RIICHI_CH = 878  # own riichi declared (fill-style row)
V4_OPP_FUURO_CH = 771   # 3 opponents x 4 fuuro x 5 rows
V4_OPP_RIICHI_CH = 857  # 3 opponents, fill-style rows
AKA_TO_TID = {34: 4, 35: 13, 36: 22}

CHANNEL_DRIFT_MSG = 'obs encoding may have changed; suite channel constants need re-validation'

def mask_candidates(masks):
    # (N,46) bool -> (N,34) float32 deaka'd discard candidate plane
    cand = masks[:, :34].astype(np.float32)
    for action, tid in AKA_TO_TID.items():
        cand[masks[:, action], tid] = 1.0
    return cand

def iter_entries(max_files=0, include_sp_labels=False):
    version = config['control']['version']
    if version != 4:
        raise ValueError('suite v1 supports version 4 only')

    holdout_path = config['dataset']['holdout_files']
    with open(holdout_path, encoding='utf-8') as f:
        files = list(filtered_trimmed_lines(f))
    if max_files > 0:
        files = files[:max_files]

    player_names_set = set()
    for filename in config['dataset']['player_names_files']:
        # utf-8-sig also strips the BOM some editors prepend (see train.py)
        with open(filename, encoding='utf-8-sig') as f:
            player_names_set.update(filtered_trimmed_lines(f))
    player_names = list(player_names_set)

    file_data = FileDatasetsIter(
        version = version,
        file_list = files,
        pts = config['env']['pts'],
        file_batch_size = config.get('suite', {}).get('file_batch_size', 10),
        num_epochs = 1,
        enable_augmentation = False,
        player_names = player_names,
        include_sp_labels = include_sp_labels,
    )
    yield from file_data

def validate_channels(obs_np, masks_np):
    can_discard = masks_np[:, :37].any(-1)
    if not can_discard.any():
        return

    cand = mask_candidates(masks_np)
    consistent = can_discard & (obs_np[:, V4_CAND_CH] == cand).all(-1)
    consistent_rate = consistent[can_discard].mean()
    if consistent_rate < 0.99:
        raise RuntimeError(
            f'ch{V4_CAND_CH} candidate-plane consistency rate {consistent_rate:.4f} < 0.99; '
            f'{CHANNEL_DRIFT_MSG}'
        )

    if consistent.any():
        keep = obs_np[:, V4_KEEP_CH]
        nxt = obs_np[:, V4_NEXT_CH]
        overlap = (keep[consistent] * nxt[consistent]).sum(-1)
        if not (overlap == 0).all():
            raise RuntimeError(
                f'ch{V4_KEEP_CH}/ch{V4_NEXT_CH} (keep/next shanten discards) are not disjoint '
                f'on consistent rows; {CHANNEL_DRIFT_MSG}'
            )

    for ch in (V4_OPP_RIICHI_CH, V4_OPP_RIICHI_CH + 1, V4_OPP_RIICHI_CH + 2, V4_OWN_RIICHI_CH):
        rows = obs_np[:, ch]
        fill_rate = (rows == rows[:, :1]).all(-1).mean()
        if fill_rate < 0.999:
            raise RuntimeError(
                f'ch{ch} is not fill-style (constant-row rate {fill_rate:.4f} < 0.999); '
                f'{CHANNEL_DRIFT_MSG}'
            )

def _add(sums, counts, tag, values, cond=None):
    v = values if cond is None else values[cond]
    n = v.shape[0]
    if n == 0:
        return
    sums[tag] = sums.get(tag, 0.0) + float(np.asarray(v, dtype=np.float64).sum())
    counts[tag] = counts.get(tag, 0) + n

def accumulate_metrics(sums, counts, obs_np, masks_np, actions_np, shanten_np, q, sp_np=None):
    cand = mask_candidates(masks_np)
    can_discard = masks_np[:, :37].any(-1)
    consistent = can_discard & (obs_np[:, V4_CAND_CH] == cand).all(-1)
    own_riichi = obs_np[:, V4_OWN_RIICHI_CH, 0] > 0
    opp_riichi = (obs_np[:, V4_OPP_RIICHI_CH:V4_OPP_RIICHI_CH + 3].max(-1) > 0).any(-1)
    opp_fuuro = (
        obs_np[:, V4_OPP_FUURO_CH:V4_OPP_FUURO_CH + 60]
        .reshape(-1, 3, 4, 5, 34).max((-1, -2, -3)) > 0
    ).any(-1)
    no_threat = ~(opp_riichi | opp_fuuro)
    fuuro_no_riichi = opp_fuuro & ~opp_riichi

    a_star = q.argmax(-1)
    agree = a_star == actions_np
    is_discard = a_star <= 36

    tid = a_star.copy()
    for action, t in AKA_TO_TID.items():
        tid[a_star == action] = t
    tid_c = np.clip(tid, 0, 33)

    keep = obs_np[:, V4_KEEP_CH]
    nxt = obs_np[:, V4_NEXT_CH]
    rows = np.arange(len(a_star))
    in_keep = keep[rows, tid_c] > 0
    in_next = nxt[rows, tid_c] > 0
    regression = ~(in_keep | in_next)
    has_next = nxt.sum(-1) > 0
    eff_base = consistent & ~own_riichi & is_discard

    eff_no_threat = eff_base & no_threat
    _add(sums, counts, 'suite/efficiency/shanten_regression_rate', regression, eff_no_threat)
    _add(sums, counts, 'suite/efficiency/advance_miss_rate', ~in_next, eff_no_threat & has_next)
    _add(sums, counts, 'suite/behavior/shanten_regression_rate_vs_riichi', regression, eff_base & opp_riichi)

    _add(sums, counts, 'suite/agreement/human_top1', agree)
    _add(sums, counts, 'suite/agreement/human_top1_no_threat', agree, no_threat)
    _add(sums, counts, 'suite/agreement/human_top1_vs_riichi', agree, opp_riichi)
    _add(sums, counts, 'suite/agreement/human_top1_vs_fuuro_no_riichi', agree, fuuro_no_riichi)
    _add(sums, counts, 'suite/agreement/human_top1_tenpai', agree, shanten_np <= 0)
    _add(sums, counts, 'suite/agreement/human_top1_one_shanten', agree, shanten_np == 1)
    _add(sums, counts, 'suite/agreement/human_top1_two_plus', agree, shanten_np >= 2)

    call_opp = masks_np[:, 38:43].any(-1) & masks_np[:, 45]
    none_taken = a_star == 45
    _add(sums, counts, 'suite/behavior/none_rate_given_call_opp_no_threat', none_taken, call_opp & no_threat)
    _add(sums, counts, 'suite/behavior/none_rate_given_call_opp_vs_riichi', none_taken, call_opp & opp_riichi)
    _add(
        sums, counts, 'suite/behavior/none_rate_given_call_opp_vs_fuuro_no_riichi',
        none_taken, call_opp & fuuro_no_riichi,
    )

    riichi_taken = a_star == 37
    _add(sums, counts, 'suite/behavior/riichi_taken_no_threat', riichi_taken, masks_np[:, 37] & no_threat)
    _add(sums, counts, 'suite/behavior/riichi_taken_vs_riichi', riichi_taken, masks_np[:, 37] & opp_riichi)

    _add(sums, counts, 'suite/behavior/agari_taken', a_star == 43, masks_np[:, 43])

    q_top2 = np.sort(q, axis=-1)[:, -2:]
    margin = q_top2[:, 1] - q_top2[:, 0]
    _add(sums, counts, 'suite/q/top1_top2_margin', margin, np.isfinite(margin))

    q_best = q_top2[:, 1]
    gap = q_best - q[rows, actions_np]
    _add(sums, counts, 'suite/q/logged_gap_to_best', gap, np.isfinite(gap))

    if sp_np is not None:
        accumulate_sp_metrics(
            sums, counts, sp_np, actions_np, masks_np,
            consistent=consistent, own_riichi=own_riichi,
            no_threat=no_threat, model_is_discard=is_discard, model_tid=tid_c,
        )

def accumulate_sp_metrics(
    sums, counts, sp_np, actions_np, masks_np, *,
    consistent, own_riichi, no_threat, model_is_discard, model_tid,
):
    # sp_np: (N,37) per-discard SP EVs aligned to the discard action space,
    # NaN where the tile is not a keep-shanten candidate or no EV exists
    # (shanten > 3, own riichi, non-discard decisions, tiles_left < 4).
    # Collapse to deaka (N,34); the loader fills at most one of {deaka, aka}
    # per state, and np.fmax ignores NaN, so this just merges the two slots.
    ev34 = sp_np[:, :34].copy()
    for action, tid in AKA_TO_TID.items():
        ev34[:, tid] = np.fmax(ev34[:, tid], sp_np[:, action])

    can_discard = masks_np[:, :37].any(-1)
    cand = mask_candidates(masks_np)
    bad = np.isfinite(ev34) & (cand == 0)
    if bad[consistent].any():
        raise RuntimeError(f'SP EV present on a non-candidate tile; {CHANNEL_DRIFT_MSG}')

    has_sp = np.isfinite(ev34).any(-1)
    _add(sums, counts, 'suite/sp/coverage', has_sp, can_discard & consistent & ~own_riichi)

    sp_base = has_sp & consistent & ~own_riichi
    ev_filled = np.where(np.isfinite(ev34), ev34, -np.inf)
    sp_best_tid = ev_filled.argmax(-1)
    sp_best_ev = ev_filled.max(-1)
    rows = np.arange(len(actions_np))

    # model argmax vs the SP-optimal discard
    m = sp_base & model_is_discard
    match = model_tid == sp_best_tid
    model_ev = ev34[rows, model_tid]
    off_table = ~np.isfinite(model_ev)  # model chose a shanten-regressing discard
    ev_loss = sp_best_ev - model_ev
    in_table = m & ~off_table
    _add(sums, counts, 'suite/sp/model_match_rate', match, m)
    _add(sums, counts, 'suite/sp/model_match_rate_no_threat', match, m & no_threat)
    _add(sums, counts, 'suite/sp/model_off_table_rate', off_table, m)
    _add(sums, counts, 'suite/sp/model_ev_loss', ev_loss, in_table)
    _add(sums, counts, 'suite/sp/model_ev_loss_no_threat', ev_loss, in_table & no_threat)

    # human (logged action) reference, as an anchor for the model metrics
    h_is_discard = actions_np <= 36
    h_tid = actions_np.copy()
    for action, tid in AKA_TO_TID.items():
        h_tid[actions_np == action] = tid
    h_tid = np.clip(h_tid, 0, 33)
    h = sp_base & h_is_discard
    h_ev = ev34[rows, h_tid]
    _add(sums, counts, 'suite/sp/human_match_rate', h_tid == sp_best_tid, h)
    _add(sums, counts, 'suite/sp/human_ev_loss', sp_best_ev - h_ev, h & np.isfinite(h_ev))

def score(mortal, dqn, device, max_files=0):
    was_training = (mortal.training, dqn.training)
    mortal.eval()
    dqn.eval()
    try:
        enable_amp = config['control']['enable_amp']
        amp_dtype = resolve_amp_dtype(config['control'])
        batch_size = config.get('suite', {}).get('batch_size', 1024)
        sp_metrics = config.get('suite', {}).get('sp_metrics', False)

        sums = {}
        counts = {}
        validated = False

        obs_list = []
        actions_list = []
        masks_list = []
        shantens_list = []
        sp_list = []

        def flush():
            nonlocal validated
            if not obs_list:
                return
            obs_np = np.stack(obs_list).astype(np.float32, copy=False)
            masks_np = np.stack(masks_list).astype(bool, copy=False)
            actions_np = np.array(actions_list, dtype=np.int64)
            shanten_np = np.array(shantens_list, dtype=np.int64)
            sp_np = np.stack(sp_list).astype(np.float32, copy=False) if sp_metrics else None

            obs_t = torch.as_tensor(obs_np, device=device)
            masks_t = torch.as_tensor(masks_np, device=device)
            with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=enable_amp):
                q = dqn(mortal(obs_t), masks_t)
            q = q.float().cpu().numpy()

            if not validated:
                validate_channels(obs_np, masks_np)
                validated = True

            accumulate_metrics(sums, counts, obs_np, masks_np, actions_np, shanten_np, q, sp_np)

            obs_list.clear()
            actions_list.clear()
            masks_list.clear()
            shantens_list.clear()
            sp_list.clear()

        entries = iter_entries(max_files=max_files, include_sp_labels=sp_metrics)
        for entry in tqdm(entries, desc='SUITE', unit='decision'):
            obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks, at_turns, shantens = entry[:8]
            obs_list.append(obs)
            actions_list.append(actions)
            masks_list.append(masks)
            shantens_list.append(shantens)
            if sp_metrics:
                sp_list.append(entry[8])
            if len(obs_list) >= batch_size:
                flush()
        flush()
    finally:
        mortal.train(was_training[0])
        dqn.train(was_training[1])

    metrics = {tag: total / counts[tag] for tag, total in sums.items()}
    return metrics, counts

def run_suite(mortal, dqn, device, writer, steps):
    t0 = time.monotonic()
    metrics, counts = score(mortal, dqn, device, max_files=config.get('suite', {}).get('max_files', 0))
    for tag, value in metrics.items():
        writer.add_scalar(tag, value, steps)
    writer.add_scalar('perf/suite_sec', time.monotonic() - t0, steps)
    writer.flush()

    nan = float('nan')
    msg = (
        'suite: '
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', required=True)
    parser.add_argument('--files', type=int, default=0)
    parser.add_argument('--tb', default='')
    parser.add_argument('--sp', action='store_true', help='force [suite].sp_metrics on for this run')
    args = parser.parse_args()

    if args.sp:
        config.setdefault('suite', {})['sp_metrics'] = True

    device = torch.device(config['control']['device'])
    state = load_checkpoint(args.state)
    engine, info = build_engine_from_state(state, device=device, name='suite')
    logging.info(f"head_kind: {info['head_kind']}, norm: {info['norm']}, version: {info['version']}")

    metrics, counts = score(engine.brain, engine.dqn, device, max_files=args.files)

    for tag, value in sorted(metrics.items()):
        print(f'{tag:<55} {value:>8.4f}  (n={counts[tag]:,})')

    if args.tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.tb)
        steps = state.get('steps', 0)
        for tag, value in metrics.items():
            writer.add_scalar(tag, value, steps)
        writer.flush()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
