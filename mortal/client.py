import prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
from os import path
from model import Brain, DQN
from player import TrainPlayer
from common import send_msg, recv_msg, UnexpectedEOF
from config import config

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    dqn = DQN(version=version).to(device)
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        t_wait = time.monotonic()
        while True:
            # survive server/trainer restarts so workers can run unattended
            try:
                with socket.socket() as conn:
                    conn.connect(remote)
                    msg = {
                        'type': 'get_param',
                        'param_version': param_version,
                    }
                    send_msg(conn, msg)
                    rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
            except (OSError, UnexpectedEOF) as e:
                logging.warning(f'server not reachable, retrying: {e}')
            time.sleep(3)
        param_wait_secs = time.monotonic() - t_wait
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        logging.info(f'param has been updated (waited {param_wait_secs:.1f}s)')

        t_play = time.monotonic()
        rankings, file_list = train_player.train_play(mortal, dqn, device)
        play_secs = time.monotonic() - t_play
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        t_submit = time.monotonic()
        while True:
            try:
                with socket.socket() as conn:
                    conn.connect(remote)
                    send_msg(conn, {
                        'type': 'submit_replay',
                        'logs': logs,
                        'param_version': param_version,
                    })
                logging.info('logs have been submitted')
                break
            except OSError as e:
                logging.warning(f'submit failed, retrying in 15s: {e}')
                time.sleep(15)
        logging.info(
            f'session timing: param_wait={param_wait_secs:.1f}s, '
            f'play={play_secs:.1f}s, submit={time.monotonic() - t_submit:.1f}s'
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
