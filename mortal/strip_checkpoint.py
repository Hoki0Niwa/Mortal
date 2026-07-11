"""Strip a checkpoint down to the keys needed for inference.

build_engine_from_state only needs {'mortal', 'current_dqn' or 'policy_net',
'config'}; the optimizer/scheduler states make pool checkpoints hundreds of
MB and they are re-loaded on every train_play session. Run this over the
files in the opponent pool directory to make those loads fast.

Usage:
    python strip_checkpoint.py <in.pth> [out.pth]   # in-place when out omitted
"""
import os
import sys
import torch
from checkpoint import load_checkpoint

KEEP_KEYS = ('mortal', 'current_dqn', 'policy_net', 'config')

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src
    before = os.path.getsize(src)

    state = load_checkpoint(src)
    stripped = {k: state[k] for k in KEEP_KEYS if k in state}
    assert 'mortal' in stripped, "checkpoint has no 'mortal' key"
    assert 'current_dqn' in stripped or 'policy_net' in stripped, \
        "checkpoint has neither 'current_dqn' nor 'policy_net'"
    assert 'config' in stripped, "checkpoint has no 'config' key"

    tmp = dst + '.tmp'
    torch.save(stripped, tmp)
    os.replace(tmp, dst)

    after = os.path.getsize(dst)
    print(f'{src} ({before / 2**20:.0f} MiB) -> {dst} ({after / 2**20:.0f} MiB)')

if __name__ == '__main__':
    main()
