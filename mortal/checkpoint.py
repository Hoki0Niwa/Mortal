"""Safe checkpoint loading.

All checkpoints in this project are plain tensors and built-in containers,
but some legacy files additionally pickled numpy scalars (e.g. best_perf
values saved as np.float64), which the default weights_only unpickler
rejects. Allowlisting numpy's scalar/dtype reconstructors keeps
weights_only=True (no arbitrary code execution) while staying compatible
with every existing checkpoint, including opponent-pool files.
"""
import torch
import numpy as np

try:
    from numpy._core.multiarray import scalar as _np_scalar  # numpy >= 2
except ImportError:
    from numpy.core.multiarray import scalar as _np_scalar  # numpy 1.x

_SAFE_GLOBALS = [_np_scalar, np.dtype]
if hasattr(np, 'dtypes'):
    _SAFE_GLOBALS += [
        getattr(np.dtypes, name)
        for name in dir(np.dtypes)
        if name.endswith('DType')
    ]

def load_checkpoint(file, map_location='cpu'):
    with torch.serialization.safe_globals(_SAFE_GLOBALS):
        return torch.load(file, weights_only=True, map_location=map_location)
