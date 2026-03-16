"""
Helper module to fix OGB loading with PyTorch 2.6+
Import this before importing OGB datasets
"""

import torch
from functools import wraps

# Store the original torch.load
_original_torch_load = torch.load

@wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    """Patched torch.load that sets weights_only=False by default for OGB compatibility"""
    # Only set weights_only=False if not explicitly specified
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Monkey patch torch.load
torch.load = _patched_torch_load

print("OGB fix applied: torch.load will use weights_only=False")