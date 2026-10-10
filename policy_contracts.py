"""Validate discrete policy actions before conversion or state advancement."""

import numpy as np


def validated_actions(actions, count, label="policy"):
    values = np.asarray(actions)
    if values.shape != (count, 3):
        raise ValueError(f"{label} must return actions with shape {(count, 3)}, got {values.shape}")
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{label} actions must be finite integers")
    if values.dtype.kind == "f":
        if (not np.isfinite(values).all() or np.any(values != np.floor(values))
                or np.any(values < -(2.**63)) or np.any(values >= 2.**63)):
            raise ValueError(f"{label} actions must be finite int64 integers")
    elif values.dtype.kind == "u" and np.any(values > np.iinfo(np.int64).max):
        raise ValueError(f"{label} actions exceed int64 range")
    # Integers outside the action space still reach the environment's illegal
    # action handling. Reject encoding errors rather than silently truncating.
    return values.astype(np.int64, copy=False)
