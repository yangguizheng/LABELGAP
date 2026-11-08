"""Numeric transformations shared by the two intrusion datasets."""

import numpy as np


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """Apply a sign-preserving log transform to numeric features."""
    return np.sign(x) * np.log1p(np.abs(x))
