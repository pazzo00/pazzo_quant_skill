"""Quarterly financial-statement helpers used by QMJ."""

from __future__ import annotations

import numpy as np


def cal_quarter_data(data, flag, idx, gap, season):
    """Convert cumulative report values to comparable quarter values."""
    result = data[:, idx, :] - data[:, idx + gap, :]
    missing = np.isnan(result)
    result[missing] = data[:, idx, :][missing] / season[missing]
    result[flag] = data[:, idx, :][flag]
    return result
