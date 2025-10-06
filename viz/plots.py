# viz/plots.py — Python 3.8 & older Matplotlib compatible

from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# ---- Safe color cycle (hex codes; no 'C0' refs) ----
COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]
plt.rc("axes", prop_cycle=cycler("color", COLORS))

def tcrit_95(n: int) -> float:
    if n <= 1:
        return float("inf")
    if n < 30:
        return 2.262
    return 1.96

def mean_ci95(vals):
    arr = np.array(list(vals), dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return float(arr[0]), 0.0
    m = float(arr.mean())
    s = float(arr.std(ddof=1))
    half = tcrit_95(n) * (s / math.sqrt(n))
    return m, half

def plot_with_ci_band(ax, xs, mean, half, *, label, line_kwargs=None, band_kwargs=None):
    """Plot mean line and shaded CI band (Py3.8 safe, old-mpl safe)."""
    line_kwargs = {} if line_kwargs is None else dict(line_kwargs)
    band = {"alpha": 0.25}
    if band_kwargs is not None:
        band.update(dict(band_kwargs))
    line, = ax.plot(xs, mean, label=label, **line_kwargs)
    ax.fill_between(xs, mean - half, mean + half, **band)
    return line
