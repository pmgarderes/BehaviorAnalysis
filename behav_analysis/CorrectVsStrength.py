from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_lick_rate_vs_strength(
    trial_data: pd.DataFrame,
    # --- binning options (choose ONE style) ---
    n_bins: int | None = 8,                 # used if bin_edges is None and bin_width is None
    bin_edges: list[float] | np.ndarray | None = None,  # explicit edges, e.g. [0,1,2,3,5,10]
    bin_width: float | None = None,         # uniform bins, e.g. 1.0
    bin_range: tuple[float, float] | None = None,       # used with n_bins or bin_width
    # --- filtering ---
    strength_min: float | None = None,
    strength_max: float | None = None,
    only_stim_present: bool = False,
    # --- misc ---
    title: str = "P(lick) vs stimulus strength",
) -> pd.DataFrame:
    """
    Plot average lick probability as a function of SE_strength with binomial error bars.

    Lick event (binary): NLicks > 0

    Binning modes (priority order):
      1) bin_edges provided -> use explicit edges
      2) bin_width provided -> uniform bins over bin_range (or data min/max)
      3) n_bins provided -> uniform bins over bin_range (or data min/max)

    Error bars: binomial SEM = sqrt(p*(1-p)/n)

    Returns a summary DataFrame:
      BinCenter, BinLeft, BinRight, N, NLick, Plick, SEM
    """
    df = trial_data.copy()

    # --- signals ---
    s = pd.to_numeric(df.get("SE_strength", 0), errors="coerce").fillna(0.0)
    y = (pd.to_numeric(df.get("NLicks", 0), errors="coerce").fillna(0) > 0).astype(int)

    # --- filtering ---
    keep = pd.Series(True, index=df.index)
    if only_stim_present:
        keep &= s > 0
    if strength_min is not None:
        keep &= s >= strength_min
    if strength_max is not None:
        keep &= s <= strength_max

    s = s.loc[keep]
    y = y.loc[keep]

    if len(s) == 0:
        raise ValueError("No trials left after filtering.")

    # ---- build edges ----
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("bin_edges must be a 1D list/array with >= 2 edges.")
        edges = np.unique(edges)
        if len(edges) < 2:
            raise ValueError("bin_edges must contain at least 2 distinct values.")
    else:
        if bin_range is None:
            lo = float(np.nanmin(s.values))
            hi = float(np.nanmax(s.values))
        else:
            lo, hi = float(bin_range[0]), float(bin_range[1])

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid bin_range: {bin_range}")

        if bin_width is not None:
            if bin_width <= 0:
                raise ValueError("bin_width must be > 0.")
            edges = np.arange(lo, hi + bin_width, bin_width, dtype=float)
            if edges[-1] < hi:
                edges = np.append(edges, hi)
        else:
            if n_bins is None or int(n_bins) < 1:
                raise ValueError("Provide n_bins>=1, or bin_edges, or bin_width.")
            edges = np.linspace(lo, hi, int(n_bins) + 1, dtype=float)

    # ---- bin + summarize ----
    b = pd.cut(s, bins=edges, include_lowest=True, right=False)
    tmp = pd.DataFrame({"strength": s.values, "lick": y.values, "bin": b}).dropna(subset=["bin"])
    if len(tmp) == 0:
        raise ValueError("All trials fell outside the bins (check bin_edges/bin_range).")

    grp = tmp.groupby("bin", observed=True)["lick"]
    summ = grp.agg(N="count", NLick="sum").reset_index()
    summ["Plick"] = summ["NLick"] / summ["N"].clip(lower=1)
    summ["SEM"] = np.sqrt(summ["Plick"] * (1 - summ["Plick"]) / summ["N"].clip(lower=1))

    # ---- robust Interval edge extraction (avoids Categorical math) ----
    if "bin" not in summ.columns:
        raise RuntimeError(f"Expected a 'bin' column in summary table, got: {summ.columns.tolist()}")

    # Convert to an IntervalArray explicitly
    intervals = pd.arrays.IntervalArray(summ["bin"].astype("object"))
    summ["BinLeft"] = intervals.left.astype(float)
    summ["BinRight"] = intervals.right.astype(float)
    summ["BinCenter"] = (summ["BinLeft"] + summ["BinRight"]) / 2.0

    summ = summ.drop(columns=["bin"]).sort_values("BinCenter").reset_index(drop=True)

    # ---- plot ----
    plt.figure()
    plt.errorbar(summ["BinCenter"], summ["Plick"], yerr=summ["SEM"], fmt="o-")
    plt.xlabel("Stimulus strength (SE_strength)")
    plt.ylabel("P(lick)")
    plt.ylim(-0.02, 1.02)
    plt.title(title)
    plt.show()

    return summ
