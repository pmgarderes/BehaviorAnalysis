import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def plot_learning_curve_gng(df: pd.DataFrame, window: int = 200):
    """
    Learning curve using Outcome_GNG_Code (1..6) with a single parameter: moving-average window size.

    Plots:
      (A) moving-window fractions of outcomes (Hit/Miss/FA/CR/AutoReward/Premature)
      (B) SE_strength over the same trial index (raw + moving average)

    Assumes these columns exist:
      - Outcome_GNG_Code (float/int codes 1..6)
      - Outcome_GNG (optional; only used for legend sanity)
      - SE_strength
      - SessionBase (optional, used to sort + draw session boundaries)
      - Trial (optional, used to break ties within a session)
    """
    if window is None or int(window) < 1:
        raise ValueError("window must be an integer >= 1")
    window = int(window)

    for col in ["Outcome_GNG_Code", "SE_strength"]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col} (run your add_* functions first)")

    d = df.copy()

    # ---------- build a reasonable chronological order ----------
    # Prefer parsing mmddyy from SessionBase; fall back to original order if not possible.
    if "SessionBase" in d.columns:
        def _extract_mmddyy(base):
            if base is None:
                return np.nan
            m = re.search(r"(\d{6})", str(base))
            return m.group(1) if m else np.nan

        mmddyy = d["SessionBase"].apply(_extract_mmddyy)
        sess_date = pd.to_datetime(mmddyy, format="%m%d%y", errors="coerce")

        d["_sess_date"] = sess_date
        # trial sort key if present
        if "Trial" in d.columns:
            trn = pd.to_numeric(d["Trial"], errors="coerce")
        else:
            trn = pd.Series(np.arange(len(d)), index=d.index, dtype=float)

        # stable fallback order
        d["_orig_idx"] = np.arange(len(d))

        # If we have at least some valid dates, sort by date then SessionBase then Trial
        if np.isfinite(d["_sess_date"].astype("int64", errors="ignore")).any():
            d = d.sort_values(by=["_sess_date", "SessionBase", trn.name if trn.name in d.columns else "_orig_idx", "_orig_idx"])
        else:
            # no valid dates -> keep original order
            d = d.sort_values(by=["_orig_idx"])

        d["_trial_order"] = np.arange(len(d))

        # session boundaries for plotting
        sess_change_idx = None
        if "SessionBase" in d.columns:
            sess_change = d["SessionBase"].astype(str).ne(d["SessionBase"].astype(str).shift(1))
            sess_change_idx = d.loc[sess_change, "_trial_order"].to_numpy()
    else:
        d["_trial_order"] = np.arange(len(d))
        sess_change_idx = None

    x = d["_trial_order"].to_numpy()

    # ---------- rolling outcome fractions (normalized to labeled trials only) ----------
    codes = pd.to_numeric(d["Outcome_GNG_Code"], errors="coerce")
    valid = codes.isin([1, 2, 3, 4, 5, 6]).astype(float)

    minp = max(5, window // 5) if window >= 5 else 1
    denom = valid.rolling(window, min_periods=minp, center=True).sum()

    code_names = {
        1: "Hit",
        2: "Miss",
        3: "FalseAlarm",
        4: "CorrectRejection",
        5: "AutoReward",
        6: "PrematureLick",
    }

    frac = {}
    for k in [1, 2, 3, 4, 5, 6]:
        num = (codes == k).astype(float).rolling(window, min_periods=minp, center=True).sum()
        f = num / denom
        f = f.where(denom > 0, np.nan)
        frac[k] = f.to_numpy()

    # ---------- SE_strength time course ----------
    strength = pd.to_numeric(d["SE_strength"], errors="coerce")
    strength_ma = strength.rolling(window, min_periods=minp, center=True).mean().to_numpy()

    # ---------- plots ----------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Outcome fractions
    for k in [1, 2, 3, 4, 5, 6]:
        ax1.plot(x, frac[k], label=code_names[k])
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Fraction (moving window)")
    ax1.set_title(f"Learning curve (window={window} trials)")
    ax1.legend(ncol=3, fontsize=9)

    # Session boundaries (optional)
    if sess_change_idx is not None and len(sess_change_idx) > 1:
        for xi in sess_change_idx[1:]:
            ax1.axvline(xi, linewidth=0.8, alpha=0.3)
            ax2.axvline(xi, linewidth=0.8, alpha=0.3)

    # Strength
    ax2.plot(x, strength_ma, label="SE_strength (moving avg)")
    ax2.scatter(x, strength.to_numpy(), s=6, alpha=0.25, label="SE_strength (raw)")
    ax2.set_ylabel("SE_strength")
    ax2.set_xlabel("Trial index (sorted)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

    # Return sorted df (useful for debugging / aligning other signals)
    d = d.drop(columns=[c for c in ["_sess_date", "_orig_idx"] if c in d.columns], errors="ignore")
    return d, fig