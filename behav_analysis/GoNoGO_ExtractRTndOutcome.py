import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_reaction_time_columns(
    df: pd.DataFrame,
    stim_times_col: str = "SE_Time_ms",
    trstart_col: str = "TrStartTime",
    lick_col: str = "LickTimes",
    early_lick_cutoff_ms: float = 500.0,  # ignore licks >= 500 ms before stim onset
    max_abs_lick: float = 1.10e9,
    max_trial_ms: float = 20000.0,
    lick_times_are_absolute: bool = True,  # if False, assumes lick times are already trial-relative
    plot: bool = True,
    bins: int = 100,
):
    """
    Adds columns:
      - FirstStim_ms        : min(SE_Time_ms) per trial (trial-relative, ms)
      - FirstLickAny_ms     : first lick after trial start (trial-relative, ms)
      - EarlyLicksRemoved   : True if any lick was <= (FirstStim_ms - early_lick_cutoff_ms)
      - FirstLick_ms        : first lick after removing 'too-early' licks
      - RT_ms               : FirstLick_ms - FirstStim_ms

    Plots 1 histogram: RT_ms (finite only).
    Returns: (df_out, fig_or_None)
    """
    out = df.copy()

    # --- checks ---
    for col in [stim_times_col, trstart_col, lick_col]:
        if col not in out.columns:
            raise KeyError(f"Missing column: {col}")

    # --- First stim (already trial-relative) ---
    se_lists = out[stim_times_col].tolist()
    first_stim = np.full(len(out), np.nan, dtype=float)

    for i, L in enumerate(se_lists):
        if not isinstance(L, (list, tuple, np.ndarray, pd.Series)) or len(L) == 0:
            continue
        arr = np.asarray(L, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = arr[(arr >= 0) & (arr <= max_trial_ms)]
        if arr.size == 0:
            continue
        first_stim[i] = float(np.min(arr))

    out["FirstStim_ms"] = first_stim

    # --- First lick (absolute -> trial-relative), then filter "too-early" licks ---
    t0 = pd.to_numeric(out[trstart_col], errors="coerce").to_numpy()
    lick_lists = out[lick_col].tolist()

    first_lick_any = np.full(len(out), np.nan, dtype=float)
    first_lick     = np.full(len(out), np.nan, dtype=float)
    early_removed  = np.zeros(len(out), dtype=bool)

    for i, L in enumerate(lick_lists):
        if not isinstance(L, (list, tuple, np.ndarray, pd.Series)) or len(L) == 0:
            continue

        arr = np.asarray(L, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = arr[(arr > 0) & (arr < max_abs_lick)]
        if arr.size == 0:
            continue

        if lick_times_are_absolute:
            if not np.isfinite(t0[i]):
                continue
            rel = arr - t0[i]
        else:
            rel = arr.copy()

        rel = rel[np.isfinite(rel) & (rel >= 0) & (rel <= max_trial_ms)]
        if rel.size == 0:
            continue

        # unfiltered first lick
        first_lick_any[i] = float(np.min(rel))

        # filter: remove licks that are >= cutoff ms before stim
        stim = first_stim[i]
        rel_valid = rel

        if np.isfinite(stim) and (early_lick_cutoff_ms is not None) and (early_lick_cutoff_ms > 0):
            cutoff_time = stim - early_lick_cutoff_ms
            too_early = (rel_valid <= cutoff_time)  # <= means "500 ms or more before stim" is excluded
            early_removed[i] = bool(np.any(too_early))
            rel_valid = rel_valid[~too_early]

        if rel_valid.size == 0:
            continue

        first_lick[i] = float(np.min(rel_valid))

    out["FirstLickAny_ms"] = first_lick_any
    out["EarlyLicksRemoved"] = early_removed
    out["FirstLick_ms"] = first_lick

    # --- Reaction time ---
    rt = out["FirstLick_ms"].to_numpy() - out["FirstStim_ms"].to_numpy()
    rt[~np.isfinite(rt)] = np.nan
    out["RT_ms"] = rt

    fig = None
    if plot:
        rt_valid = rt[np.isfinite(rt)]
        fig = plt.figure()
        plt.hist(rt_valid, bins=bins)
        plt.xlabel("RT = FirstLick_ms - FirstStim_ms (ms)")
        plt.ylabel("Count")
        plt.title(f"Reaction time (excluding licks ≤ stim-{int(early_lick_cutoff_ms)} ms)")
        plt.show()

    return out, fig


def add_gng_outcomes(
    df: pd.DataFrame,
    stim_strength_col: str = "SE_strength",
    trstart_col: str = "TrStartTime",
    rw_start_col: str = "RWStartTime",
    rw_end_col: str = "RWEndTime",
    lick_col: str = "LickTimes",
    orig_outcome_col: str | None = None,   # auto-detect if None
    max_abs_lick: float = 1.10e9,
    max_trial_ms: float = 20000.0,
    lick_times_are_absolute: bool = True,  # if False, assumes lick times are already trial-relative
    plot: bool = True,
):
    """
    Go/No-Go-style outcomes using lick presence in the Response Window (RW),
    while preserving original special codes:
      - 5 -> AutoReward
      - 6 -> PrematureLick
    (these are NOT relabeled to Hit/Miss/FA/CR)

    Adds columns:
      - RW_lo_ms, RW_hi_ms   : response window bounds (trial-relative, ms)
      - LickInRW             : any lick occurs within RW (uses ALL licks)
      - Outcome_GNG          : string label
      - Outcome_GNG_Code     : numeric code (Hit=1, Miss=2, FA=3, CR=4, AutoReward=5, PrematureLick=6)

    Produces 1 bar plot: proportions of each outcome.
    Returns: (df_out, summary_df, fig_or_None)
    """
    out = df.copy()

    # --- checks ---
    req = [stim_strength_col, trstart_col, rw_start_col, rw_end_col, lick_col]
    for col in req:
        if col not in out.columns:
            raise KeyError(f"Missing column: {col}")

    # detect outcome column if needed
    if orig_outcome_col is None:
        for cand in ["TrOutcome", "Outcome", "outcome", "TrialOutcome", "trial_outcome"]:
            if cand in out.columns:
                orig_outcome_col = cand
                break
    if orig_outcome_col is None or orig_outcome_col not in out.columns:
        raise KeyError("Missing original outcome column (e.g., TrOutcome/Outcome/TrialOutcome)")

    orig_outcome = pd.to_numeric(out[orig_outcome_col], errors="coerce").to_numpy()

    # trial start and RW bounds (absolute -> trial-relative)
    t0  = pd.to_numeric(out[trstart_col], errors="coerce").to_numpy()
    rwS = pd.to_numeric(out[rw_start_col], errors="coerce").to_numpy()
    rwE = pd.to_numeric(out[rw_end_col], errors="coerce").to_numpy()

    rw0 = rwS - t0
    rw1 = rwE - t0
    rw_lo = np.minimum(rw0, rw1)
    rw_hi = np.maximum(rw0, rw1)

    rw_lo = np.where(np.isfinite(rw_lo), np.clip(rw_lo, 0, max_trial_ms), np.nan)
    rw_hi = np.where(np.isfinite(rw_hi), np.clip(rw_hi, 0, max_trial_ms), np.nan)

    # lick in RW (uses ALL licks)
    lick_lists = out[lick_col].tolist()
    lick_in_rw = np.zeros(len(out), dtype=bool)

    for i, L in enumerate(lick_lists):
        if not (np.isfinite(rw_lo[i]) and np.isfinite(rw_hi[i])):
            continue
        if lick_times_are_absolute and (not np.isfinite(t0[i])):
            continue
        if not isinstance(L, (list, tuple, np.ndarray, pd.Series)) or len(L) == 0:
            continue

        arr = np.asarray(L, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = arr[(arr > 0) & (arr < max_abs_lick)]
        if arr.size == 0:
            continue

        if lick_times_are_absolute:
            rel = arr - t0[i]
        else:
            rel = arr.copy()

        rel = rel[np.isfinite(rel) & (rel >= 0) & (rel <= max_trial_ms)]
        if rel.size == 0:
            continue

        lick_in_rw[i] = bool(np.any((rel >= rw_lo[i]) & (rel <= rw_hi[i])))

    out["RW_lo_ms"] = rw_lo
    out["RW_hi_ms"] = rw_hi
    out["LickInRW"] = lick_in_rw

    # stim vs no-stim
    se = pd.to_numeric(out[stim_strength_col], errors="coerce").to_numpy()
    stim_trial   = np.isfinite(se) & (se > 0)
    nostim_trial = np.isfinite(se) & (se == 0)
    rw_valid = np.isfinite(rw_lo) & np.isfinite(rw_hi)

    # preserve special outcomes first
    is_autoreward = np.isfinite(orig_outcome) & (orig_outcome == 5)
    is_premature  = np.isfinite(orig_outcome) & (orig_outcome == 6)

    outcome = np.full(len(out), None, dtype=object)
    outcome[is_autoreward] = "AutoReward"
    outcome[is_premature]  = "PrematureLick"

    # label remaining trials (valid RW, stim/no-stim, not special)
    can_label = rw_valid & (stim_trial | nostim_trial) & (~is_autoreward) & (~is_premature)

    outcome[can_label & stim_trial   & ( lick_in_rw)] = "Hit"
    outcome[can_label & stim_trial   & (~lick_in_rw)] = "Miss"
    outcome[can_label & nostim_trial & ( lick_in_rw)] = "FalseAlarm"
    outcome[can_label & nostim_trial & (~lick_in_rw)] = "CorrectRejection"

    out["Outcome_GNG"] = outcome

    # numeric code mapping (kept separate from original outcome)
    code_map = {
        "Hit": 1,
        "Miss": 2,
        "FalseAlarm": 3,
        "CorrectRejection": 4,
        "AutoReward": 5,
        "PrematureLick": 6,
    }
    out["Outcome_GNG_Code"] = pd.Series(out["Outcome_GNG"]).map(code_map).astype("float")

    # summary (include special even if RW invalid)
    mask_eval = (is_autoreward | is_premature) | (rw_valid & (stim_trial | nostim_trial))
    order = ["Hit", "Miss", "FalseAlarm", "CorrectRejection", "AutoReward", "PrematureLick"]

    counts = pd.Series(out.loc[mask_eval, "Outcome_GNG"]).value_counts().reindex(order, fill_value=0)
    props  = counts / max(int(mask_eval.sum()), 1)
    summary = pd.DataFrame({"count": counts, "proportion": props})

    fig = None
    if plot:
        fig = plt.figure()
        plt.bar(summary.index, summary["proportion"].to_numpy())
        plt.ylabel("Proportion")
        plt.title("Outcomes (Go/No-Go remap + AutoReward/Premature)")
        plt.xticks(rotation=20, ha="right")
        plt.ylim(0, 1)
        plt.show()

    return out, summary, fig


def add_stim_isi_metrics(
    df: pd.DataFrame,
    stim_strength_col: str = "SE_strength",
    stim_times_col: str = "SE_Time_ms",
    max_trial_ms: float = 20000.0,
):
    """
    Stim trials only (stim_strength_col > 0). Uses stim_times_col as trial-relative ms list.

    Adds columns:
      - Stim_N             : number of (unique, sorted) stim times in trial
      - Stim_MedianISI_ms  : median(diff(sorted stim times))
      - Stim_ISI1_ms       : (2nd stim - 1st stim)
    """
    out = df.copy()

    for col in [stim_strength_col, stim_times_col]:
        if col not in out.columns:
            raise KeyError(f"Missing column: {col}")

    se = pd.to_numeric(out[stim_strength_col], errors="coerce").to_numpy()
    stim_trials = np.isfinite(se) & (se > 0)

    se_lists = out[stim_times_col].tolist()

    med_isi = np.full(len(out), np.nan, dtype=float)
    isi1    = np.full(len(out), np.nan, dtype=float)
    n_stim  = np.zeros(len(out), dtype=int)

    for i, L in enumerate(se_lists):
        if not stim_trials[i]:
            continue
        if not isinstance(L, (list, tuple, np.ndarray, pd.Series)) or len(L) == 0:
            continue

        arr = np.asarray(L, dtype=float)
        arr = arr[np.isfinite(arr)]
        arr = arr[(arr >= 0) & (arr <= max_trial_ms)]
        if arr.size == 0:
            continue

        t = np.unique(np.sort(arr))
        n_stim[i] = t.size

        if t.size >= 2:
            diffs = np.diff(t)
            diffs = diffs[np.isfinite(diffs) & (diffs >= 0)]
            if diffs.size:
                med_isi[i] = float(np.median(diffs))
                isi1[i]    = float(t[1] - t[0])

    out["Stim_N"] = n_stim
    out["Stim_MedianISI_ms"] = med_isi
    out["Stim_ISI1_ms"] = isi1

    return out