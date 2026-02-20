import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def plot_learning_curve_2afc(
    df: pd.DataFrame,
    window: int = 200,
    outcome_col: str = "TrOutcome",
    arduino_col: str = "ArduinoMode",
    session_col: str = "SessionBase",
):
    """
    2AFC learning curve:
      - plots rolling Correct rate and Incorrect rate (over trials with TrOutcome in {7,8})
      - overlays ArduinoMode==3 and ArduinoMode==4 curves (separate lines)
      - x-axis is chronological trial index; weekly date ticks from SessionBase (mmddyy)

    TrOutcome codes:
      7 = correct, 8 = incorrect, 5 = autoreward, 6 = premature, 9 = no-lick (ignored for performance rate)

    Single parameter: window (rolling-average size in trials)
    """
    if window is None or int(window) < 1:
        raise ValueError("window must be an integer >= 1")
    window = int(window)

    for col in [outcome_col, arduino_col, session_col]:
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    d = df.copy()

    # --- parse session date (mmddyy) from SessionBase ---
    def _extract_mmddyy(base):
        if base is None:
            return np.nan
        m = re.search(r"(\d{6})", str(base))
        return m.group(1) if m else np.nan

    mmddyy = d[session_col].apply(_extract_mmddyy)
    sess_date = pd.to_datetime(mmddyy, format="%m%d%y", errors="coerce")
    d["_sess_date"] = sess_date

    # --- sort chronologically (date, then SessionBase, then Trial if present) ---
    d["_orig_idx"] = np.arange(len(d))
    if "Trial" in d.columns:
        d["_trialnum"] = pd.to_numeric(d["Trial"], errors="coerce")
    else:
        d["_trialnum"] = np.nan

    # if dates are missing, this still gives a stable order
    d = d.sort_values(by=["_sess_date", session_col, "_trialnum", "_orig_idx"], na_position="last")
    d["_t"] = np.arange(len(d))  # trial index

    # --- define evaluated trials (only correct/incorrect) ---
    outc = pd.to_numeric(d[outcome_col], errors="coerce")
    is_corr = (outc == 7)
    is_inc  = (outc == 8)
    eval_mask = (is_corr | is_inc).astype(float)

    # rolling denom (evaluated trials only)
    minp = max(5, window // 5) if window >= 5 else 1
    denom = eval_mask.rolling(window, min_periods=minp, center=True).sum()

    def _roll_rate(mask_bool: pd.Series) -> np.ndarray:
        num = mask_bool.astype(float).rolling(window, min_periods=minp, center=True).sum()
        rate = num / denom
        return rate.where(denom > 0, np.nan).to_numpy()

    # overall
    r_corr_all = _roll_rate(is_corr)
    r_inc_all  = _roll_rate(is_inc)

    # ArduinoMode split
    am = pd.to_numeric(d[arduino_col], errors="coerce")
    mode3 = (am == 3)
    mode4 = (am == 4)

    # For mode-specific rates, denominator is evaluated trials within that mode
    denom3 = (eval_mask * mode3.astype(float)).rolling(window, min_periods=minp, center=True).sum()
    denom4 = (eval_mask * mode4.astype(float)).rolling(window, min_periods=minp, center=True).sum()

    def _roll_rate_mode(mask_bool: pd.Series, mode_mask: pd.Series, denom_mode: pd.Series) -> np.ndarray:
        num = (mask_bool & mode_mask).astype(float).rolling(window, min_periods=minp, center=True).sum()
        rate = num / denom_mode
        return rate.where(denom_mode > 0, np.nan).to_numpy()

    r_corr_m3 = _roll_rate_mode(is_corr, mode3, denom3)
    r_inc_m3  = _roll_rate_mode(is_inc,  mode3, denom3)
    r_corr_m4 = _roll_rate_mode(is_corr, mode4, denom4)
    r_inc_m4  = _roll_rate_mode(is_inc,  mode4, denom4)

    x = d["_t"].to_numpy()

    # --- weekly ticks from session dates ---
    # pick one tick per week (first trial of each ISO week)
    tick_pos = []
    tick_lab = []
    if d["_sess_date"].notna().any():
        week_id = d["_sess_date"].dt.isocalendar().week.astype("Int64")
        year_id = d["_sess_date"].dt.isocalendar().year.astype("Int64")
        wk_key = (year_id.astype(str) + "-" + week_id.astype(str)).fillna("nan")

        first_of_week = wk_key.ne(wk_key.shift(1))
        idxs = d.loc[first_of_week & d["_sess_date"].notna(), "_t"].to_numpy()

        # don’t overcrowd: keep roughly weekly, but if super dense, subsample
        for xi in idxs:
            tick_pos.append(int(xi))
            dt = d.loc[d["_t"] == xi, "_sess_date"].iloc[0]
            tick_lab.append(dt.strftime("%m/%d"))

        # If there are too many, keep every other tick
        if len(tick_pos) > 20:
            tick_pos = tick_pos[::2]
            tick_lab = tick_lab[::2]

    # --- plot ---
    plt.figure(figsize=(12, 5))

    # overall (thicker)
    plt.plot(x, r_corr_all, label="Correct (all)")
    plt.plot(x, r_inc_all,  label="Incorrect (all)")

    # overlays for modes 3 and 4 (same quantities, mode-specific)
    plt.plot(x, r_corr_m3, label="Correct (ArduinoMode=3)")
    plt.plot(x, r_inc_m3,  label="Incorrect (ArduinoMode=3)")
    plt.plot(x, r_corr_m4, label="Correct (ArduinoMode=4)")
    plt.plot(x, r_inc_m4,  label="Incorrect (ArduinoMode=4)")

    plt.ylim(0, 1)
    plt.ylabel("Rate (rolling window over evaluated 2AFC trials)")
    plt.xlabel("Trial index (chronological)")
    plt.title(f"2AFC learning curve (rolling window = {window} trials)")
    plt.legend(ncol=2, fontsize=9)

    if tick_pos:
        plt.xticks(tick_pos, tick_lab, rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

    # return the sorted df in case you want to align other signals to the same x-axis
    return d.drop(columns=["_orig_idx", "_trialnum"], errors="ignore")