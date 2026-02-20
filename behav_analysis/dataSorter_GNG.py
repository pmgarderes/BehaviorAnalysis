from __future__ import annotations

import re
import pandas as pd
import numpy as np


def filter_trials_gng(
    trial_data: pd.DataFrame,
    # ---- stimulus structure filters ----
    strength_min: float | None = None,
    strength_max: float | None = None,
    n_elem_min: int | None = None,
    n_elem_max: int | None = None,
    use_NoGo: bool = False,
    # ---- date filters (mmddyy inside SessionBase) ----
    date_min: str | None = None,   # "mmddyy" e.g. "011626"
    date_max: str | None = None,   # "mmddyy"
    # ---- behavioral outcome (NEW codes 1..6) ----
    outcomes: list[int | str] | None = None,
    outcome_col: str | None = None,   # default: "Outcome_GNG_Code" if present
    # ---- RT filters ----
    rt_min: float | None = None,
    rt_max: float | None = None,
    rt_col: str = "RT_ms",
    # ---- ISI filters ----
    isi1_min: float | None = None,
    isi1_max: float | None = None,
    median_isi_min: float | None = None,
    median_isi_max: float | None = None,
    isi1_col: str = "Stim_ISI1_ms",
    median_isi_col: str = "Stim_MedianISI_ms",
) -> pd.DataFrame:
    """
    Return a filtered copy of trial_data.

    Requires (if you filter on them):
      - Outcome_GNG_Code (1..6) from add_gng_outcomes()
      - RT_ms from add_reaction_time_columns()
      - Stim_ISI1_ms / Stim_MedianISI_ms from add_stim_isi_metrics()

    Outcome codes (Outcome_GNG_Code):
      1=Hit, 2=Miss, 3=FalseAlarm, 4=CorrectRejection, 5=AutoReward, 6=PrematureLick

    'outcomes' can be ints [1..6] OR strings in:
      {"hit","miss","fa","falsealarm","cr","correctrejection","autoreward","premature","prematurelick"}
    """
    df = trial_data.copy()

    # ---------------- helpers: date parsing ----------------
    def _parse_mmddyy(s: str) -> pd.Timestamp | pd.NaT:
        if s is None:
            return pd.NaT
        s = str(s).strip()
        if not re.fullmatch(r"\d{6}", s):
            return pd.NaT
        return pd.to_datetime(s, format="%m%d%y", errors="coerce")

    def _extract_date_from_base(base: str) -> pd.Timestamp | pd.NaT:
        if base is None:
            return pd.NaT
        m = re.search(r"(\d{6})", str(base))
        if not m:
            return pd.NaT
        return _parse_mmddyy(m.group(1))

    # ---------------- basic quantities ----------------
    se_strength = pd.to_numeric(df.get("SE_strength", 0), errors="coerce").fillna(0.0)
    is_nogo = se_strength == 0
    is_stim = se_strength > 0

    # N distinct stimulated elements
    if "SE_StimElem" in df.columns:
        def _n_distinct(x):
            if not isinstance(x, (list, tuple, np.ndarray, pd.Series)):
                return 0
            vals = [v for v in x if v is not None and pd.notna(v)]
            return len(set(vals))
        n_elem = df["SE_StimElem"].apply(_n_distinct)
    else:
        n_elem = pd.Series(0, index=df.index)

    keep = pd.Series(True, index=df.index)

    # ---------------- date filtering ----------------
    if date_min is not None or date_max is not None:
        if "SessionBase" not in df.columns:
            raise KeyError("Missing column: SessionBase (needed for date_min/date_max filters).")

        sess_date = df["SessionBase"].apply(_extract_date_from_base)
        dmin = _parse_mmddyy(date_min) if date_min is not None else pd.NaT
        dmax = _parse_mmddyy(date_max) if date_max is not None else pd.NaT

        if date_min is not None and pd.isna(dmin):
            raise ValueError(f"date_min='{date_min}' is not valid mmddyy.")
        if date_max is not None and pd.isna(dmax):
            raise ValueError(f"date_max='{date_max}' is not valid mmddyy.")

        if date_min is not None:
            keep &= sess_date >= dmin
        if date_max is not None:
            keep &= sess_date <= dmax

    # ---------------- strength filtering ----------------
    strength_keep = pd.Series(True, index=df.index)
    if strength_min is not None:
        strength_keep &= se_strength >= float(strength_min)
    if strength_max is not None:
        strength_keep &= se_strength <= float(strength_max)

    # ---------------- n-elem filtering ----------------
    nelem_keep = pd.Series(True, index=df.index)
    if n_elem_min is not None:
        nelem_keep &= n_elem >= int(n_elem_min)
    if n_elem_max is not None:
        nelem_keep &= n_elem <= int(n_elem_max)

    stim_filters = strength_keep & nelem_keep

    # use_NoGo override
    if use_NoGo:
        keep &= (stim_filters | is_nogo)
    else:
        keep &= stim_filters

    # ---------------- outcome filtering (NEW 1..6) ----------------
    if outcomes is not None:
        # pick outcome column
        if outcome_col is None:
            outcome_col = "Outcome_GNG_Code" if "Outcome_GNG_Code" in df.columns else None
        if outcome_col is None or outcome_col not in df.columns:
            raise KeyError(
                "Could not find outcome column for GNG filtering. "
                "Expected Outcome_GNG_Code (recommended)."
            )

        outc = pd.to_numeric(df[outcome_col], errors="coerce")

        # allow string labels
        name2code = {
            "hit": 1,
            "miss": 2,
            "fa": 3,
            "falsealarm": 3,
            "false_alarm": 3,
            "cr": 4,
            "correctrejection": 4,
            "correct_rejection": 4,
            "autoreward": 5,
            "auto_reward": 5,
            "premature": 6,
            "prematurelick": 6,
            "premature_lick": 6,
        }

        wanted_codes: list[int] = []
        for x in outcomes:
            if isinstance(x, str):
                k = x.strip().lower()
                if k not in name2code:
                    raise ValueError(f"Unknown outcome label '{x}'. Allowed: {sorted(name2code.keys())}")
                wanted_codes.append(int(name2code[k]))
            else:
                wanted_codes.append(int(x))

        keep &= outc.isin(wanted_codes)

    # ---------------- RT filtering ----------------
    if rt_min is not None or rt_max is not None:
        if rt_col not in df.columns:
            raise KeyError(f"Missing column: {rt_col} (run add_reaction_time_columns first).")

        rt = pd.to_numeric(df[rt_col], errors="coerce")
        rt_keep = rt.notna()

        if rt_min is not None:
            rt_keep &= rt >= float(rt_min)
        if rt_max is not None:
            rt_keep &= rt <= float(rt_max)

        # when RT filter is requested: require finite RT
        keep &= rt_keep

    # ---------------- ISI filtering (stim trials only) ----------------
    # (NoGo trials can still survive if use_NoGo=True; ISI constraints don't apply to them.)
    if (isi1_min is not None or isi1_max is not None) or (median_isi_min is not None or median_isi_max is not None):
        isi_keep = pd.Series(True, index=df.index)

        # ISI1
        if isi1_min is not None or isi1_max is not None:
            if isi1_col not in df.columns:
                raise KeyError(f"Missing column: {isi1_col} (run add_stim_isi_metrics first).")
            isi1 = pd.to_numeric(df[isi1_col], errors="coerce")
            k = isi1.notna()
            if isi1_min is not None:
                k &= isi1 >= float(isi1_min)
            if isi1_max is not None:
                k &= isi1 <= float(isi1_max)
            # only enforce on stim trials
            isi_keep &= (~is_stim) | k

        # Median ISI
        if median_isi_min is not None or median_isi_max is not None:
            if median_isi_col not in df.columns:
                raise KeyError(f"Missing column: {median_isi_col} (run add_stim_isi_metrics first).")
            med = pd.to_numeric(df[median_isi_col], errors="coerce")
            k = med.notna()
            if median_isi_min is not None:
                k &= med >= float(median_isi_min)
            if median_isi_max is not None:
                k &= med <= float(median_isi_max)
            isi_keep &= (~is_stim) | k

        keep &= isi_keep

    return df.loc[keep].copy()