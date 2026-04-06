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
    date_min: str | None = None,
    date_max: str | None = None,
    # ---- behavioral outcome ----
    outcomes: list[int | str] | None = None,
    outcome_col: str | None = None,
    # ---- RT filters ----
    rt_min: float | None = None,
    rt_max: float | None = None,
    rt_col: str = "RT_ms",
    rt_include_nan: bool = False,   # True = also keep trials with no RT
    # ---- ISI filters ----
    isi1_min: float | None = None,
    isi1_max: float | None = None,
    isi1_include_nan: bool = False,  # True = also keep trials with no ISI1
    median_isi_min: float | None = None,
    median_isi_max: float | None = None,
    median_isi_include_nan: bool = False,  # True = also keep trials with no median ISI
    isi1_col: str = "Stim_ISI1_ms",
    median_isi_col: str = "Stim_MedianISI_ms",
    # ---- reporting ----
    verbose: bool = True,           # print the filtering report
) -> pd.DataFrame:
    """
    Return a filtered copy of trial_data, with an optional report showing
    how many trials each filter removed *uniquely* (i.e. trials that passed
    every other filter but were removed by this one alone).

    New parameters vs previous version
    ------------------------------------
    rt_include_nan        : keep trials with RT=NaN alongside the rt window
    isi1_include_nan      : keep trials with ISI1=NaN alongside the isi1 window
    median_isi_include_nan: keep trials with median ISI=NaN alongside the window
    verbose               : print per-filter removal report (default True)
    """
    df = trial_data.copy()
    n_total = len(df)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _parse_mmddyy(s):
        if s is None:
            return pd.NaT
        s = str(s).strip()
        if not re.fullmatch(r"\d{6}", s):
            return pd.NaT
        return pd.to_datetime(s, format="%m%d%y", errors="coerce")

    def _extract_date_from_base(base):
        if base is None:
            return pd.NaT
        m = re.search(r"(\d{6})", str(base))
        return pd.NaT if not m else _parse_mmddyy(m.group(1))

    # ------------------------------------------------------------------ #
    # base quantities
    # ------------------------------------------------------------------ #
    se_strength = pd.to_numeric(df.get("SE_strength", 0), errors="coerce").fillna(0.0)
    is_nogo  = se_strength == 0
    is_stim  = se_strength > 0

    if "SE_StimElem" in df.columns:
        def _n_distinct(x):
            if not isinstance(x, (list, tuple, np.ndarray, pd.Series)):
                return 0
            return len({v for v in x if v is not None and pd.notna(v)})
        n_elem = df["SE_StimElem"].apply(_n_distinct)
    else:
        n_elem = pd.Series(0, index=df.index)

    # ------------------------------------------------------------------ #
    # build one boolean mask per filter group
    # ------------------------------------------------------------------ #
    masks = {}   # name -> pd.Series[bool]  (True = trial PASSES this filter)

    # ---- date ----
    if date_min is not None or date_max is not None:
        if "SessionBase" not in df.columns:
            raise KeyError("Missing column: SessionBase (needed for date_min/date_max).")
        sess_date = df["SessionBase"].apply(_extract_date_from_base)
        dmin = _parse_mmddyy(date_min) if date_min is not None else pd.NaT
        dmax = _parse_mmddyy(date_max) if date_max is not None else pd.NaT
        if date_min is not None and pd.isna(dmin):
            raise ValueError(f"date_min='{date_min}' is not valid mmddyy.")
        if date_max is not None and pd.isna(dmax):
            raise ValueError(f"date_max='{date_max}' is not valid mmddyy.")
        m = pd.Series(True, index=df.index)
        if date_min is not None:
            m &= sess_date >= dmin
        if date_max is not None:
            m &= sess_date <= dmax
        masks["date"] = m

    # ---- strength / n_elem ----
    strength_keep = pd.Series(True, index=df.index)
    if strength_min is not None:
        strength_keep &= se_strength >= float(strength_min)
    if strength_max is not None:
        strength_keep &= se_strength <= float(strength_max)

    nelem_keep = pd.Series(True, index=df.index)
    if n_elem_min is not None:
        nelem_keep &= n_elem >= int(n_elem_min)
    if n_elem_max is not None:
        nelem_keep &= n_elem <= int(n_elem_max)

    stim_filters = strength_keep & nelem_keep
    if use_NoGo:
        masks["stimulus"] = stim_filters | is_nogo
    else:
        masks["stimulus"] = stim_filters

    # ---- outcomes ----
    if outcomes is not None:
        _ocol = outcome_col
        if _ocol is None:
            _ocol = "Outcome_GNG_Code" if "Outcome_GNG_Code" in df.columns else None
        if _ocol is None or _ocol not in df.columns:
            raise KeyError(
                "Could not find outcome column. Expected Outcome_GNG_Code."
            )
        outc = pd.to_numeric(df[_ocol], errors="coerce")
        name2code = {
            "hit": 1, "miss": 2,
            "fa": 3, "falsealarm": 3, "false_alarm": 3,
            "cr": 4, "correctrejection": 4, "correct_rejection": 4,
            "autoreward": 5, "auto_reward": 5,
            "premature": 6, "prematurelick": 6, "premature_lick": 6,
        }
        wanted: list[int] = []
        for x in outcomes:
            if isinstance(x, str):
                k = x.strip().lower()
                if k not in name2code:
                    raise ValueError(f"Unknown outcome label '{x}'.")
                wanted.append(int(name2code[k]))
            else:
                wanted.append(int(x))
        masks["outcome"] = outc.isin(wanted)

    # ---- RT ----
    if rt_min is not None or rt_max is not None:
        if rt_col not in df.columns:
            raise KeyError(f"Missing column: {rt_col} (run add_reaction_time_columns first).")
        rt = pd.to_numeric(df[rt_col], errors="coerce")
        in_window = rt.notna()
        if rt_min is not None:
            in_window &= rt >= float(rt_min)
        if rt_max is not None:
            in_window &= rt <= float(rt_max)
        masks["RT"] = (in_window | rt.isna()) if rt_include_nan else in_window

    # ---- ISI1 ----
    if isi1_min is not None or isi1_max is not None:
        if isi1_col not in df.columns:
            raise KeyError(f"Missing column: {isi1_col} (run add_stim_isi_metrics first).")
        isi1 = pd.to_numeric(df[isi1_col], errors="coerce")
        in_window = isi1.notna()
        if isi1_min is not None:
            in_window &= isi1 >= float(isi1_min)
        if isi1_max is not None:
            in_window &= isi1 <= float(isi1_max)
        # enforce only on stim trials; nogo trials pass freely
        isi1_pass = (~is_stim) | in_window
        if isi1_include_nan:
            isi1_pass = (~is_stim) | in_window | isi1.isna()
        masks["ISI1"] = isi1_pass

    # ---- Median ISI ----
    if median_isi_min is not None or median_isi_max is not None:
        if median_isi_col not in df.columns:
            raise KeyError(f"Missing column: {median_isi_col} (run add_stim_isi_metrics first).")
        med = pd.to_numeric(df[median_isi_col], errors="coerce")
        in_window = med.notna()
        if median_isi_min is not None:
            in_window &= med >= float(median_isi_min)
        if median_isi_max is not None:
            in_window &= med <= float(median_isi_max)
        med_pass = (~is_stim) | in_window
        if median_isi_include_nan:
            med_pass = (~is_stim) | in_window | med.isna()
        masks["median_ISI"] = med_pass

    # ------------------------------------------------------------------ #
    # combine all masks → final keep
    # ------------------------------------------------------------------ #
    keep = pd.Series(True, index=df.index)
    for m in masks.values():
        keep &= m

    # ------------------------------------------------------------------ #
    # verbose report: unique removals per filter
    # ------------------------------------------------------------------ #
    if verbose and masks:
        print("=" * 52)
        print(f"  filter_trials_gng  —  filtering report")
        print("=" * 52)
        print(f"  Total trials in  : {n_total}")
        print(f"  Total trials out : {keep.sum()}  "
              f"({n_total - keep.sum()} removed total)\n")

        print(f"  {'Filter':<18}  {'Pass':>6}  {'Fail':>6}  {'Unique removal':>15}")
        print(f"  {'-'*18}  {'-'*6}  {'-'*6}  {'-'*15}")

        for name, this_mask in masks.items():
            # trials passing ALL OTHER filters
            other_keep = pd.Series(True, index=df.index)
            for other_name, other_mask in masks.items():
                if other_name != name:
                    other_keep &= other_mask
            # unique removal = passes all others but fails this one
            unique_removed = (other_keep & ~this_mask).sum()
            print(f"  {name:<18}  {this_mask.sum():>6}  {(~this_mask).sum():>6}  {unique_removed:>15}")

        print("=" * 52)

    return df.loc[keep].copy()
