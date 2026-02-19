from __future__ import annotations

import re
import pandas as pd


def filter_trials(
    trial_data: pd.DataFrame,
    strength_min: float | None = None,
    strength_max: float | None = None,
    n_elem_min: int | None = None,
    n_elem_max: int | None = None,
    use_NoGo: bool = False,
    date_min: str | None = None,   # "mmddyy" e.g. "011626"
    date_max: str | None = None,   # "mmddyy"
    include: list[str] | None = None,  # ["autoreward","premature","correct","incorrect"]
) -> pd.DataFrame:
    """
    Return a filtered copy of trial_data.

    Filters:
      - strength_min/max on SE_strength
      - n_elem_min/max on number of DISTINCT stimulated elements (from SE_StimElem)
      - use_NoGo:
          if True, always keep NoGo trials (SE_strength==0) regardless of strength/n_elem filters
      - date_min/date_max:
          filter by date parsed from SessionBase as mmddyy, e.g. "..._011626_..."
      - include:
          optional whitelist by trial_outcome codes:
            "autoreward" -> trial_outcome == 5
            "premature"  -> trial_outcome == 6
            "correct"    -> trial_outcome in {1,2,3,4,7}  (GNG correct codes unknown here; 7=2AFC correct)
            "incorrect"  -> trial_outcome in {8}          (2AFC incorrect)
          NOTE: adjust if you want stricter GNG correct/incorrect mapping.
    """
    df = trial_data.copy()

    # --- helpers ---
    def _parse_mmddyy(s: str) -> pd.Timestamp | pd.NaT:
        if s is None:
            return pd.NaT
        s = str(s).strip()
        if not re.fullmatch(r"\d{6}", s):
            return pd.NaT
        return pd.to_datetime(s, format="%m%d%y", errors="coerce")

    def _extract_date_from_base(base: str) -> pd.Timestamp | pd.NaT:
        # find first 6-digit token that looks like mmddyy
        if base is None:
            return pd.NaT
        m = re.search(r"(\d{6})", str(base))
        if not m:
            return pd.NaT
        return _parse_mmddyy(m.group(1))

    # --- compute basic quantities ---
    se_strength = pd.to_numeric(df.get("SE_strength", 0), errors="coerce").fillna(0.0)
    is_nogo = se_strength == 0

    # N distinct stimulated elements (noise already removed upstream)
    if "SE_StimElem" in df.columns:
        def _n_distinct(x):
            if not isinstance(x, (list, tuple)):
                return 0
            vals = [v for v in x if v is not None and pd.notna(v)]
            return len(set(vals))
        n_elem = df["SE_StimElem"].apply(_n_distinct)
    else:
        n_elem = pd.Series(0, index=df.index)

    keep = pd.Series(True, index=df.index)

    # --- date filtering ---
    if date_min is not None or date_max is not None:
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

    # --- strength filtering ---
    strength_keep = pd.Series(True, index=df.index)
    if strength_min is not None:
        strength_keep &= se_strength >= float(strength_min)
    if strength_max is not None:
        strength_keep &= se_strength <= float(strength_max)

    # --- n-elem filtering ---
    nelem_keep = pd.Series(True, index=df.index)
    if n_elem_min is not None:
        nelem_keep &= n_elem >= int(n_elem_min)
    if n_elem_max is not None:
        nelem_keep &= n_elem <= int(n_elem_max)

    # combine these filters
    stim_filters = strength_keep & nelem_keep

    # use_NoGo override: keep NoGo regardless of stim_filters
    if use_NoGo:
        keep &= (stim_filters | is_nogo)
    else:
        keep &= stim_filters

    # --- include by trial_outcome code ---
    if include is not None:
        include = [str(x).strip().lower() for x in include]

        # find the outcome column (your data often has TrOutcome)
        outcome_col = next((c for c in ["trial_outcome", "TrOutcome", "TrialOutcome"] if c in df.columns), None)
        if outcome_col is None:
            raise KeyError(
                "Could not find trial outcome column. Expected one of: "
                "['trial_outcome', 'TrOutcome', 'TrialOutcome'].\n"
                f"Available columns include: {list(df.columns)[:30]} ..."
            )

        outcome = pd.to_numeric(df[outcome_col], errors="coerce")  # <-- guaranteed Series

        allowed = pd.Series(False, index=df.index)

        # Your reliable mapping:
        # 5=Autoreward, 6=Premature LickAbort, 7=2AFC correct, 8=2AFC incorrect, 9=2AFC NoLick
        if "autoreward" in include:
            allowed |= (outcome == 5)
        if "premature" in include:
            allowed |= (outcome == 6)
        if "correct" in include:
            # NOTE: for 2AFC this is definitely 7; GNG (1-4) mapping depends on your convention
            allowed |= outcome.isin([7, 1, 2, 3, 4])
        if "incorrect" in include:
            allowed |= outcome.isin([8])
        if "nolick" in include:
            allowed |= outcome.isin([9])

        keep &= allowed

    return df.loc[keep].copy()
