from __future__ import annotations

import re
import pandas as pd


def filter_trials_2afc(
    trial_data: pd.DataFrame,
    strength_min: float | None = None,
    strength_max: float | None = None,
    n_elem_min: int | None = None,
    n_elem_max: int | None = None,
    date_min: str | None = None,   # "mmddyy" e.g. "011626"
    date_max: str | None = None,   # "mmddyy"
    include: list[str] | None = None,  # ["autoreward","premature","correct","incorrect","nolick"]
    arduino_mode: int | str | list[int | str] | None = None,
    arduino_mode_col: str = "ArduinoMode",
    outcome_col: str = "TrOutcome",
) -> pd.DataFrame:
    """
    Return a filtered copy of trial_data for 2AFC sessions.

    Filters:
      - strength_min/max on SE_strength
      - n_elem_min/max on number of DISTINCT stimulated elements (from SE_StimElem)
      - date_min/date_max by date parsed from SessionBase as mmddyy, e.g. "..._011626_..."
      - arduino_mode: keep only rows where ArduinoMode matches (single value or list)
      - include: whitelist by TrOutcome codes (2AFC)
          "autoreward" -> 5
          "premature"  -> 6
          "correct"    -> 7
          "incorrect"  -> 8
          "nolick"     -> 9
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
        if base is None:
            return pd.NaT
        m = re.search(r"(\d{6})", str(base))
        if not m:
            return pd.NaT
        return _parse_mmddyy(m.group(1))

    # --- compute basics ---
    se_strength = pd.to_numeric(df.get("SE_strength", 0), errors="coerce").fillna(0.0)

    # N distinct stimulated elements
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
        if "SessionBase" not in df.columns:
            raise KeyError("Missing column: SessionBase (needed for date filters).")

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
    if strength_min is not None:
        keep &= se_strength >= float(strength_min)
    if strength_max is not None:
        keep &= se_strength <= float(strength_max)

    # --- n-elem filtering ---
    if n_elem_min is not None:
        keep &= n_elem >= int(n_elem_min)
    if n_elem_max is not None:
        keep &= n_elem <= int(n_elem_max)

    # --- ArduinoMode filtering ---
    if arduino_mode is not None:
        if arduino_mode_col not in df.columns:
            raise KeyError(f"Missing column: {arduino_mode_col}")

        am = df[arduino_mode_col]

        # Allow numeric or string matching; list allowed
        if isinstance(arduino_mode, (list, tuple, set)):
            allowed_modes = set(str(x) for x in arduino_mode)
            keep &= am.astype(str).isin(allowed_modes)
        else:
            keep &= am.astype(str) == str(arduino_mode)

    # --- include by TrOutcome (2AFC codes only) ---
    if include is not None:
        if outcome_col not in df.columns:
            raise KeyError(f"Missing outcome column: {outcome_col} (expected for 2AFC).")

        include = [str(x).strip().lower() for x in include]
        outcome = pd.to_numeric(df[outcome_col], errors="coerce")

        allowed = pd.Series(False, index=df.index)

        if "autoreward" in include:
            allowed |= (outcome == 5)
        if "premature" in include:
            allowed |= (outcome == 6)
        if "correct" in include:
            allowed |= (outcome == 7)
        if "incorrect" in include:
            allowed |= (outcome == 8)
        if "nolick" in include:
            allowed |= (outcome == 9)

        keep &= allowed

    return df.loc[keep].copy()