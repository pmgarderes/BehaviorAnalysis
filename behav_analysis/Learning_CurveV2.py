from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curve_v2(
    trial_data: pd.DataFrame,
    W: int = 100,
    min_periods: int = 10,
    strength_min: float | None = None,
    strength_max: float | None = None,
    n_elem_min: int | None = None,
    n_elem_max: int | None = None,
    use_distinct_elems: bool = True,
    title: str = "Go/No-Go learning curve",
) -> pd.DataFrame:
    """
    Plots rolling fractions for:
      Hit / FalseAlarm / Miss / CorrectRejection

    Trial selection options:
      - filter by SE_strength in [strength_min, strength_max]
      - filter by number of stimulated elements in [n_elem_min, n_elem_max]
        (computed from SE_StimElem; optionally distinct)

    Returns the filtered DataFrame used for plotting (so you can inspect it).
    """
    df = trial_data.copy()

    # --- core signals ---
    se_strength = pd.to_numeric(df.get("SE_strength", 0), errors="coerce").fillna(0)
    df["StimPresent"] = se_strength > 0

    nlicks = pd.to_numeric(df.get("NLicks", 0), errors="coerce").fillna(0)
    df["Licked"] = nlicks > 0

    # --- compute number of stimulated elements per trial (noise already removed in builder) ---
    if "SE_StimElem" in df.columns:
        def _count_elems(x):
            if not isinstance(x, (list, tuple)):
                return 0
            if use_distinct_elems:
                return len(set([e for e in x if pd.notna(e)]))
            return len([e for e in x if pd.notna(e)])
        df["NStimElem"] = df["SE_StimElem"].apply(_count_elems)
    else:
        df["NStimElem"] = 0

    # --- filtering ---
    keep = pd.Series(True, index=df.index)

    if strength_min is not None:
        keep &= se_strength >= strength_min
    if strength_max is not None:
        keep &= se_strength <= strength_max

    if n_elem_min is not None:
        keep &= df["NStimElem"] >= n_elem_min
    if n_elem_max is not None:
        keep &= df["NStimElem"] <= n_elem_max

    df = df.loc[keep].copy()

    # --- outcome classification (4-way) ---
    df["Outcome4"] = np.select(
        [
            df["StimPresent"] & df["Licked"],
            (~df["StimPresent"]) & df["Licked"],
            df["StimPresent"] & (~df["Licked"]),
            (~df["StimPresent"]) & (~df["Licked"]),
        ],
        ["Hit", "FalseAlarm", "Miss", "CorrectRejection"],
        default="Other",
    )

    # --- sort for learning curve ---
    sort_cols = [c for c in ["SessionTime", "SessionBase", "TrialInSession"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df["TrialIndex"] = np.arange(1, len(df) + 1)

    # --- rolling fractions ---
    hit = (df["Outcome4"] == "Hit").rolling(W, min_periods=min_periods).mean()
    fa  = (df["Outcome4"] == "FalseAlarm").rolling(W, min_periods=min_periods).mean()
    miss = (df["Outcome4"] == "Miss").rolling(W, min_periods=min_periods).mean()
    cr   = (df["Outcome4"] == "CorrectRejection").rolling(W, min_periods=min_periods).mean()

    # --- plot ---
    plt.figure()
    plt.plot(df["TrialIndex"], hit, label="Hit")
    plt.plot(df["TrialIndex"], fa,  label="False Alarm")
    plt.plot(df["TrialIndex"], miss, label="Miss")
    plt.plot(df["TrialIndex"], cr,  label="Correct Rejection")
    plt.xlabel("Trial (filtered + concatenated)")
    plt.ylabel(f"Fraction (rolling window = {W})")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.title(title)
    plt.show()

    return df
