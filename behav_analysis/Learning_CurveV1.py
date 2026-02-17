import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_learning_curve_v1(trial_data, W=100, min_periods=10):
    df = trial_data.copy()
    df["StimPresent"] = pd.to_numeric(df["SE_strength"], errors="coerce").fillna(0) > 0
    df["Licked"] = pd.to_numeric(df["NLicks"], errors="coerce").fillna(0) > 0

    df["Outcome3"] = np.select(
        [
            df["StimPresent"] & df["Licked"],
            (~df["StimPresent"]) & df["Licked"],
            (~df["Licked"]),
        ],
        ["Hit", "FalseAlarm", "NoLick"],
        default="Other"
    )

    sort_cols = [c for c in ["SessionTime", "SessionBase", "TrialInSession"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["TrialIndex"] = np.arange(1, len(df) + 1)

    hit = (df["Outcome3"] == "Hit").rolling(W, min_periods=min_periods).mean()
    fa  = (df["Outcome3"] == "FalseAlarm").rolling(W, min_periods=min_periods).mean()
    nol = (df["Outcome3"] == "NoLick").rolling(W, min_periods=min_periods).mean()

    plt.figure()
    plt.plot(df["TrialIndex"], hit, label="Hit")
    plt.plot(df["TrialIndex"], fa,  label="False Alarm")
    plt.plot(df["TrialIndex"], nol, label="NoLick (Miss or CR)")
    plt.xlabel("Trial (concatenated)")
    plt.ylabel(f"Fraction (rolling window = {W})")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.title("Go/No-Go learning curve")
    plt.show()
