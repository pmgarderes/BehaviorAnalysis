#!/usr/bin/env python3
"""
Plot trial-averaged video motion by stimulus/condition columns.

Input is usually:
    BASENAME_quick_trial_data_with_video.csv

Typical use:
    python plot_video_motion_by_condition.py --folder "C:\\Data\\Session01"

Specify condition columns explicitly:
    python plot_video_motion_by_condition.py --input "C:\\Data\\Session01\\BASENAME_quick_trial_data_with_video.csv" --condition-cols StimFreq StimAmp

Outputs:
    INPUTSTEM_video_motion_by_condition.csv
    INPUTSTEM_video_motion_by_condition.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ask_file_if_missing(input_arg: Optional[str], folder_arg: Optional[str]) -> Path:
    if input_arg:
        return Path(input_arg).expanduser().resolve()

    folder = Path(folder_arg).expanduser().resolve() if folder_arg else None
    if folder is None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askopenfilename(
                title="Select *_quick_trial_data_with_video.csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            root.destroy()
            if selected:
                return Path(selected).resolve()
        except Exception:
            pass
        return Path(input("Path to merged behavior+video CSV: ").strip()).expanduser().resolve()

    candidates = sorted(folder.glob("*_quick_trial_data_with_video.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(f"No *_quick_trial_data_with_video.csv file found in {folder}")

    print("Multiple merged CSV files found:")
    for i, p in enumerate(candidates, start=1):
        print(f"  {i}. {p.name}")
    idx = int(input("Choose file number: ")) - 1
    return candidates[idx]


def candidate_condition_columns(df: pd.DataFrame) -> list[str]:
    keywords = [
        "stim", "freq", "frequency", "amp", "amplitude", "condition",
        "whisk", "side", "trialtype", "trial_type", "go", "nogo", "reward",
        "difficulty", "contrast", "intensity",
    ]
    exclude = ["time", "frame", "lick", "motion", "video", "roi", "camera"]
    out = []
    for c in df.columns:
        low = c.lower()
        if any(k in low for k in keywords) and not any(e in low for e in exclude):
            nunique = df[c].nunique(dropna=True)
            if 1 < nunique <= max(40, len(df) // 3):
                out.append(c)
    return out


def candidate_motion_columns(df: pd.DataFrame) -> list[str]:
    preferred = [c for c in df.columns if c.endswith("_mean_motion")]
    if preferred:
        return preferred
    return [c for c in df.columns if "mean_motion" in c.lower()]


def parse_cols_from_prompt(prompt: str, allowed: list[str]) -> list[str]:
    txt = input(prompt).strip()
    if not txt:
        return []
    raw = [x.strip() for x in txt.replace(",", " ").split() if x.strip()]
    cols = []
    for item in raw:
        if item.isdigit():
            idx = int(item) - 1
            if idx < 0 or idx >= len(allowed):
                raise SystemExit(f"Column index out of range: {item}")
            cols.append(allowed[idx])
        else:
            if item not in allowed:
                raise SystemExit(f"Column not found: {item}")
            cols.append(item)
    return cols


def choose_columns(df: pd.DataFrame, condition_cols_arg: Optional[list[str]], motion_cols_arg: Optional[list[str]]):
    if condition_cols_arg:
        condition_cols = condition_cols_arg
    else:
        cand = candidate_condition_columns(df)
        if not cand:
            print("No obvious stimulus/condition columns detected. Available columns:")
            cand = list(df.columns)
        else:
            print("Candidate condition columns:")
        for i, c in enumerate(cand, start=1):
            print(f"  {i}. {c}")
        condition_cols = parse_cols_from_prompt(
            "Type condition column name(s) or number(s), separated by spaces/commas: ",
            cand,
        )

    if not condition_cols:
        raise SystemExit("No condition columns selected.")

    if motion_cols_arg:
        motion_cols = motion_cols_arg
    else:
        cand_motion = candidate_motion_columns(df)
        if not cand_motion:
            raise SystemExit("No mean motion columns found, expected e.g. cam0_mean_motion.")
        print("\nMotion columns:")
        for i, c in enumerate(cand_motion, start=1):
            print(f"  {i}. {c}")
        selected = input("Use all motion columns? [y/n]: ").strip().lower()
        if selected in {"", "y", "yes"}:
            motion_cols = cand_motion
        else:
            motion_cols = parse_cols_from_prompt(
                "Type motion column name(s) or number(s), separated by spaces/commas: ",
                cand_motion,
            )

    for c in condition_cols + motion_cols:
        if c not in df.columns:
            raise SystemExit(f"Column not found: {c}")

    return condition_cols, motion_cols


def summarize(df: pd.DataFrame, condition_cols: list[str], motion_cols: list[str]) -> pd.DataFrame:
    work = df.copy()
    for c in motion_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    rows = []
    grouped = work.groupby(condition_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {c: v for c, v in zip(condition_cols, keys)}
        for m in motion_cols:
            vals = g[m].dropna()
            row = base.copy()
            row.update({
                "motion_column": m,
                "n_trials": int(vals.size),
                "mean_motion": float(vals.mean()) if vals.size else np.nan,
                "median_motion": float(vals.median()) if vals.size else np.nan,
                "sem_motion": float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else np.nan,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def condition_label(row: pd.Series, condition_cols: list[str]) -> str:
    parts = []
    for c in condition_cols:
        val = row[c]
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        parts.append(f"{c}={val}")
    return "\n".join(parts)


def make_plot(summary: pd.DataFrame, condition_cols: list[str], output_png: Path):
    motion_cols = list(summary["motion_column"].dropna().unique())
    for motion_col in motion_cols:
        sub = summary.loc[summary["motion_column"] == motion_col].copy()
        sub = sub.sort_values(condition_cols, kind="mergesort")
        labels = [condition_label(row, condition_cols) for _, row in sub.iterrows()]
        x = np.arange(len(sub))

        plt.figure(figsize=(max(6, 0.7 * len(sub)), 4.5))
        plt.errorbar(x, sub["mean_motion"], yerr=sub["sem_motion"], fmt="o", capsize=3)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Mean ROI motion energy / trial")
        plt.title(motion_col)
        plt.tight_layout()

        if len(motion_cols) == 1:
            this_png = output_png
        else:
            this_png = output_png.with_name(output_png.stem + f"_{motion_col}" + output_png.suffix)
        plt.savefig(this_png, dpi=200)
        plt.close()
        print(f"Saved plot: {this_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot video ROI motion by stimulus condition.")
    parser.add_argument("--input", default=None, help="Merged behavior+video CSV file.")
    parser.add_argument("--folder", default=None, help="Folder containing *_quick_trial_data_with_video.csv.")
    parser.add_argument("--condition-cols", nargs="+", default=None, help="Condition/stimulus columns to group by.")
    parser.add_argument("--motion-cols", nargs="+", default=None, help="Motion columns to plot. Default: *_mean_motion.")
    args = parser.parse_args()

    input_csv = ask_file_if_missing(args.input, args.folder)
    df = pd.read_csv(input_csv)

    condition_cols, motion_cols = choose_columns(df, args.condition_cols, args.motion_cols)
    print("\nGrouping by:", condition_cols)
    print("Motion columns:", motion_cols)

    summary = summarize(df, condition_cols, motion_cols)
    out_csv = input_csv.with_name(input_csv.stem + "_video_motion_by_condition.csv")
    out_png = input_csv.with_name(input_csv.stem + "_video_motion_by_condition.png")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    make_plot(summary, condition_cols, out_png)


if __name__ == "__main__":
    main()
