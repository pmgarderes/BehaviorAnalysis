#!/usr/bin/env python3
"""
Quick behavior + video ROI analysis wrapper, shared-metadata aware.

Designed for one folder = one behavioral session.
The folder may contain several behavior segments:
    BASENAME_S1_trials.csv, BASENAME_S1_lick.csv, ...
    BASENAME_S2_trials.csv, BASENAME_S2_lick.csv, ...

Those segment bases are loaded with the existing BehaviorAnalysis loaders, then
concatenated as one folder-level session in natural segment order.

The video basename does NOT need to match the behavior basename. The video ROI
script infers the video basename from cam0/camera0 video files in the folder.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from behav_analysis.load_session import load_all_sessions
from behav_analysis.load_trials import load_trials_for_session
from behav_analysis.load_stimuli import load_stimuli_for_session
from behav_analysis.load_lick import load_lick_for_session
from behav_analysis.load_header import load_header_for_session
from behav_analysis.build_trial_table2 import build_trial_data
from behav_analysis.header_extract import add_header_fields


def natural_key(path_or_str):
    name = path_or_str.name if isinstance(path_or_str, Path) else str(path_or_str)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def ask_folder_if_missing(folder_arg: Optional[str]) -> Path:
    if folder_arg:
        return Path(folder_arg).expanduser().resolve()

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askdirectory(title="Select folder with behavior + video files")
        root.destroy()
        if selected:
            return Path(selected).resolve()
    except Exception:
        pass

    return Path(input("Data folder: ").strip()).expanduser().resolve()


def parse_segment_base(base: str):
    """
    Return (root, segment_number) for BASENAME_S1-like bases.
    If no segment suffix is found, return (base, None).
    """
    m = re.match(r"^(?P<root>.+)_S(?P<seg>\d+)$", base, flags=re.IGNORECASE)
    if m:
        return m.group("root"), int(m.group("seg"))
    return base, None


def infer_folder_session(sessions_df: pd.DataFrame) -> tuple[str, list[str], dict[str, int]]:
    """
    Treat every discovered behavior base in the folder as part of one session.
    Segment bases are sorted by S number; non-segment bases are sorted naturally.

    Returns:
        session_root: output basename for this folder-level session
        selected_bases: behavior bases to load, in concatenation order
        segment_order: base -> integer order
    """
    if sessions_df.empty:
        raise RuntimeError("No behavior sessions found. Check that *_trials.csv files exist.")

    bases = sessions_df["SessionBase"].astype(str).tolist()
    parsed = [(b, *parse_segment_base(b)) for b in bases]  # (base, root, seg)

    roots = sorted({root for _, root, _ in parsed}, key=natural_key)
    segs = [seg for _, _, seg in parsed]

    if len(roots) == 1:
        session_root = roots[0]
    else:
        # One folder should be one session. If multiple roots exist, still concatenate all
        # without prompting. Use a conservative output base.
        session_root = "FolderSession"

    def sort_key(base: str):
        root, seg = parse_segment_base(base)
        if seg is not None:
            return (natural_key(root), 0, seg)
        return (natural_key(root), 1, 0)

    selected_bases = sorted(bases, key=sort_key)
    segment_order = {b: i + 1 for i, b in enumerate(selected_bases)}

    return session_root, selected_bases, segment_order


def subset_loaded_data(data: dict, selected_bases: list[str]) -> dict:
    return {base: data[base] for base in selected_bases if base in data}


def add_folder_session_columns(trial_data: pd.DataFrame, session_root: str, segment_order: dict[str, int]) -> pd.DataFrame:
    out = trial_data.copy()

    # Preserve original SegmentBase from the existing loader's SessionBase column.
    if "SessionBase" in out.columns:
        out.insert(0, "FolderSessionBase", session_root)
        out.insert(1, "SegmentBase", out["SessionBase"].astype(str))
        out.insert(2, "SegmentOrder", out["SegmentBase"].map(segment_order).astype("Int64"))
    else:
        out.insert(0, "FolderSessionBase", session_root)
        out.insert(1, "SegmentBase", pd.NA)
        out.insert(2, "SegmentOrder", pd.NA)

    sort_cols = [c for c in ["SegmentOrder", "Trial"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return out


def wide_video_by_trial(video_by_trial: pd.DataFrame) -> pd.DataFrame:
    """
    Convert video trial summary from long camera format:
        CameraID, Trial, mean_motion, ...
    to wide format:
        Trial, cam0_mean_motion, cam1_mean_motion, ...
    so behavior trials are not duplicated when two cameras exist.
    """
    if video_by_trial.empty or "Trial" not in video_by_trial.columns:
        return video_by_trial

    if "CameraID" not in video_by_trial.columns:
        return video_by_trial

    out = None
    for cam, df_cam in video_by_trial.groupby("CameraID", dropna=False):
        cam_str = str(cam)
        df_cam = df_cam.copy()
        keep_cols = [c for c in df_cam.columns if c not in {"CameraID"}]
        df_cam = df_cam[keep_cols]
        rename = {c: f"{cam_str}_{c}" for c in df_cam.columns if c != "Trial"}
        df_cam = df_cam.rename(columns=rename)
        out = df_cam if out is None else out.merge(df_cam, on="Trial", how="outer")

    return out


def main():
    parser = argparse.ArgumentParser(description="Quick behavior + video ROI analysis for one folder-level session.")
    parser.add_argument("--folder", default=None, help="Folder containing behavior segment files and videos.")
    parser.add_argument("--radius", type=int, default=40, help="Video ROI radius in pixels.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Video ROI selection display scale.")
    parser.add_argument("--metadata-mode", choices=["auto", "camera-column", "alternating", "duplicate"], default="auto",
                        help="How to split a shared metadata CSV across cameras.")
    parser.add_argument("--metadata-file", action="append", default=None,
                        help="Explicit metadata CSV filename/path. Can be passed multiple times.")
    parser.add_argument("--camera-column", default=None, help="Explicit camera column in video metadata.")
    parser.add_argument("--frame-column", default=None, help="Explicit frame column in video metadata.")
    parser.add_argument("--trial-column", default=None, help="Explicit trial column in video metadata.")
    parser.add_argument("--time-column", default=None, help="Explicit timestamp column in video metadata.")
    parser.add_argument("--camera-offset", type=int, default=0,
                        help="Subtract this from numeric camera IDs after normalization. Use 1 if metadata uses 1/2 for cam0/cam1.")
    parser.add_argument("--skip-video", action="store_true", help="Only build the behavior table; do not run video ROI analysis.")
    parser.add_argument("--video-script", default=None, help="Path to simple_video_motion_roi_v3.py if not next to this script.")
    args = parser.parse_args()

    data_folder = ask_folder_if_missing(args.folder)
    if not data_folder.exists():
        raise SystemExit(f"Folder does not exist: {data_folder}")

    print(f"\nSelected folder:\n{data_folder}\n")

    loaders = {
        "trials": load_trials_for_session,
        "stimuli": load_stimuli_for_session,
        "lick": load_lick_for_session,
        "header": load_header_for_session,
    }

    loaded = load_all_sessions(
        data_folder,
        loaders=loaders,
        recursive=False,
        require_trials=True,
        on_error="skip",
    )

    sessions_df = loaded["sessions"]
    if sessions_df.empty:
        raise RuntimeError("No behavior session found. Check that *_trials.csv exists.")

    session_root, selected_bases, segment_order = infer_folder_session(sessions_df)
    data = subset_loaded_data(loaded["data"], selected_bases)

    print("Behavior segments to concatenate:")
    for b in selected_bases:
        print(f"  {segment_order[b]:02d}. {b}")
    print(f"\nFolder-level session/output base: {session_root}")

    trial_data = build_trial_data(data)
    trial_data = add_header_fields(trial_data)
    trial_data = add_folder_session_columns(trial_data, session_root, segment_order)

    out_behavior = data_folder / f"{session_root}_quick_trial_data.csv"
    trial_data.to_csv(out_behavior, index=False)

    print(f"\nBehavior table:")
    print(f"  {trial_data.shape[0]} trials × {trial_data.shape[1]} columns")
    print(f"  saved: {out_behavior}")

    if not args.skip_video:
        if args.video_script is not None:
            roi_script = Path(args.video_script).expanduser().resolve()
        else:
            roi_script = Path(__file__).resolve().parent / "simple_video_motion_roi_v3.py"

        if not roi_script.exists():
            print("\nVideo ROI script not found:")
            print(f"  expected: {roi_script}")
            print("Behavior table was still saved.")
        else:
            print("\nRunning video ROI motion analysis...")
            cmd = [
                sys.executable,
                str(roi_script),
                "--folder",
                str(data_folder),
                "--output-base",
                session_root,
                "--radius",
                str(args.radius),
                "--display-scale",
                str(args.display_scale),
                "--metadata-mode",
                args.metadata_mode,
                "--camera-offset",
                str(args.camera_offset),
            ]
            for meta_file in args.metadata_file or []:
                cmd.extend(["--metadata-file", str(meta_file)])
            optional_pairs = [
                ("--camera-column", args.camera_column),
                ("--frame-column", args.frame_column),
                ("--trial-column", args.trial_column),
                ("--time-column", args.time_column),
            ]
            for flag, value in optional_pairs:
                if value:
                    cmd.extend([flag, str(value)])
            subprocess.run(cmd, check=True)

    # Merge video trial summary, if present.
    video_trial_file = data_folder / f"{session_root}_video_motion_roi_by_trial.csv"

    if video_trial_file.exists():
        video_by_trial = pd.read_csv(video_trial_file)
        video_by_trial_wide = wide_video_by_trial(video_by_trial)

        if "Trial" in video_by_trial_wide.columns and "Trial" in trial_data.columns:
            # Keep Trial numeric when possible.
            trial_data["Trial"] = pd.to_numeric(trial_data["Trial"], errors="coerce").astype("Int64")
            video_by_trial_wide["Trial"] = pd.to_numeric(video_by_trial_wide["Trial"], errors="coerce").astype("Int64")

            merged = trial_data.merge(video_by_trial_wide, on="Trial", how="left")

            out_merged = data_folder / f"{session_root}_quick_trial_data_with_video.csv"
            merged.to_csv(out_merged, index=False)

            print(f"\nMerged behavior + video table:")
            print(f"  {merged.shape[0]} trials × {merged.shape[1]} columns")
            print(f"  saved: {out_merged}")
        else:
            print("\nVideo summary found, but could not merge because Trial column is missing.")
    else:
        print("\nNo video trial summary found yet.")
        if args.skip_video:
            print("This is expected because --skip-video was used.")
        else:
            print("Check that frame metadata contains a Trial column.")


if __name__ == "__main__":
    main()
