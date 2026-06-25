#!/usr/bin/env python3
"""
Simple video ROI motion-energy extractor, shared-metadata aware version.

Use case:
- One folder = one behavioral session.
- The folder may contain behavior CSVs, videos, and one shared camera metadata CSV.
- Video basename does NOT need to match the behavior basename.
- Multiple video segments are allowed and are processed in natural order.
- The script infers the video basename from cam0/camera0 AVI/MP4 files.
- It asks you to click the whisker ROI once per camera.
- It computes frame-to-frame motion energy in a square ROI.
- It attaches frame metadata even if a single metadata CSV contains intermingled rows
  from cam0 and cam1.
- It saves frame-level and trial-level CSV files in the same folder.

Typical run:
    python simple_video_motion_roi_v3.py --folder "C:\\Data\\Session01" --radius 40

If metadata has no explicit camera column but rows alternate cam0, cam1, cam0, cam1:
    python simple_video_motion_roi_v3.py --folder "C:\\Data\\Session01" --metadata-mode alternating

Useful options:
    --video-base          manually specify video basename if needed
    --output-base         basename used for output CSV files, e.g. behavior session root
    --metadata-file       explicitly choose one metadata CSV; can be passed multiple times
    --camera-column       explicitly name the camera column in metadata
    --frame-column        explicitly name the frame column in metadata
    --trial-column        explicitly name the trial column in metadata
    --time-column         explicitly name the time column in metadata
    --metadata-mode       auto, camera-column, alternating, or duplicate

Expected frame metadata columns, if available:
- Trial / trial / trial_number / TrialNumber / TrialInSession / trial_id
- Frame / frame / frame_idx / frame_number / FrameNumber
- timestamp_s / FrameTime_s / time_s / camera_time_s / host_time_s / Time
- CameraID / camera / Camera / cam / camera_index / device / source

If no Trial column is found after metadata attachment, frame-level motion is still saved,
but no trial summary is produced.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required. Install with:\n\n"
        "    python -m pip install opencv-python pandas numpy\n"
    ) from exc


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
BEHAV_SUFFIXES = ("_trials", "_stimuli", "_lick", "_header")
OUTPUT_MARKERS = ("video_motion_roi", "motion_roi", "quick_trial_data", "by_stim_condition")

FRAME_COL_CANDIDATES = [
    "Frame", "frame", "frame_idx", "FrameIndex", "frame_index",
    "FrameNumber", "frame_number", "nFrame", "frame_id", "image_number",
    "ImageNumber", "ChunkFrameID", "frameCounter", "FrameCounter",
]
TRIAL_COL_CANDIDATES = [
    "Trial", "trial", "trial_number", "TrialNumber", "TrialInSession",
    "trial_id", "trial_idx", "trial_index", "TrialID", "TrialIdx",
]
TIME_COL_CANDIDATES = [
    "FrameTime_s", "frame_time_s", "timestamp_s", "Timestamp_s",
    "camera_time_s", "CameraTime_s", "host_time_s", "HostTime_s",
    "time_s", "Time_s", "Time", "time", "timestamp", "Timestamp",
]
CAMERA_COL_CANDIDATES = [
    "CameraID", "camera_id", "CameraId", "camera", "Camera",
    "cam", "Cam", "CamID", "cam_id", "CameraIndex", "camera_index",
    "camera_number", "CameraNumber", "Device", "device", "Source", "source",
]
VIDEO_FILE_COL_CANDIDATES = [
    "video_file", "VideoFile", "filename", "Filename", "file", "File",
    "avi_file", "AVIFile", "MovieFile", "movie_file",
]
SEGMENT_COL_CANDIDATES = [
    "Segment", "segment", "SegmentID", "segment_id", "SegmentNumber",
    "segment_number", "S", "segmentIndex", "SegmentIndex",
]


# -------------------------
# File-name helpers
# -------------------------

def natural_key(path_or_str):
    """Sort strings/paths with numbers in human order: S2 before S10."""
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
        selected = filedialog.askdirectory(title="Select folder containing behavior + video files")
        root.destroy()
        if selected:
            return Path(selected).resolve()
    except Exception:
        pass

    return Path(input("Session folder: ").strip()).expanduser().resolve()


def is_behavior_csv(path: Path) -> bool:
    stem_low = path.stem.lower()
    return any(stem_low.endswith(suf) for suf in BEHAV_SUFFIXES)


def is_output_csv(path: Path) -> bool:
    low = path.stem.lower()
    return any(marker in low for marker in OUTPUT_MARKERS)


def guess_camera_id(video_path: Path) -> str:
    name = video_path.stem
    m = re.search(r"(?:cam|camera)[_-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"cam{m.group(1)}"
    return "cam0"


def normalize_camera_value(x) -> Optional[str]:
    """Convert metadata camera values into cam0/cam1/... when possible."""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None

    m = re.search(r"(?:cam|camera)[_ -]?(\d+)", s, flags=re.IGNORECASE)
    if m:
        return f"cam{int(m.group(1))}"

    # Numeric camera values: assume 0/1 are already zero-based; 1/2 may be one-based.
    # We keep 0 -> cam0, 1 -> cam1. If your metadata uses 1/2 for cam0/cam1,
    # pass --camera-offset 1.
    try:
        val = int(float(s))
        return f"cam{val}"
    except Exception:
        return s.lower()


def guess_segment_id(path: Path) -> Optional[int]:
    """Return segment number from patterns like _S1, -S2, _seg3, segment4."""
    stem = path.stem
    patterns = [
        r"(?:^|[_-])S(\d+)(?:[_-]|$)",
        r"(?:^|[_-])seg(?:ment)?[_-]?(\d+)(?:[_-]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, stem, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def video_root_from_cam0_stem(stem: str) -> str:
    """
    Infer common video basename from a cam0/camera0 filename stem.

    Examples:
        MyVideo_cam0          -> MyVideo
        MyVideo_S1_cam0       -> MyVideo
        MyVideo_cam0_S1       -> MyVideo
        MyVideo_segment2_cam0 -> MyVideo
    """
    s = stem
    s = re.sub(r"(?:^|[_-])(?:cam|camera)[_-]?0(?:[_-]|$)", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:[_-])S\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:[_-])seg(?:ment)?[_-]?\d+$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:[_-])S\d+(?:[_-])?$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"(?:[_-])seg(?:ment)?[_-]?\d+(?:[_-])?$", "", s, flags=re.IGNORECASE)
    return s.strip("_-")


def infer_video_base_from_cam0(folder: Path) -> str:
    cam0_videos = []
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        if re.search(r"(?:cam|camera)[_-]?0", p.stem, flags=re.IGNORECASE):
            cam0_videos.append(p)

    if not cam0_videos:
        raise SystemExit(
            "No cam0/camera0 video file found. Either rename the video to include cam0 "
            "or pass --video-base manually."
        )

    groups: dict[str, list[Path]] = {}
    for p in cam0_videos:
        root = video_root_from_cam0_stem(p.stem)
        groups.setdefault(root, []).append(p)

    if len(groups) > 1:
        msg = ["Multiple possible video basenames found from cam0 files:"]
        for root, vids in sorted(groups.items(), key=lambda kv: natural_key(kv[0])):
            msg.append(f"  {root}: {', '.join(v.name for v in sorted(vids, key=natural_key))}")
        msg.append("Pass --video-base explicitly to avoid ambiguity.")
        raise SystemExit("\n".join(msg))

    video_base = next(iter(groups.keys()))
    print(f"Detected video basename from cam0 file: {video_base}")
    return video_base


def find_videos(folder: Path, video_base: str) -> list[Path]:
    videos = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and p.stem.startswith(video_base)
    ]
    return sorted(videos, key=lambda p: (guess_camera_id(p), guess_segment_id(p) or 0, natural_key(p)))


def find_metadata_csvs(folder: Path, video_base: str, explicit_files: Optional[list[str]] = None) -> list[Path]:
    if explicit_files:
        return [Path(x).expanduser().resolve() if Path(x).is_absolute() else (folder / x) for x in explicit_files]

    csvs = []
    for p in folder.glob("*.csv"):
        if not p.is_file():
            continue
        if is_behavior_csv(p) or is_output_csv(p):
            continue
        low = p.stem.lower()
        # In your current acquisition, the shared frame metadata is named like the video base:
        # Day5bis_Het3_behavior.csv, while analog gets Day5bis_Het3_behavior_Analog.csv.
        # Keep the base file and camera-related files; exclude obvious analog files.
        if "analog" in low:
            continue
        if p.stem.startswith(video_base) or "cam" in low or "camera" in low:
            csvs.append(p)
    return sorted(csvs, key=natural_key)


def find_first_existing_col(df: pd.DataFrame, candidates: list[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit in df.columns:
            return explicit
        lower_map = {str(c).lower(): c for c in df.columns}
        if explicit.lower() in lower_map:
            return lower_map[explicit.lower()]
        raise SystemExit(f"Requested column {explicit!r} not found. Available columns: {list(df.columns)}")

    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# -------------------------
# Metadata handling
# -------------------------

@dataclass
class MetadataConfig:
    mode: str
    camera_column: Optional[str] = None
    frame_column: Optional[str] = None
    trial_column: Optional[str] = None
    time_column: Optional[str] = None
    video_file_column: Optional[str] = None
    segment_column: Optional[str] = None
    camera_offset: int = 0


def read_and_normalize_metadata(
    metadata_csvs: list[Path],
    videos: list[Path],
    cfg: MetadataConfig,
) -> pd.DataFrame:
    """Read candidate metadata CSVs and normalize common columns."""
    if not metadata_csvs:
        return pd.DataFrame()

    parts = []
    for i, path in enumerate(metadata_csvs):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  warning: could not read metadata {path.name}: {exc}")
            continue
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df.insert(0, "metadata_file", path.name)
        df.insert(1, "metadata_file_order", i)
        df.insert(2, "metadata_row_in_file", np.arange(len(df), dtype=int))
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    meta = pd.concat(parts, ignore_index=True, sort=False)
    meta.insert(0, "metadata_global_row", np.arange(len(meta), dtype=int))

    frame_col = find_first_existing_col(meta, FRAME_COL_CANDIDATES, cfg.frame_column)
    trial_col = find_first_existing_col(meta, TRIAL_COL_CANDIDATES, cfg.trial_column)
    time_col = find_first_existing_col(meta, TIME_COL_CANDIDATES, cfg.time_column)
    cam_col = find_first_existing_col(meta, CAMERA_COL_CANDIDATES, cfg.camera_column)
    video_col = find_first_existing_col(meta, VIDEO_FILE_COL_CANDIDATES, cfg.video_file_column)
    seg_col = find_first_existing_col(meta, SEGMENT_COL_CANDIDATES, cfg.segment_column)

    if frame_col is not None:
        frame_values = pd.to_numeric(meta[frame_col], errors="coerce")
        # Do not force 1-based conversion globally; final per-video assignment re-normalizes.
        meta["metadata_frame_value"] = frame_values.astype("Int64")
    else:
        meta["metadata_frame_value"] = pd.NA

    if trial_col is not None:
        meta["Trial"] = pd.to_numeric(meta[trial_col], errors="coerce").astype("Int64")

    if time_col is not None:
        meta["FrameTime_s"] = pd.to_numeric(meta[time_col], errors="coerce")

    if video_col is not None:
        meta["metadata_video_file"] = meta[video_col].astype(str)

    if seg_col is not None:
        meta["metadata_segment_id"] = pd.to_numeric(meta[seg_col], errors="coerce").astype("Int64")

    camera_ids = sorted({guess_camera_id(v) for v in videos}, key=natural_key)

    if cam_col is not None and cfg.mode in {"auto", "camera-column"}:
        meta["metadata_camera_id"] = meta[cam_col].apply(normalize_camera_value)
        if cfg.camera_offset != 0:
            def offset_cam(x):
                if x is None or pd.isna(x):
                    return None
                m = re.match(r"cam(\d+)$", str(x))
                if not m:
                    return x
                return f"cam{int(m.group(1)) - cfg.camera_offset}"
            meta["metadata_camera_id"] = meta["metadata_camera_id"].apply(offset_cam)
        print(f"  metadata camera column: {cam_col}")
    elif cfg.mode == "camera-column":
        raise SystemExit(
            "--metadata-mode camera-column was requested, but no camera column was found. "
            "Pass --camera-column COLUMN_NAME or use --metadata-mode alternating."
        )
    elif cfg.mode == "alternating" or (cfg.mode == "auto" and len(camera_ids) > 1):
        # Fallback for intermingled metadata where rows are cam0, cam1, cam0, cam1...
        # This is only used when no explicit camera column exists.
        if cam_col is None:
            ncam = len(camera_ids)
            meta["metadata_camera_id"] = [camera_ids[i % ncam] for i in range(len(meta))]
            print(
                "  no metadata camera column found; assigning rows by alternating camera order: "
                + ", ".join(camera_ids)
            )
        else:
            meta["metadata_camera_id"] = meta[cam_col].apply(normalize_camera_value)
    elif cfg.mode == "duplicate":
        # Same metadata rows are attached to every camera. Useful only if metadata lacks camera rows
        # and contains one row per simultaneous frame, not one row per camera frame.
        meta["metadata_camera_id"] = pd.NA
        print("  metadata mode duplicate: same metadata will be reused for each camera")
    else:
        meta["metadata_camera_id"] = pd.NA

    print("  metadata normalized columns:")
    print(f"    frame:  {frame_col}")
    print(f"    trial:  {trial_col}")
    print(f"    time:   {time_col}")
    print(f"    camera: {cam_col}")
    print(f"    video:  {video_col}")
    print(f"    segment:{seg_col}")

    return meta


def metadata_rows_for_video(
    meta: pd.DataFrame,
    video: Path,
    camera_id: str,
    n_frames: int,
    consumed_by_camera: dict[str, int],
    cfg: MetadataConfig,
) -> Optional[pd.DataFrame]:
    """
    Select metadata rows for this video.

    Priority:
    1. rows whose metadata_video_file matches the video file/stem
    2. rows matching camera + segment
    3. sequential slice within this camera's metadata rows
    4. duplicate mode: first n rows, reused for each camera
    """
    if meta is None or meta.empty:
        return None

    seg = guess_segment_id(video)

    # 1. Exact video/file match if available.
    if "metadata_video_file" in meta.columns:
        vf = meta["metadata_video_file"].astype(str)
        mask_video = vf.str.contains(re.escape(video.name), case=False, na=False) | vf.str.contains(re.escape(video.stem), case=False, na=False)
        by_video = meta.loc[mask_video].copy()
        if not by_video.empty:
            print(f"  metadata matched by video filename: {len(by_video)} rows")
            return finalize_metadata_for_video(by_video, n_frames)

    # 2/3. Select rows by camera, then optionally by segment.
    if cfg.mode == "duplicate":
        rows = meta.sort_values(["metadata_file_order", "metadata_row_in_file"], kind="mergesort").head(n_frames).copy()
        print(f"  metadata duplicated for this camera: {len(rows)} rows")
        return finalize_metadata_for_video(rows, n_frames)

    if "metadata_camera_id" in meta.columns and meta["metadata_camera_id"].notna().any():
        cam_rows = meta.loc[meta["metadata_camera_id"].astype(str).str.lower() == camera_id.lower()].copy()
    else:
        cam_rows = meta.copy()

    if cam_rows.empty:
        print(f"  warning: no metadata rows found for {camera_id}")
        return None

    cam_rows = cam_rows.sort_values(["metadata_file_order", "metadata_row_in_file"], kind="mergesort")

    if seg is not None and "metadata_segment_id" in cam_rows.columns:
        seg_rows = cam_rows.loc[pd.to_numeric(cam_rows["metadata_segment_id"], errors="coerce") == seg].copy()
        if len(seg_rows) >= max(1, int(0.8 * n_frames)):
            print(f"  metadata matched by camera + segment S{seg}: {len(seg_rows)} rows")
            return finalize_metadata_for_video(seg_rows, n_frames)

    # Sequential slice for this camera. This handles multiple video segments when the metadata file
    # is one continuous table ordered by acquisition time.
    start = consumed_by_camera.get(camera_id, 0)
    stop = start + n_frames
    rows = cam_rows.iloc[start:stop].copy()
    consumed_by_camera[camera_id] = stop

    if rows.empty:
        print(f"  warning: metadata sequential slice empty for {camera_id} rows {start}:{stop}")
        return None

    print(f"  metadata matched by sequential {camera_id} slice: rows {start}:{stop} ({len(rows)} rows)")
    if len(rows) != n_frames:
        print(f"  warning: video has {n_frames} frames but metadata slice has {len(rows)} rows")

    return finalize_metadata_for_video(rows, n_frames)


def finalize_metadata_for_video(rows: pd.DataFrame, n_frames: int) -> pd.DataFrame:
    """Create a clean metadata table with local frame index for merging to motion."""
    out = rows.copy().reset_index(drop=True)

    # Prefer explicit frame values only if they look like local frame numbers.
    frame_idx = None
    if "metadata_frame_value" in out.columns and out["metadata_frame_value"].notna().any():
        fv = pd.to_numeric(out["metadata_frame_value"], errors="coerce")
        mn = fv.min(skipna=True)
        mx = fv.max(skipna=True)
        if pd.notna(mn) and pd.notna(mx):
            if mn == 0 and mx <= n_frames - 1:
                frame_idx = fv
            elif mn == 1 and mx <= n_frames:
                frame_idx = fv - 1

    if frame_idx is None:
        frame_idx = pd.Series(np.arange(len(out)), index=out.index)

    out["frame_idx_in_segment"] = pd.to_numeric(frame_idx, errors="coerce").astype("Int64")

    # Keep at most one metadata row per local frame index.
    out = out.dropna(subset=["frame_idx_in_segment"]).drop_duplicates("frame_idx_in_segment", keep="first")

    # Make Trial numeric if present.
    if "Trial" in out.columns:
        out["Trial"] = pd.to_numeric(out["Trial"], errors="coerce").astype("Int64")

    return out


# -------------------------
# ROI selection + motion
# -------------------------

def read_first_frame(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read first frame from: {video_path}")
    return frame


def select_roi_center(frame: np.ndarray, radius: int, display_scale: float) -> dict:
    """Ask user to click the ROI center. Returns square ROI coordinates in original pixels."""
    if display_scale <= 0:
        display_scale = 1.0

    shown = frame.copy()
    if display_scale != 1.0:
        shown = cv2.resize(
            shown,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA,
        )

    click_xy = {"x": None, "y": None}
    window = "Click whisker ROI center, then press ENTER. Press r to reset."

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_xy["x"] = int(round(x / display_scale))
            click_xy["y"] = int(round(y / display_scale))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        canvas = shown.copy()
        if click_xy["x"] is not None:
            sx = int(round(click_xy["x"] * display_scale))
            sy = int(round(click_xy["y"] * display_scale))
            sr = int(round(radius * display_scale))
            cv2.rectangle(canvas, (sx - sr, sy - sr), (sx + sr, sy + sr), (0, 255, 0), 2)
            cv2.circle(canvas, (sx, sy), 3, (0, 255, 0), -1)

        cv2.imshow(window, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, 10):  # Enter
            if click_xy["x"] is not None:
                break
        elif key == ord("r"):
            click_xy = {"x": None, "y": None}
        elif key == 27:  # Escape
            cv2.destroyWindow(window)
            raise SystemExit("ROI selection cancelled.")

    cv2.destroyWindow(window)

    h, w = frame.shape[:2]
    x = int(click_xy["x"])
    y = int(click_xy["y"])
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    return {
        "x_center": x,
        "y_center": y,
        "radius_px": int(radius),
        "x1": int(x1),
        "x2": int(x2),
        "y1": int(y1),
        "y2": int(y2),
        "width_px": int(x2 - x1),
        "height_px": int(y2 - y1),
    }


def compute_motion_for_video(
    video_path: Path,
    camera_id: str,
    roi: dict,
    diff_threshold: float,
    global_start_frame: int,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    segment_id = guess_segment_id(video_path)

    x1, x2, y1, y2 = roi["x1"], roi["x2"], roi["y1"], roi["y2"]
    roi_pixels = max(1, (x2 - x1) * (y2 - y1))

    rows = []
    prev_roi = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        roi_img = gray[y1:y2, x1:x2]

        if prev_roi is None:
            mean_absdiff = np.nan
            sum_absdiff = np.nan
            frac_above_threshold = np.nan
        else:
            diff = np.abs(roi_img - prev_roi)
            mean_absdiff = float(np.mean(diff))
            sum_absdiff = float(np.sum(diff))
            frac_above_threshold = float(np.mean(diff > diff_threshold))

        rows.append({
            "CameraID": camera_id,
            "SegmentID_from_video_name": segment_id,
            "video_file": video_path.name,
            "frame_idx_in_segment": frame_idx,
            "global_frame_idx_per_camera": global_start_frame + frame_idx,
            "video_fps_reported": fps,
            "video_n_frames_reported": n_reported,
            "roi_x_center": roi["x_center"],
            "roi_y_center": roi["y_center"],
            "roi_radius_px": roi["radius_px"],
            "roi_x1": x1,
            "roi_x2": x2,
            "roi_y1": y1,
            "roi_y2": y2,
            "roi_pixels": roi_pixels,
            "motion_energy_mean_absdiff": mean_absdiff,
            "motion_energy_sum_absdiff": sum_absdiff,
            "motion_frac_pixels_above_threshold": frac_above_threshold,
        })

        prev_roi = roi_img.copy()
        frame_idx += 1

    cap.release()
    return pd.DataFrame(rows)


def attach_metadata(motion: pd.DataFrame, meta: Optional[pd.DataFrame]) -> pd.DataFrame:
    if meta is None or meta.empty:
        return motion

    # Avoid overwriting core motion columns except normalized frame index.
    drop_cols = [c for c in meta.columns if c in motion.columns and c != "frame_idx_in_segment"]
    meta2 = meta.drop(columns=drop_cols)
    return motion.merge(meta2, on="frame_idx_in_segment", how="left")


def summarize_by_trial(frame_table: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "Trial" not in frame_table.columns:
        return None

    valid = frame_table.dropna(subset=["Trial"]).copy()
    if valid.empty:
        return None

    valid["Trial"] = pd.to_numeric(valid["Trial"], errors="coerce").astype("Int64")

    group_cols = ["CameraID", "Trial"]
    agg_kwargs = dict(
        n_video_frames=("motion_energy_mean_absdiff", "size"),
        mean_motion=("motion_energy_mean_absdiff", "mean"),
        median_motion=("motion_energy_mean_absdiff", "median"),
        max_motion=("motion_energy_mean_absdiff", "max"),
        mean_motion_frac_above_threshold=("motion_frac_pixels_above_threshold", "mean"),
        first_frame=("global_frame_idx_per_camera", "min"),
        last_frame=("global_frame_idx_per_camera", "max"),
    )

    if "FrameTime_s" in valid.columns:
        agg_kwargs.update(
            first_time_s=("FrameTime_s", "min"),
            last_time_s=("FrameTime_s", "max"),
        )

    return valid.groupby(group_cols, dropna=False).agg(**agg_kwargs).reset_index()


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute simple ROI motion energy from video files.")
    parser.add_argument("--folder", default=None, help="Folder containing behavior files, videos, and metadata.")
    parser.add_argument("--video-base", default=None, help="Video basename. If omitted, inferred from cam0/camera0 video.")
    parser.add_argument("--output-base", default=None, help="Output basename. Default: video basename.")
    parser.add_argument("--base", default=None, help="Alias for --video-base, kept for older calls.")
    parser.add_argument("--radius", type=int, default=30, help="Half-size of square ROI around clicked point, in pixels.")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale factor for ROI selection display only.")
    parser.add_argument("--diff-threshold", type=float, default=10.0, help="Pixel abs-difference threshold for fraction-above-threshold metric.")
    parser.add_argument("--metadata-file", action="append", default=None, help="Explicit metadata CSV filename/path. Can be passed multiple times.")
    parser.add_argument("--metadata-mode", choices=["auto", "camera-column", "alternating", "duplicate"], default="auto",
                        help="How to split shared metadata rows across cameras.")
    parser.add_argument("--camera-column", default=None, help="Explicit camera column in metadata.")
    parser.add_argument("--frame-column", default=None, help="Explicit frame column in metadata.")
    parser.add_argument("--trial-column", default=None, help="Explicit trial column in metadata.")
    parser.add_argument("--time-column", default=None, help="Explicit timestamp column in metadata.")
    parser.add_argument("--video-file-column", default=None, help="Explicit video filename column in metadata.")
    parser.add_argument("--segment-column", default=None, help="Explicit segment column in metadata.")
    parser.add_argument("--camera-offset", type=int, default=0,
                        help="Subtract this from numeric camera IDs after normalization. Use 1 if metadata has cameras 1/2 but videos are cam0/cam1.")
    args = parser.parse_args()

    folder = ask_folder_if_missing(args.folder)
    if not folder.exists():
        raise SystemExit(f"Folder does not exist: {folder}")

    video_base = args.video_base or args.base or infer_video_base_from_cam0(folder)
    if not video_base:
        raise SystemExit("No video basename provided or inferred.")

    output_base = args.output_base or video_base

    videos = find_videos(folder, video_base)
    if not videos:
        raise SystemExit(f"No video files starting with {video_base!r} were found in {folder}")

    metadata_csvs = find_metadata_csvs(folder, video_base, args.metadata_file)

    print("\nVideos to process:")
    for v in videos:
        print(f"  - {v.name}")

    if metadata_csvs:
        print("\nCandidate frame metadata CSVs:")
        for m in metadata_csvs:
            print(f"  - {m.name}")
    else:
        print("\nNo frame metadata CSVs found. Motion will be saved without Trial/Time columns.")

    cfg = MetadataConfig(
        mode=args.metadata_mode,
        camera_column=args.camera_column,
        frame_column=args.frame_column,
        trial_column=args.trial_column,
        time_column=args.time_column,
        video_file_column=args.video_file_column,
        segment_column=args.segment_column,
        camera_offset=args.camera_offset,
    )
    metadata = read_and_normalize_metadata(metadata_csvs, videos, cfg)

    # One ROI per detected camera.
    camera_to_videos: dict[str, list[Path]] = {}
    for v in videos:
        camera_to_videos.setdefault(guess_camera_id(v), []).append(v)

    rois = {}
    for camera_id, vids in sorted(camera_to_videos.items(), key=lambda kv: natural_key(kv[0])):
        vids = sorted(vids, key=lambda p: (guess_segment_id(p) or 0, natural_key(p)))
        print(f"\nSelect ROI for {camera_id} using first frame of {vids[0].name}")
        first_frame = read_first_frame(vids[0])
        rois[camera_id] = select_roi_center(first_frame, args.radius, args.display_scale)
        print(f"Selected ROI for {camera_id}: {rois[camera_id]}")

    all_motion = []
    global_start_frame_by_camera = {cam: 0 for cam in camera_to_videos}
    consumed_metadata_by_camera = {cam: 0 for cam in camera_to_videos}

    for video in videos:
        camera_id = guess_camera_id(video)
        roi = rois[camera_id]
        print(f"\nProcessing {video.name} ...")

        motion = compute_motion_for_video(
            video_path=video,
            camera_id=camera_id,
            roi=roi,
            diff_threshold=args.diff_threshold,
            global_start_frame=global_start_frame_by_camera[camera_id],
        )
        global_start_frame_by_camera[camera_id] += len(motion)

        meta_for_video = metadata_rows_for_video(
            meta=metadata,
            video=video,
            camera_id=camera_id,
            n_frames=len(motion),
            consumed_by_camera=consumed_metadata_by_camera,
            cfg=cfg,
        )
        if meta_for_video is not None:
            motion = attach_metadata(motion, meta_for_video)
        else:
            print("  no metadata attached to this video")

        all_motion.append(motion)

    frame_table = pd.concat(all_motion, ignore_index=True)

    # Useful final cleanup.
    if "Trial" in frame_table.columns:
        frame_table["Trial"] = pd.to_numeric(frame_table["Trial"], errors="coerce").astype("Int64")

    frame_out = folder / f"{output_base}_video_motion_roi.csv"
    trial_out = folder / f"{output_base}_video_motion_roi_by_trial.csv"
    roi_out = folder / f"{output_base}_video_motion_roi_definition.json"

    frame_table.to_csv(frame_out, index=False)

    trial_table = summarize_by_trial(frame_table)
    if trial_table is not None:
        trial_table.to_csv(trial_out, index=False)

    with open(roi_out, "w", encoding="utf-8") as f:
        json.dump({
            "video_base": video_base,
            "output_base": output_base,
            "folder": str(folder),
            "radius_px": args.radius,
            "diff_threshold": args.diff_threshold,
            "metadata_mode": args.metadata_mode,
            "rois_by_camera": rois,
            "videos": [v.name for v in videos],
            "metadata_csvs": [m.name for m in metadata_csvs],
        }, f, indent=2)

    print("\nSaved:")
    print(f"  {frame_out}")
    if trial_table is not None:
        print(f"  {trial_out}")
    else:
        print("  no trial summary saved because no Trial column was found after metadata attachment")
    print(f"  {roi_out}")


if __name__ == "__main__":
    main()
