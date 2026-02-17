from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Any
import pandas as pd


# ---- Types ----
LoaderFn = Callable[[Path, str], pd.DataFrame]  # (folder, base) -> DataFrame


@dataclass(frozen=True)
class SessionMeta:
    base: str
    session_time: pd.Timestamp
    files: Dict[str, Path]  # keys: header/lick/stimuli/trials


def list_sessions(
    folder: str | Path,
    recursive: bool = False,
    require_trials: bool = True,
) -> pd.DataFrame:
    """
    List sessions by discovering BASENAME_* files.

    - Groups files by BASENAME
    - Optionally searches recursively
    - Sorts sessions by earliest modification time across the discovered files
    - Returns a metadata DataFrame (one row per session)
    """
    folder = Path(folder).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    it = folder.rglob("*.csv") if recursive else folder.glob("*.csv")

    suffixes = {
        "header": "_header.csv",
        "lick": "_lick.csv",
        "stimuli": "_stimuli.csv",
        "trials": "_trials.csv",
    }

    # Map: base -> {filetype: Path}
    by_base: Dict[str, Dict[str, Path]] = {}

    for p in it:
        name_low = p.name.lower()
        for ftype, suf in suffixes.items():
            if name_low.endswith(suf):
                base = p.name[: -len(suf)]  # preserve original case in base
                by_base.setdefault(base, {})[ftype] = p
                break

    # Optionally require trials file to consider it a session
    rows = []
    for base, files in by_base.items():
        if require_trials and ("trials" not in files):
            continue

        mtimes = [fp.stat().st_mtime for fp in files.values()]
        t0 = pd.to_datetime(min(mtimes), unit="s") if mtimes else pd.NaT

        rows.append(
            {
                "SessionBase": base,
                "SessionTime": t0,
                "HasHeader": "header" in files,
                "HasLick": "lick" in files,
                "HasStimuli": "stimuli" in files,
                "HasTrials": "trials" in files,
                "HeaderFile": str(files.get("header", "")),
                "LickFile": str(files.get("lick", "")),
                "StimuliFile": str(files.get("stimuli", "")),
                "TrialsFile": str(files.get("trials", "")),
            }
        )

    sessions_df = pd.DataFrame(rows)
    if sessions_df.empty:
        return sessions_df

    sessions_df = sessions_df.sort_values("SessionTime", kind="mergesort").reset_index(drop=True)
    return sessions_df


def load_all_sessions(
    folder: str | Path,
    loaders: Dict[str, LoaderFn],
    recursive: bool = False,
    require_trials: bool = True,
    on_error: str = "skip",  # "skip" or "raise"
) -> Dict[str, Any]:
    """
    Main orchestrator:
      1) List sessions (by BASENAME)
      2) For each session, call loaders for requested file types
      3) Return:
         - sessions: metadata DataFrame
         - data: dict[base] -> dict[filetype] -> DataFrame (or None if missing)

    loaders: dict like:
      {
        "trials": load_trials,     # function(folder_path, base) -> DataFrame
        "stimuli": load_stimuli,
        "lick": load_lick,
        "header": load_header,
      }

    Each loader should internally locate the correct file for that base and filetype.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError("on_error must be 'skip' or 'raise'")

    folder = Path(folder).expanduser().resolve()

    sessions = list_sessions(folder, recursive=recursive, require_trials=require_trials)
    data: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}

    if sessions.empty:
        return {"sessions": sessions, "data": data}

    for base in sessions["SessionBase"].tolist():
        data[base] = {}

        for ftype, loader in loaders.items():
            try:
                data[base][ftype] = loader(folder, base)
            except FileNotFoundError:
                # file missing for that session (allowed)
                data[base][ftype] = None
            except Exception as e:
                if on_error == "raise":
                    raise
                print(f"WARNING: failed loading {ftype} for {base}: {e}")
                data[base][ftype] = None

    return {"sessions": sessions, "data": data}
