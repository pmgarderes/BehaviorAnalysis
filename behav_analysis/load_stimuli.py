from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_stimuli_for_session(folder: str | Path, base: str) -> pd.DataFrame:
    """
    Load BASENAME_stimuli.csv (event-based table).
    - Drops corrupt rows where all columns are 0
    - Keeps 'Trial' to map events back to trials
    - Adds minimal metadata: SessionBase, StimuliFile, SessionTime
    """
    folder = Path(folder).expanduser().resolve()
    stimuli_path = folder / f"{base}_stimuli.csv"
    if not stimuli_path.exists():
        # try case-insensitive fallback (Windows usually doesn't need it, but harmless)
        matches = [p for p in folder.glob("*.csv") if p.name.lower() == f"{base}_stimuli.csv".lower()]
        if not matches:
            raise FileNotFoundError(f"Missing stimuli file for session '{base}': {stimuli_path}")
        stimuli_path = matches[0]

    # Your files are tab-delimited (as in your example)
    df = pd.read_csv(stimuli_path, sep="\t", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Drop rows where all entries are 0 (common corrupt rows)
    if len(df) > 0:
        num = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df = df.loc[(num != 0).any(axis=1)].copy()

    # Also drop Trial==0 if present (usually same corrupt rows)
    if "Trial" in df.columns:
        trial = pd.to_numeric(df["Trial"], errors="coerce").fillna(0).astype(int)
        df = df.loc[trial != 0].copy()

    # Metadata
    mtime = stimuli_path.stat().st_mtime
    df.insert(0, "SessionBase", base)
    df.insert(1, "StimuliFile", str(stimuli_path))
    df.insert(2, "SessionTime", pd.to_datetime(mtime, unit="s"))

    # Nice ordering if available
    if "Time_ms" in df.columns and "Trial" in df.columns:
        df["Time_ms"] = pd.to_numeric(df["Time_ms"], errors="coerce")
        df = df.sort_values(["Trial", "Time_ms"], kind="mergesort").reset_index(drop=True)

    return df
