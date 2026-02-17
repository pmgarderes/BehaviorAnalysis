from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_trials_for_session(folder: str | Path, base: str) -> pd.DataFrame:
    """
    Load BASENAME_trials.csv (trial-based table).
    - Loads ALL columns
    - Adds metadata: SessionBase, TrialsFile, SessionTime, TrialInSession
    """
    folder = Path(folder).expanduser().resolve()
    trials_path = folder / f"{base}_trials.csv"

    if not trials_path.exists():
        # mild case-insensitive fallback
        matches = [p for p in folder.glob("*.csv") if p.name.lower() == f"{base}_trials.csv".lower()]
        if not matches:
            raise FileNotFoundError(f"Missing trials file for session '{base}': {trials_path}")
        trials_path = matches[0]

    # Your files are tab-delimited (as in your example)
    df = pd.read_csv(trials_path, sep="\t", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    # Metadata
    mtime = trials_path.stat().st_mtime
    df.insert(0, "SessionBase", base)
    df.insert(1, "TrialsFile", str(trials_path))
    df.insert(2, "SessionTime", pd.to_datetime(mtime, unit="s"))
    df.insert(3, "TrialInSession", range(1, len(df) + 1))

    return df
