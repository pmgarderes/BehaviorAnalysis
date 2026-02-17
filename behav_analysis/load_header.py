from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_header_for_session(folder: str | Path, base: str) -> pd.DataFrame:
    """
    Load BASENAME_header.csv (no recursion).

    Supports:
      A) key/value table (2 columns, many rows) -> converts to 1-row wide table
      B) already-wide table                    -> keeps as-is

    Always adds: SessionBase, HeaderFile, SessionTime
    """
    folder = Path(folder).expanduser().resolve()
    path = folder / f"{base}_header.csv"
    if not path.exists():
        matches = [p for p in folder.glob("*.csv") if p.name.lower() == f"{base}_header.csv".lower()]
        if not matches:
            raise FileNotFoundError(f"Missing header file for session '{base}': {path}")
        path = matches[0]

    # First try: read as raw tab table with no header
    df = pd.read_csv(path, sep="\t", header=None, engine="python")

    # Strip whitespace from string cells (applymap replacement)
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))

    # Case A: 2-col key/value table -> make 1-row wide
    if df.shape[1] == 2 and df.shape[0] > 1:
        keys = df.iloc[:, 0].astype(str).str.strip()
        vals = df.iloc[:, 1]
        wide = pd.DataFrame([vals.to_list()], columns=keys.to_list())
    else:
        # Case B: already-wide (try reading with header row)
        try:
            wide = pd.read_csv(path, sep="\t", engine="python")
        except Exception:
            wide = df.copy()
            wide.columns = [str(c) for c in range(wide.shape[1])]

    wide.columns = [str(c).strip() for c in wide.columns]

    mtime = path.stat().st_mtime
    wide.insert(0, "SessionBase", base)
    wide.insert(1, "HeaderFile", str(path))
    wide.insert(2, "SessionTime", pd.to_datetime(mtime, unit="s"))
    return wide


'''
def load_header_for_session(folder: str | Path, base: str) -> pd.DataFrame:
    """
    Load BASENAME_header.csv (session metadata table).
    Loads ALL columns and adds: SessionBase, HeaderFile, SessionTime
    """
    folder = Path(folder).expanduser().resolve()
    path = folder / f"{base}_header.csv"
    if not path.exists():
        matches = [p for p in folder.glob("*.csv") if p.name.lower() == f"{base}_header.csv".lower()]
        if not matches:
            raise FileNotFoundError(f"Missing header file for session '{base}': {path}")
        path = matches[0]

    df = pd.read_csv(path, sep="\t", engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    mtime = path.stat().st_mtime
    df.insert(0, "SessionBase", base)
    df.insert(1, "HeaderFile", str(path))
    df.insert(2, "SessionTime", pd.to_datetime(mtime, unit="s"))
    return df
'''