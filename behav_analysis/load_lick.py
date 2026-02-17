from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_lick_for_session(folder: str | Path, base: str) -> pd.DataFrame:
    """
    Load BASENAME_lick.csv and return a TRIAL-level table.

    File format:
      Two vertical blocks with same number of rows:
        Block 1: [Trial, t1, t2, ... 0 0]
        Block 2: [Trial, s1, s2, ... 0 0]  (spout side codes, e.g. 1/2)

    Output columns:
      SessionBase, LickFile, SessionTime, Trial, LickTimes, LickSides, NLicks
    """
    folder = Path(folder).expanduser().resolve()
    path = folder / f"{base}_lick.csv"
    if not path.exists():
        matches = [p for p in folder.glob("*.csv") if p.name.lower() == f"{base}_lick.csv".lower()]
        if not matches:
            raise FileNotFoundError(f"Missing lick file for session '{base}': {path}")
        path = matches[0]

    wide = pd.read_csv(path, sep="\t", header=None, engine="python")
    wide = wide.apply(pd.to_numeric, errors="coerce").fillna(0)

    nrows = len(wide)
    if nrows % 2 != 0:
        raise ValueError(f"Lick file does not have an even number of rows (expected 2 blocks): {path} (nrows={nrows})")

    half = nrows // 2
    times_blk = wide.iloc[:half, :].copy()
    sides_blk = wide.iloc[half:, :].copy()

    trial_t = times_blk.iloc[:, 0].astype(int).to_numpy()
    trial_s = sides_blk.iloc[:, 0].astype(int).to_numpy()

    # Basic alignment check (same trial order)
    if not (trial_t == trial_s).all():
        raise ValueError(
            f"Trial numbers differ between time and side blocks in {path.name}.\n"
            f"First few time trials: {trial_t[:5].tolist()}\n"
            f"First few side trials: {trial_s[:5].tolist()}"
        )

    times = times_blk.iloc[:, 1:].to_numpy()
    sides = sides_blk.iloc[:, 1:].to_numpy()

    lick_times_list = []
    lick_sides_list = []

    for r in range(half):
        trow = times[r, :]
        srow = sides[r, :]

        mask = trow > 0  # keep events with a time
        lick_times = trow[mask].tolist()
        lick_sides = srow[mask].tolist()  # aligned to times

        lick_times_list.append(lick_times)
        lick_sides_list.append(lick_sides)

    out = pd.DataFrame(
        {
            "Trial": trial_t,
            "LickTimes": lick_times_list,
            "LickSides": lick_sides_list,
        }
    )
    out["NLicks"] = out["LickTimes"].apply(len)

    mtime = path.stat().st_mtime
    out.insert(0, "SessionBase", base)
    out.insert(1, "LickFile", str(path))
    out.insert(2, "SessionTime", pd.to_datetime(mtime, unit="s"))

    return out
