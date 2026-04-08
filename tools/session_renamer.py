# ═══════════════════════════════════════════════════════════════════
#  Session File Renamer
#  Renames CSVs to  ANIMAL_MMDDYY_IT_SXX_TYPE  convention
#  where TYPE ∈ {header, lick, stimuli, trials}
#
#  USAGE from Google Colab (recommended):
#      import session_renamer
#      session_renamer.ANIMAL  = "PM31"
#      session_renamer.DRY_RUN = True
#      session_renamer.run()
#
#  USAGE local:
#      Set MODE = "local" and LOCAL_DIR below, then call run().
# ═══════════════════════════════════════════════════════════════════

import os
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# ── USER SETTINGS (override in Colab before calling run()) ─────────
ANIMAL    = "SCN2AWTPM1"
MODE      = "colab"          # "colab" | "local"
LOCAL_DIR = "/path/to/data"  # only used when MODE = "local"
DRY_RUN   = True             # True = preview only, False = rename files
# ──────────────────────────────────────────────────────────────────

KNOWN_TYPES = {"header", "lick", "stimuli", "trials"}


# ── Timezone helper ───────────────────────────────────────────────
try:
    from zoneinfo import ZoneInfo
    _PACIFIC = ZoneInfo("America/Los_Angeles")
    def _to_pacific(dt_utc: datetime) -> datetime:
        return dt_utc.replace(tzinfo=timezone.utc).astimezone(_PACIFIC)
except ImportError:
    _PACIFIC = timezone(timedelta(hours=-8))
    def _to_pacific(dt_utc: datetime) -> datetime:
        print("⚠️  zoneinfo not available — using fixed UTC-8 (no DST). "
              "Run: !pip install -q tzdata")
        return dt_utc.replace(tzinfo=timezone.utc).astimezone(_PACIFIC)


# ── File helpers ──────────────────────────────────────────────────
def _get_mtime_pacific(fpath: Path) -> datetime:
    """Return file modification time (or ctime on Windows) in Pacific Time."""
    stat = fpath.stat()
    raw_ts = stat.st_ctime if os.name == "nt" else stat.st_mtime
    return _to_pacific(datetime.utcfromtimestamp(raw_ts))


def _extract_date_from_stem(stem: str) -> datetime | None:
    """Extract embedded date from filename stem. Tries MMDDYYYY then MMDDYY."""
    for pattern, fmt in [
        (r"(?:^|_)(\d{8})(?:_|$)", "%m%d%Y"),
        (r"(?:^|_)(\d{6})(?:_|$)", "%m%d%y"),
    ]:
        m = re.search(pattern, stem)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                continue
    return None


def _extract_segment(stem: str) -> str | None:
    """Return zero-padded 'SXX' anchored to the end of the stem, or None."""
    m = re.search(r"_(S(\d{1,2}))$", stem, re.IGNORECASE)
    return f"S{int(m.group(2)):02d}" if m else None


def _extract_file_type(stem: str) -> str | None:
    """Return the recognised type suffix or None."""
    for t in KNOWN_TYPES:
        if stem.lower().endswith(f"_{t}"):
            return t
    return None


def _strip_type(stem: str) -> str:
    """Remove trailing _type from stem."""
    for t in KNOWN_TYPES:
        if stem.lower().endswith(f"_{t}"):
            return stem[: -(len(t) + 1)]
    return stem


# ── Main entry point ──────────────────────────────────────────────
def run():
    """
    Read module-level settings and execute the rename pipeline.
    Returns plan_df and issues_df so the caller can display them in Colab:

        plan_df, issues_df = session_renamer.run()

        with pd.option_context("display.max_rows", None):
            display(plan_df)

        display(issues_df)   # only flagged rows
    """
    import session_renamer as _self
    plan_df, issues_df = _run(
        animal=_self.ANIMAL,
        mode=_self.MODE,
        local_dir=_self.LOCAL_DIR,
        dry_run=_self.DRY_RUN,
    )
    return plan_df, issues_df


def _run(animal: str, mode: str, local_dir: str, dry_run: bool) -> tuple[pd.DataFrame, pd.DataFrame]:

    # ── 1. Resolve folder ─────────────────────────────────────────
    if mode == "colab":
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        DATA_FOLDER = Path("/content/drive/Shareddrives/BehaviorAnalysis_data") / animal
    else:
        DATA_FOLDER = Path(local_dir)

    assert DATA_FOLDER.is_dir(), f"Folder not found: {DATA_FOLDER}"
    print(f"📂  Scanning: {DATA_FOLDER}\n")

    # ── 2. Collect files and parse metadata ───────────────────────
    csv_files = sorted(DATA_FOLDER.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files.\n")
    if not csv_files:
        raise FileNotFoundError("No CSV files found — check DATA_FOLDER and ANIMAL.")

    file_info = []
    for fpath in csv_files:
        stem      = fpath.stem
        file_type = _extract_file_type(stem)
        base_stem = _strip_type(stem)
        segment   = _extract_segment(base_stem)
        date_dt   = _get_mtime_pacific(fpath)
        date_str  = date_dt.strftime("%m%d%y")
        canonical = re.sub(r"_?S\d{1,2}$", "", base_stem, flags=re.IGNORECASE).strip("_")

        file_info.append({
            "path"      : fpath,
            "date_str"  : date_str,
            "date_meta" : date_dt,
            "date_fname": _extract_date_from_stem(base_stem),
            "segment"   : segment or "S01",
            "file_type" : file_type,
            "canonical" : canonical,
        })

    # ── 3. Assign iteration numbers per date ──────────────────────
    date_to_canonicals: dict[str, list[str]] = defaultdict(list)
    seen = set()
    for info in file_info:
        key = (info["date_str"], info["canonical"])
        if key not in seen:
            date_to_canonicals[info["date_str"]].append(info["canonical"])
            seen.add(key)

    for d in date_to_canonicals:
        date_to_canonicals[d].sort()

    iteration_map = {
        (date_str, canonical): idx
        for date_str, canonicals in date_to_canonicals.items()
        for idx, canonical in enumerate(canonicals, start=1)
    }

    # ── 4. Build rename plan ──────────────────────────────────────
    rename_plan = []
    for info in file_info:
        date_str  = info["date_str"]
        canonical = info["canonical"]
        segment   = info["segment"]
        ftype     = info["file_type"]
        iteration = iteration_map[(date_str, canonical)]

        new_stem = f"{animal}_{date_str}_{iteration:02d}_{segment}"
        if ftype:
            new_stem = f"{new_stem}_{ftype}"

        old_path = info["path"]
        new_name = f"{new_stem}.csv"
        rename_plan.append({
            "old_name"  : old_path.name,
            "new_name"  : new_name,
            "file_type" : ftype or "⚠️ unknown",
            "segment"   : segment,
            "iteration" : iteration,
            "date"      : date_str,
            "date_meta" : info["date_meta"],
            "date_fname": info["date_fname"],
            "changed"   : old_path.name != new_name,
            "old_path"  : old_path,
            "new_path"  : old_path.parent / new_name,
        })

    # ── 5. Preview table ──────────────────────────────────────────
    plan_df = pd.DataFrame([{
        "old_name"     : r["old_name"],
        "new_name"     : r["new_name"],
        "type"         : r["file_type"],
        "date_meta"    : r["date_meta"].strftime("%Y-%m-%d"),
        "date_filename": r["date_fname"].strftime("%Y-%m-%d") if r["date_fname"] else "❓ not found",
        "date_match"   : (1 if r["date_fname"] and r["date_meta"].date() == r["date_fname"].date() else 0)
                          if r["date_fname"] else "❓",
        "iter"         : r["iteration"],
        "segment"      : r["segment"],
        "status"       : "✏️  rename" if r["changed"] else "✅ ok",
    } for r in rename_plan])

    try:
        from IPython.display import display
        display(plan_df)
    except ImportError:
        print(plan_df.to_string(index=False))

    n_changed = sum(r["changed"] for r in rename_plan)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}→ {n_changed} files will be renamed "
          f"out of {len(rename_plan)} total.\n")

    # ── 5b. Issues table ──────────────────────────────────────────
    # Flag 1: date mismatch
    date_mismatch = set(
        (r["date"], r["iteration"], r["segment"])
        for r in rename_plan
        if str(plan_df.loc[rename_plan.index(r), "date_match"]) != "1"
    )

    # Flag 2: sessions/segments with fewer than 4 file types
    from collections import Counter
    session_key = lambda r: (r["date"], r["iteration"], r["segment"])
    type_counts = Counter(session_key(r) for r in rename_plan if r["file_type"] != "⚠️ unknown")
    incomplete = {k for k, v in type_counts.items() if v < 4}

    issue_rows = []
    for r in rename_plan:
        key = session_key(r)
        flags = []
        dm = plan_df.loc[rename_plan.index(r), "date_match"]
        if str(dm) != "1":
            flags.append(f"⚠️ date_match={dm}")
        if key in incomplete:
            flags.append(f"⚠️ only {type_counts[key]}/4 file types")
        if flags:
            issue_rows.append({
                "file"      : r["old_name"],
                "date_meta" : r["date_meta"].strftime("%Y-%m-%d"),
                "date_fname": r["date_fname"].strftime("%Y-%m-%d") if r["date_fname"] else "❓",
                "iter"      : r["iteration"],
                "segment"   : r["segment"],
                "issues"    : "  |  ".join(flags),
            })

    issues_df = pd.DataFrame(issue_rows) if issue_rows else pd.DataFrame(
        columns=["file", "date_meta", "date_fname", "iter", "segment", "issues"]
    )

    if issue_rows:
        print(f"⚠️  {len(set((r['iter'], r['segment'], r['date_meta']) for r in issue_rows))} "
              f"session(s)/segment(s) with issues — inspect issues_df for details.")
    else:
        print("✅  No issues detected.")

    unknown = plan_df[plan_df["type"] == "⚠️ unknown"]
    if not unknown.empty:
        print(f"⚠️  {len(unknown)} file(s) have no recognised type suffix:")
        print(unknown[["old_name"]].to_string(index=False))

    # ── 6. Execute renames ────────────────────────────────────────
    if not dry_run:
        errors = []
        for r in rename_plan:
            if r["changed"]:
                try:
                    r["old_path"].rename(r["new_path"])
                    print(f"  ✅  {r['old_name']}  →  {r['new_name']}")
                except Exception as e:
                    errors.append((r["old_name"], str(e)))
                    print(f"  ❌  {r['old_name']}  →  ERROR: {e}")
        if errors:
            print(f"\n⚠️  {len(errors)} file(s) could not be renamed.")
        else:
            print(f"\n✅  All {n_changed} files renamed successfully.")
    else:
        print("ℹ️  DRY_RUN = True — no files were touched. Set DRY_RUN = False to apply.")

    return plan_df, issues_df
