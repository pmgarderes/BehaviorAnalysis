from __future__ import annotations

import re
import pandas as pd


def add_header_fields(trial_data: pd.DataFrame) -> pd.DataFrame:
    """
    Add selected header-derived fields to a trial-level table.

    Requires in trial_data:
      - SessionBase (str)
      - Header (dict or None)  -> produced by your builder
      - StimNum (int-like)     -> trial stimulus index (for MW_* lookup)

    Adds session-level columns (same value for all trials in a session):
      - ArduinoMode
      - Servo_Amplitude
      - TS_G ... TS_Z          (TrialSettings letter params as numeric; if present)
      - lick0thr_ms            (alias for TS_G)
      - lick1thr_ms            (alias for TS_H)
      - RW_start_ms            (alias for TS_I)
      - RW_dur_ms              (alias for TS_J)
      - Solenoid_us            (alias for TS_K)
      - Timeout_ms             (alias for TS_L)
      - behavMode              (alias for TS_N)
      - UseAutoReward          (alias for TS_O)
      - RewardTime_ms          (alias for TS_P)
      - tone_onset_ms          (alias for TS_T)
      - tone_freq_Hz           (alias for TS_U)
      - tone_dur_ms            (alias for TS_V)
      - TaskIs2AFC              (alias for TS_X)  (0=GNG, 1=2AFC)

    Adds per-trial columns (depends on StimNum):
      - MW_ElementList
      - MW_RelAmp
      - MW_Tone
      - MW_Prob
      - MW_target
      - MW_Laser
      - MW_v3Jit
      - MW_v3freq
    """
    out = trial_data.copy()

    # ---- helpers ----
    def _to_float(x):
        try:
            if x is None:
                return None
            if isinstance(x, (int, float)):
                return float(x)
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def _get_any(h: dict, keys: list[str]):
        # try exact match, then case-insensitive match
        for k in keys:
            if k in h:
                return h[k]
        low = {str(k).strip().lower(): k for k in h.keys()}
        for k in keys:
            kk = str(k).strip().lower()
            if kk in low:
                return h[low[kk]]
        return None

    mw_re = re.compile(r"^MW_(ElementList|RelAmp|Tone|Prob|target|Laser|v3Jit|v3freq)(\d+)$", re.IGNORECASE)

    # cache per session
    sess_cache: dict[str, dict] = {}

    for sess, g in out.groupby("SessionBase", sort=False):
        # pick one header dict for the session
        hdr = None
        for v in g["Header"].tolist():
            if isinstance(v, dict) and len(v) > 0:
                hdr = v
                break

        sess_info = {
            "ArduinoMode": None,
            "Servo_Amplitude": None,
            "TS": {c: None for c in list("GHIJKLMNOPQRSTUVWXYZ")},
            "MW": {},  # MW[field][stimnum] = value
        }

        if isinstance(hdr, dict) and len(hdr) > 0:
            # (1) ArduinoMode
            sess_info["ArduinoMode"] = _to_float(_get_any(hdr, ["ArduinoMode"]))

            # (2) Servo amplitude
            # accept either explicit key or the TrialSettings letter Z
            servo_amp = _get_any(hdr, ["Servo_Amplitude", "ServoAmplitude", "Servo_Amplitude "])
            if servo_amp is None:
                # often stored as "Z" or "TrialSettingsZ" or similar
                servo_amp = _get_any(hdr, ["Z", "TrialSettingsZ", "TrialSettings_Z", "Servo movement amplitude"])
            sess_info["Servo_Amplitude"] = _to_float(servo_amp)

            # (3) TrialSettings letter params G..Z
            # try direct keys like "G" or "TrialSettingsG" etc.
            for c in list("GHIJKLMNOPQRSTUVWXYZ"):
                v = _get_any(hdr, [c, f"TrialSettings{c}", f"TrialSettings_{c}", f"TS_{c}"])
                sess_info["TS"][c] = _to_float(v)

            # (4) MW_* per StimNum (keys like MW_RelAmp4)
            for k, v in hdr.items():
                m = mw_re.match(str(k).strip())
                if not m:
                    continue
                field = m.group(1)
                stimnum = int(m.group(2))
                field_std = field[0].upper() + field[1:]  # normalize e.g. RelAmp / ElementList
                sess_info["MW"].setdefault(field_std, {})[stimnum] = _to_float(v) if field_std != "ElementList" else v

        sess_cache[sess] = sess_info

    # ---- write session-level columns ----
    out["ArduinoMode"] = out["SessionBase"].map(lambda s: sess_cache[s]["ArduinoMode"])
    out["Servo_Amplitude"] = out["SessionBase"].map(lambda s: sess_cache[s]["Servo_Amplitude"])

    for c in list("GHIJKLMNOPQRSTUVWXYZ"):
        out[f"TS_{c}"] = out["SessionBase"].map(lambda s, cc=c: sess_cache[s]["TS"][cc])

    # named aliases (handy)
    alias = {
        "lick0thr_ms": "G",
        "lick1thr_ms": "H",
        "RW_start_ms": "I",
        "RW_dur_ms": "J",
        "Solenoid_us": "K",
        "Timeout_ms": "L",
        "behavMode": "N",
        "UseAutoReward": "O",
        "RewardTime_ms": "P",
        "tone_onset_ms": "T",
        "tone_freq_Hz": "U",
        "tone_dur_ms": "V",
        "TaskIs2AFC": "X",
    }
    for name, letter in alias.items():
        out[name] = out[f"TS_{letter}"]

    # ---- per-trial MW columns (look up by StimNum) ----
    stimnum = pd.to_numeric(out.get("StimNum", pd.Series([pd.NA] * len(out))), errors="coerce").astype("Int64")
    out["_StimNum_int"] = stimnum

    def _mw_lookup(row, field_std: str):
        s = row["SessionBase"]
        sn = row["_StimNum_int"]
        if pd.isna(sn):
            return None
        return sess_cache[s]["MW"].get(field_std, {}).get(int(sn), None)

    # keep these as scalar trial columns
    out["MW_ElementList"] = out.apply(lambda r: _mw_lookup(r, "ElementList"), axis=1)
    out["MW_RelAmp"] = out.apply(lambda r: _mw_lookup(r, "RelAmp"), axis=1)
    out["MW_Tone"] = out.apply(lambda r: _mw_lookup(r, "Tone"), axis=1)
    out["MW_Prob"] = out.apply(lambda r: _mw_lookup(r, "Prob"), axis=1)
    out["MW_target"] = out.apply(lambda r: _mw_lookup(r, "Target"), axis=1)
    out["MW_Laser"] = out.apply(lambda r: _mw_lookup(r, "Laser"), axis=1)
    out["MW_v3Jit"] = out.apply(lambda r: _mw_lookup(r, "V3Jit"), axis=1)
    out["MW_v3freq"] = out.apply(lambda r: _mw_lookup(r, "V3freq"), axis=1)

    out = out.drop(columns=["_StimNum_int"])
    return out
