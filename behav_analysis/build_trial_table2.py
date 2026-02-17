from __future__ import annotations
import pandas as pd


def build_trial_data(data: dict) -> pd.DataFrame:
    rows = []

    for base, d in data.items():
        trials = d.get("trials")
        if trials is None or len(trials) == 0:
            continue

        stim = d.get("stimuli")
        lick = d.get("lick")
        header = d.get("header")

        trials2 = trials.copy()

        # ---- Canonical Trial IDs ----
        if "Trial" in trials2.columns:
            trials2["Trial"] = pd.to_numeric(trials2["Trial"], errors="coerce").astype("Int64")
        elif isinstance(lick, pd.DataFrame) and ("Trial" in lick.columns) and (len(lick) == len(trials2)):
            trials2["Trial"] = pd.to_numeric(lick["Trial"], errors="coerce").astype("Int64").to_numpy()
        else:
            trials2["Trial"] = pd.to_numeric(trials2.get("TrialInSession"), errors="coerce").astype("Int64")

        # ---- Lick maps ----
        lick_time_map, lick_side_map, nlick_map = {}, {}, {}
        if isinstance(lick, pd.DataFrame) and ("Trial" in lick.columns):
            lick_trial = pd.to_numeric(lick["Trial"], errors="coerce").fillna(-1).astype(int)

            lick_times = lick["LickTimes"] if "LickTimes" in lick.columns else [[]] * len(lick)
            lick_time_map = {t: lt for t, lt in zip(lick_trial, lick_times)}

            if "LickSides" in lick.columns:
                lick_side_map = {t: ls for t, ls in zip(lick_trial, lick["LickSides"])}
            else:
                lick_side_map = {t: [] for t in lick_trial}

            if "NLicks" in lick.columns:
                nlick_map = {t: (int(nl) if pd.notna(nl) else 0) for t, nl in zip(lick_trial, lick["NLicks"])}

        # ---- Stimulus processing (FAST) ----
        posn_map, elem_map, time_map, ampl_map = {}, {}, {}, {}
        noise_map, strength_map = {}, {}

        if isinstance(stim, pd.DataFrame) and ("Trial" in stim.columns) and len(stim) > 0:
            stim2 = stim.copy()
            stim2["Trial"] = pd.to_numeric(stim2["Trial"], errors="coerce").astype("Int64")

            # SE_noise from StimElem==10 (take first non-NaN Ampl)
            if "StimElem" in stim2.columns and "Ampl" in stim2.columns:
                noise_df = stim2.loc[stim2["StimElem"] == 10, ["Trial", "Ampl"]].copy()
                noise_df["Ampl"] = pd.to_numeric(noise_df["Ampl"], errors="coerce")
                noise_df = noise_df.dropna(subset=["Trial"]).sort_values("Trial")
                # keep first per Trial
                noise_first = noise_df.dropna(subset=["Ampl"]).drop_duplicates(subset=["Trial"], keep="first")
                noise_map = {int(t): float(a) for t, a in zip(noise_first["Trial"].astype(int), noise_first["Ampl"])}

            # Remove noise events for lists + strength
            if "StimElem" in stim2.columns:
                stim2 = stim2.loc[stim2["StimElem"] != 10].copy()

            # Build per-trial lists (still ok; few groups)
            def _map_list(colname: str) -> dict:
                return stim2.groupby("Trial")[colname].apply(list).to_dict() if colname in stim2.columns else {}

            posn_map = _map_list("Posn")
            elem_map = _map_list("StimElem")
            time_map = _map_list("Time_ms")
            ampl_map = _map_list("Ampl")

            # SE_strength = count * mean(ampl) computed vectorized
            if "Ampl" in stim2.columns:
                a = pd.to_numeric(stim2["Ampl"], errors="coerce")
                stim2 = stim2.assign(_AmplNum=a)

                grp = stim2.groupby("Trial")["_AmplNum"]
                count = grp.size()
                mean = grp.mean()
                strength = (count * mean).fillna(0.0)

                strength_map = {int(t): float(v) for t, v in strength.items() if pd.notna(t)}

        header_dict = header.iloc[0].to_dict() if isinstance(header, pd.DataFrame) and len(header) > 0 else None

        # ---- Build per-trial rows ----
        for _, tr in trials2.iterrows():
            t = tr.get("Trial", None)
            if pd.isna(t):
                continue
            t_int = int(t)

            row = tr.to_dict()
            row["SessionBase"] = base

            row["LickTimes"] = lick_time_map.get(t_int, [])
            row["LickSides"] = lick_side_map.get(t_int, [])
            row["NLicks"] = nlick_map.get(t_int, len(row["LickTimes"]) if row["LickTimes"] is not None else 0)

            row["SE_Posn"] = posn_map.get(t_int, [])
            row["SE_StimElem"] = elem_map.get(t_int, [])
            row["SE_Time_ms"] = time_map.get(t_int, [])
            row["SE_Ampl"] = ampl_map.get(t_int, [])

            row["SE_noise"] = noise_map.get(t_int, pd.NA)
            row["SE_strength"] = strength_map.get(t_int, 0.0)

            row["Header"] = header_dict
            rows.append(row)

    trial_data = pd.DataFrame(rows)

    sort_cols = [c for c in ["SessionTime", "SessionBase", "Trial"] if c in trial_data.columns]
    if sort_cols and not trial_data.empty:
        trial_data = trial_data.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    return trial_data
