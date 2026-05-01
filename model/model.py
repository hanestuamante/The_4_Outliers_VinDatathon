#!/usr/bin/env python3

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd

P = lambda *a, **kw: print(*a, **kw, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
HELLO_DIR = SCRIPT_DIR.parent.parent  
MODELLING_DIR = HELLO_DIR.parent
HUNG_DIR = MODELLING_DIR / "hungkhuu404"
PROJECT_DIR = MODELLING_DIR.parent
OUT_DIR = HELLO_DIR / "submission" / "official_csv_rebuild" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGETS = ["Revenue", "COGS"]


V26_WEIGHT = 0.60
V14_WEIGHT = 0.40

CAL_SHAPE_S = 0.92

COGS_SCALE = 1.09
SPRING_FACTOR = 1.10
SPRING_MONTHS = [4, 5, 6]

V52_CURRENT_W = 0.68
V52_ALT_W = 0.32

DAILY_SHAPE_S = 0.66
MONTH_SHAPE_S = 0.45

SHAPE_STRENGTH = 0.15
RAW_KEEP = 1.00
RIDGE_ALPHA = 16.0


def load_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", *TARGETS]].sort_values("Date").reset_index(drop=True)


def sanitize(df):
    out = df[["Date", *TARGETS]].copy()
    for t in TARGETS:
        out[t] = out[t].astype(float).replace([np.inf, -np.inf], np.nan)
        out[t] = out[t].interpolate(limit_direction="both").fillna(out[t].median())
        out[t] = out[t].clip(lower=1.0).round(2)
    return out.sort_values("Date").reset_index(drop=True)


def load_sales():
    for p in [HELLO_DIR/"submission"/"official_csv_rebuild"/"clean"/"sales.parquet",
              PROJECT_DIR/"datathon-2026-round-1"/"sales.csv"]:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if "Date" in df.columns: df = df.rename(columns={"Date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            return df.sort_values("date").reset_index(drop=True)
    raise FileNotFoundError("sales not found")


def load_sample():
    for p in [HELLO_DIR/"submission"/"official_csv_rebuild"/"clean"/"sample_submission.parquet",
              PROJECT_DIR/"datathon-2026-round-1"/"sample_submission.csv"]:
        if p.exists():
            df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
            if "date" in df.columns: df = df.rename(columns={"date": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])
            return df[["Date", *TARGETS]].sort_values("Date").reset_index(drop=True)
    raise FileNotFoundError("sample_submission not found")


# ═══════════════════════════════════════════════════════
# STEP 1: Load base models v14 + v26
# ═══════════════════════════════════════════════════════
def step1_load_base():
    P("\n[STEP 1] Loading v14 + v26 base models...")
    v26 = load_csv(HUNG_DIR / "submission_v26_optuna_tuned.csv")
    v14 = load_csv(HUNG_DIR / "submission_v14d_a8_f.csv")
    P(f"  v26: Rev mean = {v26['Revenue'].mean():,.0f}")
    P(f"  v14: Rev mean = {v14['Revenue'].mean():,.0f}")
    return v14, v26


# ═══════════════════════════════════════════════════════
# STEP 2: Blend v14 + v26 → v35 anchor
# ═══════════════════════════════════════════════════════
def step2_blend(v14, v26):
    P(f"\n[STEP 2] Blending v26({V26_WEIGHT}) + v14({V14_WEIGHT}) → v35 anchor...")
    out = v26[["Date"]].copy()
    for t in TARGETS:
        out[t] = v26[t] * V26_WEIGHT + v14[t] * V14_WEIGHT
    out = sanitize(out)
    P(f"  v35 anchor: Rev mean = {out['Revenue'].mean():,.0f}")
    return out


# ═══════════════════════════════════════════════════════
# STEP 3: Calendar shape (v41 logic)
# ═══════════════════════════════════════════════════════
def step3_cal_shape(anchor, sales):
    P(f"\n[STEP 3] Calendar shape refinement (s={CAL_SHAPE_S})...")
    hist = sales.copy()
    hist["month"] = hist["date"].dt.month
    hist["dom"] = hist["date"].dt.day
    hist["dow"] = hist["date"].dt.dayofweek
    hist["period"] = hist["date"].dt.to_period("M")

    out = anchor.copy()
    periods = out["Date"].dt.to_period("M")

    for t in TARGETS:
        mm = hist.groupby("period")[t].transform("mean").astype(float)
        hist[f"shape_{t}"] = (hist[t].astype(float) / mm).clip(0.2, 4.0)
        dom_prof = hist.groupby(["month", "dom"])[f"shape_{t}"].mean().to_dict()
        dow_prof = hist.groupby(["month", "dow"])[f"shape_{t}"].mean().to_dict()

        hist_shape = []
        for _, row in out.iterrows():
            m, d, w = row["Date"].month, row["Date"].day, row["Date"].dayofweek
            hist_shape.append(np.nanmean([dom_prof.get((m, d), 1.0), dow_prof.get((m, w), 1.0)]))
        hist_shape = pd.Series(hist_shape, index=out.index).clip(0.2, 4.0)

        anchor_mm = out.groupby(periods)[t].transform("mean").astype(float)
        anchor_shape = (out[t].astype(float) / anchor_mm).clip(0.2, 4.0)
        new_shape = anchor_shape * np.power((hist_shape / anchor_shape).clip(0.25, 4.0), CAL_SHAPE_S)

        adjusted = out[t].astype(float).copy()
        for _, idx in pd.DataFrame({"p": periods}).groupby("p").groups.items():
            idx = list(idx)
            s = np.asarray(new_shape.iloc[idx], float)
            s = np.clip(s, 0.2, 4.0); s = s / s.mean()
            adjusted.iloc[idx] = anchor_mm.iloc[idx].values * s
        out[t] = adjusted

    out = sanitize(out)
    P(f"  After cal shape: Rev mean = {out['Revenue'].mean():,.0f}")
    return out


# ═══════════════════════════════════════════════════════
# STEP 4: COGS scale + Spring boost (v45 logic)
# ═══════════════════════════════════════════════════════
def step4_cogs_spring(base):
    P(f"\n[STEP 4] COGS ×{COGS_SCALE} + Spring ×{SPRING_FACTOR} (M{SPRING_MONTHS})...")
    out = base.copy()
    # COGS scale
    out["COGS"] = out["COGS"].astype(float) * COGS_SCALE

    # Spring reallocation with mean preservation
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    original = out.copy()
    mask = out["month"].isin(SPRING_MONTHS)

    for t in TARGETS:
        out.loc[mask, t] = out.loc[mask, t].astype(float) * SPRING_FACTOR
        # Preserve yearly mean
        for yr in out["year"].unique():
            yr_idx = out[out["year"] == yr].index
            cur_mean = out.loc[yr_idx, t].mean()
            orig_mean = original.loc[yr_idx, t].mean()
            if cur_mean > 0:
                out.loc[yr_idx, t] = out.loc[yr_idx, t].astype(float) * (orig_mean / cur_mean)

    out = sanitize(out)
    P(f"  After COGS+Spring: Rev={out['Revenue'].mean():,.0f}, COGS={out['COGS'].mean():,.0f}")
    return out


# ═══════════════════════════════════════════════════════
# STEP 5: Blend with v26-bestfamily (v52 logic)
# ═══════════════════════════════════════════════════════
def step5_v52_blend(current, sales):
    P(f"\n[STEP 5] v52 blend: current({V52_CURRENT_W}) + v26-bestfamily({V52_ALT_W})...")
    # Build v26-bestfamily: pure v26 with same cal_shape + COGS + spring
    v26 = load_csv(HUNG_DIR / "submission_v26_optuna_tuned.csv")
    v26_shaped = step3_cal_shape(v26, sales)
    v26_boosted = step4_cogs_spring(v26_shaped)

    out = current[["Date"]].copy()
    for t in TARGETS:
        out[t] = current[t] * V52_CURRENT_W + v26_boosted[t] * V52_ALT_W
    out = sanitize(out)
    P(f"  After v52 blend: Rev={out['Revenue'].mean():,.0f}")
    return out


# ═══════════════════════════════════════════════════════
# STEP 6: Seasonal prior from BTC baseline (v62 logic)
# Uses historical seasonal profile instead of sample_submission
# ═══════════════════════════════════════════════════════
def step6_seasonal_prior(current, sales):
    P(f"\n[STEP 6] BTC seasonal prior (daily={DAILY_SHAPE_S}, month={MONTH_SHAPE_S})...")
    # Build BTC seasonal baseline from historical data
    hist = sales.copy()
    hist["year"] = hist["date"].dt.year
    hist["month"] = hist["date"].dt.month
    hist["dom"] = hist["date"].dt.day
    hist["period"] = hist["date"].dt.to_period("M")

    prior = current[["Date"]].copy()
    periods_out = current["Date"].dt.to_period("M")

    for t in TARGETS:
        # Build annual-normalized seasonal profile from history
        ann_mean = hist.groupby("year")[t].transform("mean").astype(float)
        hist[f"norm_{t}"] = hist[t].astype(float) / ann_mean
        md_prof = hist.groupby(["month", "dom"])[f"norm_{t}"].mean().to_dict()

        # Project to forecast dates using BTC growth
        last_year = int(hist["year"].max())
        base_level = float(hist[hist["year"] == last_year][t].mean())
        annual = hist.groupby("year")[t].sum()
        yoy = annual.pct_change().dropna()
        growth = float((1 + yoy.iloc[-1]))  # last year growth

        vals = []
        for _, row in current.iterrows():
            dt = row["Date"]
            years_ahead = (dt.year - last_year) + (dt.month - 6.5) / 12.0
            level = base_level * np.power(growth, years_ahead * 1.5)
            season = md_prof.get((dt.month, dt.day), 1.0)
            vals.append(level * season)
        prior[t] = vals

    prior = sanitize(prior)

    # Apply daily shape (same as v62 apply_sample_shape)
    out = current.copy()
    for t in TARGETS:
        cur_mm = current.groupby(periods_out)[t].transform("mean").astype(float)
        pri_mm = prior.groupby(prior["Date"].dt.to_period("M"))[t].transform("mean").astype(float)
        cur_shape = (current[t].astype(float) / cur_mm).clip(0.2, 4.0)
        pri_shape = (prior[t].astype(float) / pri_mm).clip(0.2, 4.0)
        new_shape = cur_shape * np.power((pri_shape / cur_shape).clip(0.25, 4.0), DAILY_SHAPE_S)

        adjusted = out[t].astype(float).copy()
        for _, idx in pd.DataFrame({"p": periods_out}).groupby("p").groups.items():
            idx = list(idx)
            s = np.asarray(new_shape.iloc[idx], float)
            s = np.clip(s, 0.2, 4.0); s = s / s.mean()
            adjusted.iloc[idx] = cur_mm.iloc[idx].values * s
        out[t] = adjusted

    # Apply month-year prior (same as v62 apply_sample_month_prior)
    out["period"] = out["Date"].dt.to_period("M")
    out["year"] = out["Date"].dt.year
    prior["period"] = prior["Date"].dt.to_period("M")
    prior["year"] = prior["Date"].dt.year

    for t in TARGETS:
        base_month_mean = out.groupby("period")[t].mean()
        prior_yr_mean = prior.groupby("year")[t].transform("mean").astype(float)
        prior_rel = prior[t].astype(float) / prior_yr_mean
        prior_month_rel = prior_rel.groupby(prior["period"]).mean()
        base_year_mean = out.groupby("year")[t].mean()
        desired = pd.Series({
            period: float(base_year_mean.loc[period.year]) * float(rel)
            for period, rel in prior_month_rel.items()
        })
        ratio = (desired / base_month_mean).reindex(base_month_mean.index).astype(float).clip(0.5, 1.5)
        out[t] = out[t].astype(float) * out["period"].map(np.power(ratio, MONTH_SHAPE_S)).astype(float)

    out = sanitize(out)
    P(f"  After seasonal prior: Rev={out['Revenue'].mean():,.0f}")
    return out


# ═══════════════════════════════════════════════════════
# STEP 7: Ridge recalibration (v73 logic)
# ═══════════════════════════════════════════════════════
def safe_feature_matrix(dates):
    d = pd.to_datetime(dates)
    out = pd.DataFrame(index=np.arange(len(d)))
    iso = d.dt.isocalendar()
    out["year"] = d.dt.year; out["dom"] = d.dt.day; out["doy"] = d.dt.dayofyear
    out["week"] = iso.week.astype(int); out["dim"] = d.dt.days_in_month
    out["dtme"] = out["dim"] - out["dom"]
    out["dom_frac"] = (out["dom"] - 1) / out["dim"].clip(lower=1)
    out["is_we"] = (d.dt.dayofweek >= 5).astype(int)
    out["ms3"] = (out["dom"] <= 3).astype(int); out["ms5"] = (out["dom"] <= 5).astype(int)
    out["me3"] = (out["dtme"] <= 2).astype(int); out["me5"] = (out["dtme"] <= 4).astype(int)
    out["cash"] = ((out["dom"] <= 5) | (out["dtme"] <= 6)).astype(int)
    out["mid"] = ((out["dom"] >= 14) & (out["dom"] <= 16)).astype(int)
    out["dom_eq_m"] = (out["dom"] == d.dt.month).astype(int)
    out["sin_dow"] = np.sin(2*np.pi*d.dt.dayofweek/7); out["cos_dow"] = np.cos(2*np.pi*d.dt.dayofweek/7)
    out["sin_m"] = np.sin(2*np.pi*d.dt.month/12); out["cos_m"] = np.cos(2*np.pi*d.dt.month/12)
    out["sin_dom"] = np.sin(2*np.pi*out["dom_frac"]); out["cos_dom"] = np.cos(2*np.pi*out["dom_frac"])
    out["sin_doy"] = np.sin(2*np.pi*out["doy"]/366); out["cos_doy"] = np.cos(2*np.pi*out["doy"]/366)
    num = out.astype(float)
    for col, vals in [("q", d.dt.quarter), ("mo", d.dt.month), ("dow", d.dt.dayofweek),
                      ("wom", ((d.dt.day-1)//7+1)),
                      ("db", pd.cut(d.dt.day, [0,5,10,15,20,25,32], labels=range(6)))]:
        dummies = pd.get_dummies(vals.astype(str), prefix=col, dtype=float)
        dummies.index = num.index
        num = pd.concat([num, dummies], axis=1)
    return num.loc[:, ~num.columns.duplicated()].replace([np.inf, -np.inf], np.nan).fillna(0)


def step7_ridge(anchor):
    P(f"\n[STEP 7] Ridge recalibration (alpha={RIDGE_ALPHA}, shape_s={SHAPE_STRENGTH}, raw_keep={RAW_KEEP})...")
    x = safe_feature_matrix(anchor["Date"])
    feat_cols = list(x.columns)
    periods = anchor["Date"].dt.to_period("M")

    # Build safe projection
    safe_proj = anchor[["Date"]].copy()
    for t in TARGETS:
        month_mean = anchor.groupby(periods)[t].transform("mean").astype(float)
        raw_shape = (anchor[t].astype(float) / month_mean).clip(0.2, 4.0)
        log_shape = np.log(raw_shape)
        xv = x.astype(float).values; xm = xv.mean(0); xs = xv.std(0); xs[xs == 0] = 1
        z = (xv - xm) / xs; d = np.column_stack([np.ones(len(z)), z])
        reg = np.eye(d.shape[1]) * RIDGE_ALPHA; reg[0, 0] = 0
        beta = np.linalg.solve(d.T @ d + reg, d.T @ log_shape.values)
        pred = pd.Series(np.exp(d @ beta), index=anchor.index).clip(0.2, 4.0)

        adjusted = anchor[t].astype(float).copy()
        for _, idx in pd.DataFrame({"p": periods}).groupby("p").groups.items():
            idx = list(idx)
            s = pred.loc[idx].values; s = s / s.mean()
            adjusted.loc[idx] = month_mean.loc[idx].values * s
        safe_proj[t] = adjusted
    safe_proj = sanitize(safe_proj)

    # Mix projection with anchor (raw_keep controls how much anchor shape to keep)
    mixed = anchor[["Date"]].copy()
    for t in TARGETS:
        anc_mm = anchor.groupby(periods)[t].transform("mean").astype(float)
        saf_mm = safe_proj.groupby(safe_proj["Date"].dt.to_period("M"))[t].transform("mean").astype(float)
        anc_shape = (anchor[t].astype(float) / anc_mm).clip(0.2, 4.0)
        saf_shape = (safe_proj[t].astype(float) / saf_mm).clip(0.2, 4.0)
        mix_shape = np.power(anc_shape, RAW_KEEP) * np.power(saf_shape, 1.0 - RAW_KEEP)

        adjusted = anchor[t].astype(float).copy()
        for _, idx in pd.DataFrame({"p": periods}).groupby("p").groups.items():
            idx = list(idx)
            s = np.asarray(mix_shape.loc[idx], float)
            s = np.clip(s, 0.2, 4.0); s = s / s.mean()
            adjusted.loc[idx] = anc_mm.loc[idx].values * s
        mixed[t] = adjusted
    prior = sanitize(mixed)

    # Move anchor shape toward prior (final v73 step)
    final = anchor[["Date", *TARGETS]].copy()
    for t in TARGETS:
        anc_mm = anchor.groupby(periods)[t].transform("mean").astype(float)
        pri_mm = prior.groupby(prior["Date"].dt.to_period("M"))[t].transform("mean").astype(float)
        anc_shape = (anchor[t].astype(float) / anc_mm).clip(0.2, 4.0)
        pri_shape = (prior[t].astype(float) / pri_mm).clip(0.2, 4.0)
        new_shape = anc_shape * np.power((pri_shape / anc_shape).clip(0.25, 4.0), SHAPE_STRENGTH)

        adjusted = anchor[t].astype(float).copy()
        for _, idx in pd.DataFrame({"p": periods}).groupby("p").groups.items():
            idx = list(idx)
            s = np.asarray(new_shape.loc[idx], float)
            s = np.clip(s, 0.2, 4.0); s = s / s.mean()
            adjusted.loc[idx] = anc_mm.loc[idx].values * s
        final[t] = adjusted

    final = sanitize(final)
    P(f"  After Ridge: Rev={final['Revenue'].mean():,.0f}")
    return final, feat_cols


# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    P("=" * 70)
    P("V80 MASTER PIPELINE — REPRODUCE V73 FROM SCRATCH")
    P("=" * 70)

    sales = load_sales()
    sample = load_sample()

    # Load v73 reference for comparison
    v73_path = HELLO_DIR / "v73_candidates" / "submission_v73_rawkeep100_shape_s15.csv"
    v73 = load_csv(str(v73_path)) if v73_path.exists() else None

    # === FULL CHAIN ===
    v14, v26 = step1_load_base()
    v35 = step2_blend(v14, v26)
    v41 = step3_cal_shape(v35, sales)
    v45 = step4_cogs_spring(v41)
    v52 = step5_v52_blend(v45, sales)
    v62 = step6_seasonal_prior(v52, sales)
    final, feat_cols = step7_ridge(v62)

    # === ALSO TRY: skip step5 (direct v45 → v62 → v73) ===
    v62b = step6_seasonal_prior(v45, sales)
    final_b, _ = step7_ridge(v62b)

    # === ALSO TRY: use v62 file directly if exists ===
    v62_file = HELLO_DIR / "v62_candidates" / "submission_v62_shape_d66_m45.csv"
    final_c = None
    if v62_file.exists():
        P("\n[BONUS] Loading actual v62 file and applying Ridge...")
        v62_actual = load_csv(str(v62_file))
        final_c, _ = step7_ridge(v62_actual)

    # === SUMMARY ===
    P("\n" + "=" * 70)
    P("FINAL SUMMARY")
    P("=" * 70)
    candidates = [
        ("v80_master_full_chain.csv", final),
        ("v80_master_skip_v52.csv", final_b),
    ]
    if final_c is not None:
        candidates.append(("v80_master_v62file_ridge.csv", final_c))

    rows = []
    for name, df in candidates:
        df.to_csv(OUT_DIR / name, index=False)
        margin = (df["Revenue"] - df["COGS"]) / df["Revenue"].clip(lower=1)
        row = {"file": name,
               "rev_mean": f"{df['Revenue'].mean():,.0f}",
               "cogs_mean": f"{df['COGS'].mean():,.0f}",
               "margin": f"{margin.mean():.3f}"}
        if v73 is not None:
            dist = float(np.mean([np.abs(df[t].values - v73[t].values).mean() for t in TARGETS]))
            row["dist_v73"] = f"{dist:,.0f}"
        rows.append(row)

    summary = pd.DataFrame(rows)
    P(summary.to_string(index=False))

    # Feature audit
    P(f"\n--- Ridge features ({len(feat_cols)} total) ---")
    P(f"  All Date-derived. Zero external data.")
    audit = pd.DataFrame({"feature": feat_cols, "source": "Date", "audit": "allowed"})
    audit.to_csv(OUT_DIR / "v80_master_feature_audit.csv", index=False)
    summary.to_csv(OUT_DIR / "v80_master_summary.csv", index=False)
    P(f"\nFiles saved to: {OUT_DIR}")
    P("DONE!")


if __name__ == "__main__":
    main()
