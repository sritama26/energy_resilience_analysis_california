from __future__ import annotations
import pandas as pd
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COUNTY = "county"
MONTH = "month"
POP = "census_population_estimate"
SVI = "svi_rpl_themes"
INTERCONNECT = "dgstats_interconnected_proxy_count"
APPS = "dgstats_app_received_count"
OUTAGES = "customers_out"

HAZARD_COUNTS = ["psps_event_count", "calfire_fire_count", "noaa_event_count"]
HAZARD_DAMAGE = "noaa_property_damage_usd"
HAZARD_BURDEN = "hazard_burden"
CES = "ces_ciscorep_pop_weighted"
CONVERSION_RATE = "conversion_rate"
CONVERSION_GAP = "conversion_gap_per_100k_pop"
FEMA_RESL = "fema_resl_score"
GAP_FLAG = "high_conversion_gap_flag"
RECOVERY_WINDOW = 6
RECOVERY_THRESHOLD_FRAC = 0.1
CONCENTRATION_WINDOW = 12


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, name=col)
    return pd.to_numeric(df[col], errors="coerce")


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if MONTH not in df.columns or COUNTY not in df.columns:
        raise ValueError(f"Input must contain '{COUNTY}' and '{MONTH}' columns.")
    df[MONTH] = pd.to_datetime(df[MONTH]).dt.to_period("M")
    df["month_ts"] = df[MONTH].dt.to_timestamp()
    df["year"] = df[MONTH].dt.year
    return df.sort_values([COUNTY, MONTH]).reset_index(drop=True)


def coverage_by_year(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for year, chunk in df.groupby("year"):
        rec = {"year": int(year)}
        for c in cols:
            s = numeric(chunk, c)
            rec[c] = round(float(((s.notna()) & (s != 0)).mean()), 3)
        rows.append(rec)
    return pd.DataFrame(rows).set_index("year")


def assign_svi_terciles(df: pd.DataFrame) -> pd.Series:
    county_svi = numeric(df, SVI).groupby(df[COUNTY]).mean()
    pct = county_svi.rank(pct=True)
    tier = pd.cut(
        pct,
        bins=[0, 1 / 3, 2 / 3, 1.0],
        labels=["Low SVI (low vuln)", "Mid SVI", "High SVI (high vuln)"],
        include_lowest=True,
    )
    return df[COUNTY].map(tier).astype("object")


def assign_tercile(df: pd.DataFrame, col: str, low_label: str, high_label: str,
                    mid_label: str = "Mid") -> pd.Series:
    county_val = numeric(df, col).groupby(df[COUNTY]).mean()
    pct = county_val.rank(pct=True)
    tier = pd.cut(
        pct,
        bins=[0, 1 / 3, 2 / 3, 1.0],
        labels=[low_label, mid_label, high_label],
        include_lowest=True,
    )
    return df[COUNTY].map(tier).astype("object")


def linear_trend(y: pd.Series) -> dict:
    s = y.dropna()
    if len(s) < 3:
        return {"slope_per_year": None, "pearson_r": None, "n": int(len(s))}
    t = np.arange(len(s), dtype=float)
    slope_per_month = np.polyfit(t, s.values, 1)[0]
    r = float(np.corrcoef(t, s.values)[0, 1])
    return {
        "slope_per_year": float(slope_per_month * 12),
        "pearson_r": r,
        "n": int(len(s)),
    }


def analysis_solar_divergence(df: pd.DataFrame, outdir: Path) -> dict:
    df = df.copy()
    df["_svi_tercile"] = assign_svi_terciles(df)
    df["_ic"] = numeric(df, INTERCONNECT).fillna(0)
    df["_apps"] = numeric(df, APPS).fillna(0)
    df["_pop"] = numeric(df, POP)

    grp = df.groupby(["_svi_tercile", "month_ts"], observed=True)
    agg = grp.agg(ic=("_ic", "sum"), apps=("_apps", "sum"), pop=("_pop", "sum"))
    agg["ic_per_100k"] = 1e5 * agg["ic"] / agg["pop"].replace({0: np.nan})
    agg["apps_per_100k"] = 1e5 * agg["apps"] / agg["pop"].replace({0: np.nan})
    agg = agg.reset_index()

    ic_wide = agg.pivot(index="month_ts", columns="_svi_tercile", values="ic_per_100k").sort_index()
    ic_smooth = ic_wide.rolling(12, min_periods=3).mean()

    low, high = "Low SVI (low vuln)", "High SVI (high vuln)"
    gap = None
    gap_trend = {}
    if low in ic_wide.columns and high in ic_wide.columns:
        gap = (ic_wide[low] - ic_wide[high]).rename("low_minus_high_per_100k")
        gap_trend = linear_trend(gap)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tercile in ["Low SVI (low vuln)", "Mid SVI", "High SVI (high vuln)"]:
        if tercile in ic_smooth.columns:
            ax.plot(ic_smooth.index, ic_smooth[tercile], label=tercile, linewidth=2)
    ax.set_title("Solar interconnections per 100k by SVI tercile (12-mo avg)")
    ax.set_ylabel("Interconnections per 100k residents")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "solar_adoption_by_svi.png", dpi=150)
    plt.close(fig)

    if gap is not None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(gap.index, gap.values, color="#333", alpha=0.5, label="monthly gap")
        ax.plot(gap.index, gap.rolling(12, min_periods=3).mean().values,
                color="#c0392b", linewidth=2, label="12-mo avg")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title("Adoption gap: low-vulnerability minus high-vulnerability")
        ax.set_ylabel("Difference in interconnections per 100k")
        ax.set_xlabel("Month")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / "solar_adoption_gap.png", dpi=150)
        plt.close(fig)

    annual = (
        df.assign(year=df["month_ts"].dt.year)
        .groupby(["_svi_tercile", "year"], observed=True)
        .apply(lambda g: 1e5 * g["_ic"].sum() / max(g["_pop"].sum(), 1), include_groups=False)
        .rename("ic_per_100k")
        .reset_index()
    )
    annual.to_csv(outdir / "solar_adoption_annual_by_svi.csv", index=False)
    ic_wide.to_csv(outdir / "solar_adoption_monthly_by_svi.csv")

    return {
        "gap_trend_low_minus_high": gap_trend,
        "gap_widening": (gap_trend.get("slope_per_year") or 0) > 0,
        "figures": ["solar_adoption_by_svi.png", "solar_adoption_gap.png"],
        "tables": ["solar_adoption_annual_by_svi.csv", "solar_adoption_monthly_by_svi.csv"],
    }


def cross_correlation(target: pd.Series, driver: pd.Series, max_lag: int = 6) -> pd.DataFrame:
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        r = target.corr(driver.shift(lag))
        rows.append({"lag_months": lag, "correlation": None if pd.isna(r) else float(r)})
    return pd.DataFrame(rows)


def analysis_hazard_outage(df: pd.DataFrame, outdir: Path) -> dict:
    df = df.copy()
    present_hazards = [c for c in HAZARD_COUNTS if c in df.columns]
    df["_hazard_events"] = sum(numeric(df, c).fillna(0) for c in present_hazards)
    df["_out"] = numeric(df, OUTAGES).fillna(0)

    state = df.groupby("month_ts").agg(
        outages=("_out", "sum"),
        hazard_events=("_hazard_events", "sum"),
    ).sort_index()
    extra_cols = present_hazards + ([HAZARD_DAMAGE] if HAZARD_DAMAGE in df.columns else [])
    for c in extra_cols:
        df[f"_num_{c}"] = numeric(df, c).fillna(0)
        state[c] = df.groupby("month_ts")[f"_num_{c}"].sum()

    contemp = {}
    for c in ["hazard_events"] + present_hazards + ([HAZARD_DAMAGE] if HAZARD_DAMAGE in df.columns else []):
        r = state["outages"].corr(state[c])
        contemp[c] = None if pd.isna(r) else round(float(r), 3)

    xcorr = cross_correlation(state["outages"], state["hazard_events"], max_lag=6)
    valid = xcorr.dropna(subset=["correlation"])
    best = valid.loc[valid["correlation"].idxmax()].to_dict() if len(valid) else {}

    g = df.groupby(COUNTY)
    out_dm = df["_out"] - g["_out"].transform("mean")
    haz_dm = df["_hazard_events"] - g["_hazard_events"].transform("mean")
    within_r = out_dm.corr(haz_dm)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(state.index, state["outages"], color="#2c3e50", linewidth=1.8, label="customers out")
    ax1.set_ylabel("Customers out (statewide sum)", color="#2c3e50")
    ax1.set_xlabel("Month")
    ax2 = ax1.twinx()
    ax2.plot(state.index, state["hazard_events"], color="#e67e22", linewidth=1.5,
             alpha=0.8, label="hazard events")
    ax2.set_ylabel("Hazard events (PSPS+fire+storm)", color="#e67e22")
    ax1.set_title("Outages vs. hazard events, statewide")
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "hazard_vs_outages_timeseries.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#e67e22" if v == (best.get("lag_months")) else "#95a5a6"
              for v in xcorr["lag_months"]]
    ax.bar(xcorr["lag_months"], xcorr["correlation"].fillna(0), color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_title("Cross-correlation: hazard events vs. outages")
    ax.set_xlabel("Lag (months);  positive = hazard leads outages")
    ax.set_ylabel("Correlation")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outdir / "hazard_outage_crosscorr.png", dpi=150)
    plt.close(fig)

    state.to_csv(outdir / "hazard_outage_monthly_statewide.csv")
    xcorr.to_csv(outdir / "hazard_outage_crosscorr.csv", index=False)

    return {
        "hazards_used": present_hazards,
        "contemporaneous_corr_with_outages": contemp,
        "best_lead_lag": {
            "lag_months": None if not best else int(best["lag_months"]),
            "correlation": None if not best else round(float(best["correlation"]), 3),
            "note": "positive lag = hazard leads outages",
        },
        "within_county_corr": None if pd.isna(within_r) else round(float(within_r), 3),
        "figures": ["hazard_vs_outages_timeseries.png", "hazard_outage_crosscorr.png"],
        "tables": ["hazard_outage_monthly_statewide.csv", "hazard_outage_crosscorr.csv"],
    }


def _conversion_trend_by_group(df: pd.DataFrame, tercile_col: str, low_label: str,
                                high_label: str, tag: str, outdir: Path) -> dict:
    d = df.copy()
    d["_rate"] = numeric(d, CONVERSION_RATE)
    d["_gap"] = numeric(d, CONVERSION_GAP)

    rate_wide = (
        d.groupby([tercile_col, "month_ts"], observed=True)["_rate"]
        .mean().unstack(tercile_col).sort_index()
    )
    gap_wide = (
        d.groupby([tercile_col, "month_ts"], observed=True)["_gap"]
        .mean().unstack(tercile_col).sort_index()
    )
    rate_smooth = rate_wide.rolling(12, min_periods=3).mean()

    rate_diff = None
    rate_diff_trend = {}
    gap_diff = None
    gap_diff_trend = {}
    if low_label in rate_wide.columns and high_label in rate_wide.columns:
        rate_diff = (rate_wide[low_label] - rate_wide[high_label]).rename("low_minus_high_rate")
        rate_diff_trend = linear_trend(rate_diff)
    if low_label in gap_wide.columns and high_label in gap_wide.columns:
        gap_diff = (gap_wide[high_label] - gap_wide[low_label]).rename("high_minus_low_gap")
        gap_diff_trend = linear_trend(gap_diff)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label in [low_label, "Mid", high_label]:
        if label in rate_smooth.columns:
            ax.plot(rate_smooth.index, rate_smooth[label], label=label, linewidth=2)
    ax.set_title(f"Conversion rate by {tag} tercile (12-mo avg)")
    ax.set_ylabel("Conversion rate (interconnections / applications)")
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / f"conversion_rate_by_{tag}.png", dpi=150)
    plt.close(fig)

    if gap_diff is not None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(gap_diff.index, gap_diff.values, color="#333", alpha=0.5, label="monthly gap")
        ax.plot(gap_diff.index, gap_diff.rolling(12, min_periods=3).mean().values,
                color="#c0392b", linewidth=2, label="12-mo avg")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"Conversion gap: high {tag} minus low {tag}")
        ax.set_ylabel("Difference in conversion gap per 100k")
        ax.set_xlabel("Month")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"conversion_gap_by_{tag}_divergence.png", dpi=150)
        plt.close(fig)

    rate_wide.to_csv(outdir / f"conversion_rate_monthly_by_{tag}.csv")
    gap_wide.to_csv(outdir / f"conversion_gap_monthly_by_{tag}.csv")

    return {
        "rate_gap_trend_low_minus_high": rate_diff_trend,
        "conversion_gap_trend_high_minus_low": gap_diff_trend,
        "high_group_falling_behind": (rate_diff_trend.get("slope_per_year") or 0) > 0
            or (gap_diff_trend.get("slope_per_year") or 0) > 0,
        "worsening_over_time": (gap_diff_trend.get("slope_per_year") or 0) > 0,
        "figures": [f"conversion_rate_by_{tag}.png", f"conversion_gap_by_{tag}_divergence.png"],
        "tables": [f"conversion_rate_monthly_by_{tag}.csv", f"conversion_gap_monthly_by_{tag}.csv"],
    }


def analysis_conversion_vs_hazard_and_ces(df: pd.DataFrame, outdir: Path) -> dict:
    df = df.copy()
    df["_hazard_tercile"] = assign_tercile(df, HAZARD_BURDEN, "Low hazard burden", "High hazard burden")
    df["_ces_tercile"] = assign_tercile(df, CES, "Low CES", "High CES")

    return {
        "by_hazard_burden": _conversion_trend_by_group(
            df, "_hazard_tercile", "Low hazard burden", "High hazard burden", "hazard", outdir),
        "by_ces": _conversion_trend_by_group(
            df, "_ces_tercile", "Low CES", "High CES", "ces", outdir),
    }


def build_outage_event_curves(df: pd.DataFrame, group_col: str, window: int = RECOVERY_WINDOW,
                               pre: int = 2) -> pd.DataFrame:
    d = df.sort_values([COUNTY, "month_ts"]).copy()
    d["_out100k"] = 1e5 * numeric(d, OUTAGES) / numeric(d, POP)
    present_hazards = [c for c in HAZARD_COUNTS if c in d.columns]
    d["_hazard_events"] = sum(numeric(d, c).fillna(0) for c in present_hazards)
    d["_is_event"] = d["_hazard_events"] > 0

    rows = []
    event_id = 0
    for _, g in d.groupby(COUNTY, sort=False):
        g = g.reset_index(drop=True)
        baseline = g.loc[~g["_is_event"], "_out100k"].median()
        if pd.isna(baseline):
            baseline = g["_out100k"].median()
        if pd.isna(baseline):
            continue
        group_label = g[group_col].iloc[0]
        for i in g.index[g["_is_event"]]:
            event_id += 1
            for rel in range(-pre, window + 1):
                j = i + rel
                if 0 <= j < len(g) and pd.notna(g.loc[j, "_out100k"]):
                    rows.append({
                        "group": group_label,
                        "event_id": event_id,
                        "rel_month": rel,
                        "response": float(g.loc[j, "_out100k"] - baseline),
                    })
    return pd.DataFrame(rows)


def _resilience_comparison(stats: dict, low_label: str, high_label: str) -> dict | None:
    if low_label not in stats or high_label not in stats:
        return None
    lo, hi = stats[low_label], stats[high_label]
    larger_peak = (lo["peak_response_per_100k"] > hi["peak_response_per_100k"]
                   if None not in (lo["peak_response_per_100k"], hi["peak_response_per_100k"]) else None)
    slower_recovery = (lo["months_to_recover"] > hi["months_to_recover"]
                        if None not in (lo["months_to_recover"], hi["months_to_recover"]) else None)
    return {"low_resilience_larger_peak_outage": larger_peak,
            "low_resilience_slower_recovery": slower_recovery}


def analysis_recovery_by_resilience(df: pd.DataFrame, outdir: Path) -> dict:
    df = df.copy()
    low_label, high_label = "Low FEMA resilience", "High FEMA resilience"
    df["_resl_tercile"] = assign_tercile(df, FEMA_RESL, low_label, high_label)

    curves = build_outage_event_curves(df, "_resl_tercile")
    if curves.empty:
        return {"note": "No hazard events found; recovery analysis skipped."}

    avg_curve = (
        curves.groupby(["group", "rel_month"])["response"]
        .median().unstack("group").sort_index()
    )

    stats = {}
    for label in [low_label, high_label]:
        if label not in avg_curve.columns:
            continue
        post = avg_curve.loc[avg_curve.index >= 0, label]
        peak = float(post.max())
        peak_month = int(post.idxmax())
        threshold = RECOVERY_THRESHOLD_FRAC * peak if peak > 0 else 0.0
        recovered = post[(post.index > peak_month) & (post <= threshold)]
        recovery_month = int(recovered.index.min()) if len(recovered) else None
        stats[label] = {
            "peak_response_per_100k": round(peak, 2),
            "peak_month": peak_month,
            "months_to_recover": None if recovery_month is None else recovery_month - peak_month,
        }

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [(low_label, "#c0392b"), (high_label, "#2980b9")]:
        if label in avg_curve.columns:
            ax.plot(avg_curve.index, avg_curve[label], marker="o", linewidth=2, label=label, color=color)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("Outage response to hazard events, by FEMA resilience tercile")
    ax.set_xlabel("Months relative to hazard event (0 = event month)")
    ax.set_ylabel("Customers out per 100k, relative to county baseline")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "outage_recovery_by_fema_resilience.png", dpi=150)
    plt.close(fig)

    avg_curve.to_csv(outdir / "outage_recovery_curve_by_fema_resilience.csv")

    return {
        "stats_by_group": stats,
        "comparison": _resilience_comparison(stats, low_label, high_label),
        "figures": ["outage_recovery_by_fema_resilience.png"],
        "tables": ["outage_recovery_curve_by_fema_resilience.csv"],
    }


def gini_coefficient(values: np.ndarray) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return float("nan")
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x) / (n * total)) - (n + 1) / n)


def analysis_gap_flag_concentration(df: pd.DataFrame, outdir: Path) -> dict:
    d = df.copy()
    d["_flag"] = numeric(d, GAP_FLAG).fillna(0)
    d["_svi_tercile"] = assign_svi_terciles(d)
    county_tier = d.groupby(COUNTY)["_svi_tercile"].first()
    high_svi_label = "High SVI (high vuln)"

    flag_wide = d.pivot(index="month_ts", columns=COUNTY, values="_flag").sort_index().fillna(0)
    months = flag_wide.index

    persistence_rows = []
    prev_set = None
    for m in months:
        cur_set = set(flag_wide.columns[flag_wide.loc[m] > 0])
        if prev_set is not None:
            union = prev_set | cur_set
            jaccard = len(prev_set & cur_set) / len(union) if union else None
            persistence_rows.append({"month_ts": m, "jaccard_vs_prior_month": jaccard})
        prev_set = cur_set
    persistence = pd.DataFrame(persistence_rows).set_index("month_ts")
    persistence_trend = linear_trend(persistence["jaccard_vs_prior_month"])

    rolling_counts = flag_wide.rolling(CONCENTRATION_WINDOW, min_periods=CONCENTRATION_WINDOW).sum()
    gini_series = rolling_counts.dropna(how="all").apply(lambda row: gini_coefficient(row.values), axis=1)
    gini_series.name = "gini_flag_concentration"
    gini_trend = linear_trend(gini_series)

    overlap_rows = []
    for m in months:
        flagged = flag_wide.columns[flag_wide.loc[m] > 0]
        if len(flagged) == 0:
            continue
        tiers = county_tier.reindex(flagged)
        overlap_rows.append({
            "month_ts": m,
            "n_flagged": int(len(flagged)),
            "share_high_svi": float((tiers == high_svi_label).mean()),
        })
    overlap = pd.DataFrame(overlap_rows).set_index("month_ts")
    overlap_trend = linear_trend(overlap["share_high_svi"])
    baseline_share = float((county_tier == high_svi_label).mean())

    repeat_offenders = (
        d.groupby(COUNTY)["_flag"].sum().rename("months_flagged").astype(int)
        .to_frame()
        .assign(svi_tercile=lambda t: county_tier.reindex(t.index))
        .sort_values("months_flagged", ascending=False)
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(persistence.index, persistence["jaccard_vs_prior_month"], color="#7f8c8d", alpha=0.5)
    axes[0].plot(persistence.index,
                 persistence["jaccard_vs_prior_month"].rolling(12, min_periods=3).mean(),
                 color="#2980b9", linewidth=2)
    axes[0].set_title("Month-to-month persistence of flagged counties (Jaccard overlap)")
    axes[0].set_ylabel("Jaccard overlap")
    axes[0].grid(alpha=0.3)

    axes[1].plot(gini_series.index, gini_series.values, color="#c0392b", linewidth=2)
    axes[1].set_title(f"Concentration of flag among counties (Gini, trailing {CONCENTRATION_WINDOW}-mo)")
    axes[1].set_ylabel("Gini coefficient")
    axes[1].grid(alpha=0.3)

    axes[2].plot(overlap.index, overlap["share_high_svi"], color="#8e44ad", alpha=0.5, label="monthly")
    axes[2].plot(overlap.index, overlap["share_high_svi"].rolling(12, min_periods=3).mean(),
                 color="#8e44ad", linewidth=2, label="12-mo avg")
    axes[2].axhline(baseline_share, color="gray", linewidth=1, linestyle="--",
                     label=f"expected by chance ({baseline_share:.2f})")
    axes[2].set_title("Share of flagged counties that are high-SVI (vulnerable)")
    axes[2].set_ylabel("Share high-SVI")
    axes[2].set_xlabel("Month")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "gap_flag_concentration.png", dpi=150)
    plt.close(fig)

    persistence.to_csv(outdir / "gap_flag_persistence_monthly.csv")
    gini_series.to_csv(outdir / "gap_flag_gini_monthly.csv")
    overlap.to_csv(outdir / "gap_flag_svi_overlap_monthly.csv")
    repeat_offenders.to_csv(outdir / "gap_flag_repeat_offenders.csv")

    return {
        "persistence_trend": persistence_trend,
        "persistence_increasing": (persistence_trend.get("slope_per_year") or 0) > 0,
        "concentration_gini_trend": gini_trend,
        "concentration_increasing": (gini_trend.get("slope_per_year") or 0) > 0,
        "high_svi_share_trend": overlap_trend,
        "high_svi_share_baseline": round(baseline_share, 3),
        "high_svi_share_above_baseline_and_rising": (
            (overlap["share_high_svi"].mean() > baseline_share)
            and (overlap_trend.get("slope_per_year") or 0) > 0
        ),
        "top_repeat_offenders": repeat_offenders.head(10).reset_index().to_dict("records"),
        "figures": ["gap_flag_concentration.png"],
        "tables": [
            "gap_flag_persistence_monthly.csv",
            "gap_flag_gini_monthly.csv",
            "gap_flag_svi_overlap_monthly.csv",
            "gap_flag_repeat_offenders.csv",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Path to the panel CSV")
    ap.add_argument("--outdir", default=Path("trend_outputs"), type=Path)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = load_panel(args.input)

    dynamic_cols = ([OUTAGES, INTERCONNECT, APPS] + HAZARD_COUNTS
                     + [HAZARD_DAMAGE, HAZARD_BURDEN, CONVERSION_RATE, CONVERSION_GAP, GAP_FLAG])
    coverage = coverage_by_year(df, [c for c in dynamic_cols if c in df.columns])
    coverage.to_csv(args.outdir / "coverage_by_year.csv")

    summary = {
        "rows": int(len(df)),
        "counties": int(df[COUNTY].nunique()),
        "months": int(df[MONTH].nunique()),
        "solar_divergence": analysis_solar_divergence(df, args.outdir),
        "hazard_outage": analysis_hazard_outage(df, args.outdir),
        "conversion_vs_hazard_and_ces": analysis_conversion_vs_hazard_and_ces(df, args.outdir),
        "recovery_by_fema_resilience": analysis_recovery_by_resilience(df, args.outdir),
        "gap_flag_concentration": analysis_gap_flag_concentration(df, args.outdir),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Coverage (fraction non-zero by year) — check before trusting trends ===")
    print(coverage.to_string())
    print("\n=== Solar adoption divergence (low-vuln minus high-vuln) ===")
    print(json.dumps(summary["solar_divergence"]["gap_trend_low_minus_high"], indent=2))
    print("gap widening over time:", summary["solar_divergence"]["gap_widening"])
    print("\n=== Hazard vs. outage co-movement ===")
    print("contemporaneous corr:", json.dumps(summary["hazard_outage"]["contemporaneous_corr_with_outages"]))
    print("best lead-lag:", json.dumps(summary["hazard_outage"]["best_lead_lag"]))
    print("within-county corr:", summary["hazard_outage"]["within_county_corr"])
    print("\n=== Conversion falling behind: high-hazard-burden vs. high-CES counties ===")
    cvh = summary["conversion_vs_hazard_and_ces"]
    print("by hazard_burden: falling behind =", cvh["by_hazard_burden"]["high_group_falling_behind"],
          "| worsening over time =", cvh["by_hazard_burden"]["worsening_over_time"])
    print("by CES: falling behind =", cvh["by_ces"]["high_group_falling_behind"],
          "| worsening over time =", cvh["by_ces"]["worsening_over_time"])
    print("\n=== Outage recovery: low vs. high FEMA resilience ===")
    print(json.dumps(summary["recovery_by_fema_resilience"].get("stats_by_group", {}), indent=2))
    print("comparison:", json.dumps(summary["recovery_by_fema_resilience"].get("comparison")))
    print("\n=== Is the conversion-gap flag concentrating in the same vulnerable counties? ===")
    gfc = summary["gap_flag_concentration"]
    print("persistence trend (Jaccard overlap/yr):", json.dumps(gfc["persistence_trend"]),
          "| increasing:", gfc["persistence_increasing"])
    print("concentration trend (Gini/yr):", json.dumps(gfc["concentration_gini_trend"]),
          "| increasing:", gfc["concentration_increasing"])
    print("high-SVI share of flagged counties trend:", json.dumps(gfc["high_svi_share_trend"]),
          "| baseline:", gfc["high_svi_share_baseline"],
          "| above baseline & rising:", gfc["high_svi_share_above_baseline_and_rising"])
    print("top repeat offenders:", json.dumps(gfc["top_repeat_offenders"][:5]))
    print(f"\nWrote figures + tables to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
