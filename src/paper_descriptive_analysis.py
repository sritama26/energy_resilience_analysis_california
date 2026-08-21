from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COUNTY = "county"
MONTH = "month"
INTERCONNECTIONS = "dgstats_interconnected_proxy_count"
APPLICATIONS = "dgstats_app_received_count"
OUTAGES = "customers_out"
NOAA_EVENTS = "noaa_event_count"
CALFIRE_EVENTS = "calfire_fire_count"
PSPS_EVENTS = "psps_event_count"
SVI = "svi_rpl_themes"
STATE = "county_resilience_class"
HIGH_GAP = "high_conversion_gap_flag"
CONVERSION_GAP = "conversion_gap_per_100k_pop"
PENALTY = "hazard_burden"
POPULATION = "census_population_estimate"


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        COUNTY,
        MONTH,
        INTERCONNECTIONS,
        APPLICATIONS,
        OUTAGES,
        NOAA_EVENTS,
        CALFIRE_EVENTS,
        PSPS_EVENTS,
        SVI,
        STATE,
        HIGH_GAP,
        CONVERSION_GAP,
        PENALTY,
        POPULATION,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[MONTH] = pd.to_datetime(df[MONTH])
    df["year"] = df[MONTH].dt.year
    df["month_num"] = df[MONTH].dt.month
    return df.sort_values([COUNTY, MONTH]).reset_index(drop=True)


def write_basic_checks(df: pd.DataFrame, outdir: Path) -> None:
    checks = {
        "rows": len(df),
        "columns": len(df.columns),
        "counties": df[COUNTY].nunique(),
        "months": df[MONTH].nunique(),
        "start_month": df[MONTH].min().strftime("%Y-%m"),
        "end_month": df[MONTH].max().strftime("%Y-%m"),
        "total_interconnections": int(df[INTERCONNECTIONS].sum()),
        "total_applications": int(df[APPLICATIONS].sum()),
        "total_customers_out": int(df[OUTAGES].sum()),
        "total_noaa_events": int(df[NOAA_EVENTS].sum()),
        "total_calfire_events": int(df[CALFIRE_EVENTS].sum()),
        "total_psps_events": int(df[PSPS_EVENTS].sum()),
    }
    pd.Series(checks, name="value").to_csv(outdir / "dataset_summary.csv")
    df.isna().sum().rename("missing_values").to_csv(outdir / "missing_values.csv")


def yearly_totals(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    totals = df.groupby("year", as_index=False).agg(
        interconnections=(INTERCONNECTIONS, "sum"),
        applications=(APPLICATIONS, "sum"),
        customers_out=(OUTAGES, "sum"),
        noaa_events=(NOAA_EVENTS, "sum"),
        calfire_events=(CALFIRE_EVENTS, "sum"),
        psps_events=(PSPS_EVENTS, "sum"),
    )
    totals.to_csv(outdir / "yearly_totals.csv", index=False)
    return totals


def county_summary(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    summary = df.groupby(COUNTY, as_index=False).agg(
        mean_svi=(SVI, "mean"),
        cumulative_customers_out=(OUTAGES, "sum"),
        mean_population=(POPULATION, "mean"),
        total_interconnections=(INTERCONNECTIONS, "sum"),
        total_applications=(APPLICATIONS, "sum"),
        mean_conversion_gap_per_100k=(CONVERSION_GAP, "mean"),
        mean_hazard_burden=(PENALTY, "mean"),
        total_noaa_events=(NOAA_EVENTS, "sum"),
        total_calfire_events=(CALFIRE_EVENTS, "sum"),
        total_psps_events=(PSPS_EVENTS, "sum"),
    )
    summary.to_csv(outdir / "county_summary.csv", index=False)
    return summary


def plot_interconnection_activity(df: pd.DataFrame, outdir: Path) -> None:
    monthly = df.groupby(MONTH, as_index=False).agg(
        interconnections=(INTERCONNECTIONS, "sum"),
        applications=(APPLICATIONS, "sum"),
    )

    plt.figure(figsize=(10, 5))
    plt.plot(monthly[MONTH], monthly["interconnections"], label="Interconnections")
    plt.plot(monthly[MONTH], monthly["applications"], label="Applications")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.title("Monthly Interconnection and Application Activity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "fig1_interconnection_application_trends.png", dpi=300)
    plt.close()


def plot_outage_hazard_trends(df: pd.DataFrame, outdir: Path) -> None:
    monthly = df.groupby(MONTH, as_index=False).agg(
        customers_out=(OUTAGES, "sum"),
        noaa_events=(NOAA_EVENTS, "sum"),
        calfire_events=(CALFIRE_EVENTS, "sum"),
        psps_events=(PSPS_EVENTS, "sum"),
    )

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(monthly[MONTH], monthly["customers_out"], color="black", label="Customers out")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Customers out")

    ax2 = ax1.twinx()
    ax2.plot(monthly[MONTH], monthly["noaa_events"], label="NOAA events")
    ax2.plot(monthly[MONTH], monthly["calfire_events"], label="CAL FIRE events")
    ax2.plot(monthly[MONTH], monthly["psps_events"], label="PSPS events")
    ax2.set_ylabel("Event count")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Monthly Outage and Hazard Burden")
    plt.tight_layout()
    plt.savefig(outdir / "fig2_outage_hazard_trends.png", dpi=300)
    plt.close()


def plot_state_distribution(df: pd.DataFrame, outdir: Path) -> None:
    county_states = df[[COUNTY, STATE]].drop_duplicates()
    state_counts = county_states.groupby(STATE).size().sort_index()
    state_changes = df.groupby(COUNTY)[STATE].nunique()

    transition_check = pd.DataFrame(
        {
            "counties_with_more_than_one_state": [int((state_changes > 1).sum())],
            "maximum_states_per_county": [int(state_changes.max())],
        }
    )
    transition_check.to_csv(outdir / "state_transition_check.csv", index=False)

    plt.figure(figsize=(7, 5))
    state_counts.plot(kind="bar")
    plt.xlabel("County resilience class")
    plt.ylabel("Number of counties")
    plt.title("County Distribution by Resilience Class")
    plt.tight_layout()
    plt.savefig(outdir / "fig3_state_distribution_by_county.png", dpi=300)
    plt.close()


def plot_svi_outage_scatter(summary: pd.DataFrame, outdir: Path) -> None:
    corr = summary["mean_svi"].corr(summary["cumulative_customers_out"])
    pd.Series({"pearson_r": corr}, name="value").to_csv(outdir / "svi_outage_correlation.csv")

    plt.figure(figsize=(8, 6))
    plt.scatter(summary["mean_svi"], summary["cumulative_customers_out"])
    for _, row in summary.nlargest(5, "cumulative_customers_out").iterrows():
        plt.annotate(row[COUNTY], (row["mean_svi"], row["cumulative_customers_out"]))

    plt.xlabel("Mean SVI overall percentile")
    plt.ylabel("Cumulative customers out")
    plt.title("SVI and Cumulative Outage Burden by County")
    plt.tight_layout()
    plt.savefig(outdir / "fig4_svi_outage_burden_scatter.png", dpi=300)
    plt.close()


def plot_conversion_gap(summary: pd.DataFrame, outdir: Path) -> None:
    top_gap = summary.nlargest(15, "mean_conversion_gap_per_100k")

    plt.figure(figsize=(9, 6))
    plt.barh(top_gap[COUNTY], top_gap["mean_conversion_gap_per_100k"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean conversion gap per 100k population")
    plt.ylabel("County")
    plt.title("Counties with Largest Application-to-Interconnection Gap")
    plt.tight_layout()
    plt.savefig(outdir / "fig5_conversion_gap_top_counties.png", dpi=300)
    plt.close()

    top_gap[[COUNTY, "mean_conversion_gap_per_100k"]].to_csv(
        outdir / "top_conversion_gap_counties.csv", index=False
    )


def plot_high_gap_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    heat = df.pivot_table(index=COUNTY, columns=MONTH, values=HIGH_GAP, aggfunc="mean")
    county_order = df.groupby(COUNTY)[SVI].mean().sort_values(ascending=False).index
    heat = heat.loc[county_order]

    plt.figure(figsize=(12, 10))
    plt.imshow(heat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="High conversion-gap flag")
    plt.yticks(range(len(heat.index)), heat.index, fontsize=6)

    labels = [m.strftime("%Y-%m") for m in heat.columns]
    tick_positions = range(0, len(labels), 12)
    plt.xticks(tick_positions, [labels[i] for i in tick_positions], rotation=45)

    plt.xlabel("Month")
    plt.ylabel("County")
    plt.title("High Conversion-Gap Flag by County and Month")
    plt.tight_layout()
    plt.savefig(outdir / "fig6_high_gap_months_heatmap.png", dpi=300)
    plt.close()


def lagged_screening(df: pd.DataFrame, outdir: Path) -> None:
    outcomes = [
        OUTAGES,
        NOAA_EVENTS,
        CALFIRE_EVENTS,
        PSPS_EVENTS,
        PENALTY,
        SVI,
    ]
    lags = [1, 3, 6, 12]
    rows = []

    for outcome in outcomes:
        for lag in lags:
            county_corrs = []
            for _, g in df.sort_values(MONTH).groupby(COUNTY):
                g = g.copy()
                g["future_outcome"] = g[outcome].shift(-lag)
                valid = g[[INTERCONNECTIONS, "future_outcome"]].dropna()
                if len(valid) <= 2:
                    continue
                if valid[INTERCONNECTIONS].std() == 0 or valid["future_outcome"].std() == 0:
                    continue
                county_corrs.append(valid[INTERCONNECTIONS].corr(valid["future_outcome"]))

            if county_corrs:
                rows.append(
                    {
                        "outcome": outcome,
                        "lag_months": lag,
                        "median_county_corr": float(np.median(county_corrs)),
                        "positive_counties": int(sum(c > 0 for c in county_corrs)),
                        "negative_counties": int(sum(c < 0 for c in county_corrs)),
                        "county_count": int(len(county_corrs)),
                    }
                )

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "lagged_screening_all_lags.csv", index=False)
    best = (
        results.assign(abs_median=lambda x: x["median_county_corr"].abs())
        .sort_values(["outcome", "abs_median"], ascending=[True, False])
        .groupby("outcome")
        .head(1)
        .drop(columns="abs_median")
    )
    best.to_csv(outdir / "lagged_screening_best_lags.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/resilience_planning_dataset.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("paper_outputs"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df = load_panel(args.input)

    write_basic_checks(df, args.outdir)
    yearly_totals(df, args.outdir)
    summary = county_summary(df, args.outdir)

    plot_interconnection_activity(df, args.outdir)
    plot_outage_hazard_trends(df, args.outdir)
    plot_state_distribution(df, args.outdir)
    plot_svi_outage_scatter(summary, args.outdir)
    plot_conversion_gap(summary, args.outdir)
    plot_high_gap_heatmap(df, args.outdir)
    lagged_screening(df, args.outdir)

    print(f"Wrote paper figures and summary tables to {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
