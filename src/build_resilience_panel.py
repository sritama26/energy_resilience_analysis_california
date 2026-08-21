from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_LINKS = {
    "DGStats": "https://www.californiadgstats.ca.gov/downloads/",
    "CPUC PSPS reports": "https://www.cpuc.ca.gov/consumer-support/psps/utility-company-psps-reports-post-event-and-post-season",
    "CAL FIRE incidents": "https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters",
    "NOAA Storm Events": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
    "CDC ATSDR SVI": "https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html",
    "CalEnviroScreen": "https://oehha.ca.gov/calenviroscreen",
    "FEMA National Risk Index": "https://hazards.fema.gov/nri/data-resources",
    "Census population estimates": "https://www.census.gov/programs-surveys/popest/data/data-sets.html",
}


OUTPUT_COLUMNS = [
    "county",
    "month",
    "customers_out",
    "year",
    "month_num",
    "county_resilience_class",
    "dgstats_interconnected_proxy_count",
    "dgstats_app_received_count",
    "conversion_gap_per_100k_pop",
    "conversion_rate",
    "high_conversion_gap_flag",
    "hazard_burden",
    "svi_rpl_themes",
    "svi_rpl_theme1",
    "svi_rpl_theme2",
    "svi_rpl_theme3",
    "svi_rpl_theme4",
    "fema_risk_score",
    "fema_sovi_score",
    "fema_resl_score",
    "ces_ciscorep_pop_weighted",
    "noaa_property_damage_usd",
    "noaa_event_count",
    "calfire_fire_count",
    "psps_event_count",
    "census_population_estimate",
]


DYNAMIC_ZERO_COLUMNS = [
    "customers_out",
    "dgstats_interconnected_proxy_count",
    "dgstats_app_received_count",
    "high_conversion_gap_flag",
    "noaa_property_damage_usd",
    "noaa_event_count",
    "calfire_fire_count",
    "psps_event_count",
]


CONTEXT_COLUMNS = [
    "county_resilience_class",
    "svi_rpl_themes",
    "svi_rpl_theme1",
    "svi_rpl_theme2",
    "svi_rpl_theme3",
    "svi_rpl_theme4",
    "fema_risk_score",
    "fema_sovi_score",
    "fema_resl_score",
    "ces_ciscorep_pop_weighted",
    "census_population_estimate",
]


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.copy()


def clean_month_county(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["county"] = df["county"].astype(str).str.strip()
    df["month"] = pd.to_datetime(df["month"]).dt.to_period("M").astype(str)
    dt = pd.to_datetime(df["month"])
    df["year"] = dt.dt.year
    df["month_num"] = dt.dt.month
    return df.sort_values(["county", "month"]).reset_index(drop=True)


def complete_county_month_index(df: pd.DataFrame) -> pd.DataFrame:
    counties = sorted(df["county"].dropna().unique())
    months = pd.period_range(df["month"].min(), df["month"].max(), freq="M").astype(str)
    full_index = pd.MultiIndex.from_product([counties, months], names=["county", "month"])
    df = df.set_index(["county", "month"]).reindex(full_index).reset_index()
    dt = pd.to_datetime(df["month"])
    df["year"] = dt.dt.year
    df["month_num"] = dt.dt.month
    return df


def fill_expected_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in DYNAMIC_ZERO_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    present_context = [col for col in CONTEXT_COLUMNS if col in df.columns]
    df[present_context] = df.groupby("county", group_keys=False)[present_context].apply(
        lambda g: g.ffill().bfill()
    )

    if "conversion_gap_per_100k_pop" not in df.columns:
        df["conversion_gap_per_100k_pop"] = np.nan

    if "conversion_rate" not in df.columns:
        df["conversion_rate"] = np.nan

    pop = pd.to_numeric(df.get("census_population_estimate"), errors="coerce")
    apps = pd.to_numeric(df.get("dgstats_app_received_count"), errors="coerce").fillna(0)
    connected = pd.to_numeric(df.get("dgstats_interconnected_proxy_count"), errors="coerce").fillna(0)

    missing_gap = df["conversion_gap_per_100k_pop"].isna()
    df.loc[missing_gap, "conversion_gap_per_100k_pop"] = (
        (apps - connected).clip(lower=0) / pop.replace({0: np.nan}) * 100000
    )

    missing_rate = df["conversion_rate"].isna() & (apps > 0)
    df.loc[missing_rate, "conversion_rate"] = connected[missing_rate] / apps[missing_rate]

    if "hazard_burden" not in df.columns:
        df["hazard_burden"] = np.nan

    return df


def validate_panel(df: pd.DataFrame) -> None:
    missing_cols = [col for col in OUTPUT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing output columns: {missing_cols}")

    duplicate_rows = df.duplicated(["county", "month"]).sum()
    if duplicate_rows:
        raise ValueError(f"Found duplicate county-month rows: {duplicate_rows}")

    county_count = df["county"].nunique()
    month_count = df["month"].nunique()
    expected_rows = county_count * month_count
    if len(df) != expected_rows:
        raise ValueError(f"Panel is not complete: {len(df)} rows, expected {expected_rows}")


def build_panel(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = load_input(input_path)
    df = clean_month_county(df)
    df = complete_county_month_index(df)
    df = fill_expected_values(df)
    validate_panel(df)

    out = df[OUTPUT_COLUMNS].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/resilience_planning_dataset.csv"))
    args = parser.parse_args()

    panel = build_panel(args.input, args.output)
    print(f"Wrote {len(panel):,} rows and {len(panel.columns)} columns to {args.output}")
    print(f"Coverage: {panel['county'].nunique()} counties, {panel['month'].nunique()} months")


if __name__ == "__main__":
    main()
