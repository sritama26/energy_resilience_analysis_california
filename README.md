# Energy Resilience Analysis California

This repository supports a county-month dataset paper on California energy resilience, clean-energy adoption, outage burden, hazard exposure, and community vulnerability from January 2017 through December 2024.

The released panel is designed for reproducible descriptive analysis and downstream modeling. Each row is one California county in one month.

## Repository structure

```text
.
├── data/
│   ├── resilience_planning_dataset.csv
│   └── resilience_planning_dataset_data_dictionary.xlsx
└── src/
    ├── build_resilience_panel.py
    ├── paper_descriptive_analysis.py
    └── resilience_trend_analysis.py
```

## Main dataset

`data/resilience_planning_dataset.csv`

Panel coverage:

- 58 California counties
- 96 months
- January 2017 to December 2024
- 5,568 county-month rows
- 26 released columns

The data dictionary in `data/resilience_planning_dataset_data_dictionary.xlsx` gives column-level descriptions.

## Data sources

The panel combines public administrative, climate, energy, and demographic sources. The source links below identify the public data families used for the curation.

| Source | Role in panel | Public link |
|---|---|---|
| California Distributed Generation Statistics | Solar interconnection and application activity | https://www.californiadgstats.ca.gov/downloads/ |
| CPUC Public Safety Power Shutoff reports | PSPS event exposure | https://www.cpuc.ca.gov/consumer-support/psps/utility-company-psps-reports-post-event-and-post-season |
| CAL FIRE fire perimeter / incident resources | Wildfire event burden | https://www.fire.ca.gov/what-we-do/fire-resource-assessment-program/fire-perimeters |
| NOAA Storm Events | Storm event counts and property damage | https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/ |
| CDC/ATSDR Social Vulnerability Index | Overall and theme-level social vulnerability | https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html |
| CalEnviroScreen | Environmental burden context | https://oehha.ca.gov/calenviroscreen |
| FEMA National Risk Index | Risk, social vulnerability, and resilience context | https://hazards.fema.gov/nri/data-resources |
| Census population estimates | County population denominators | https://www.census.gov/programs-surveys/popest/data/data-sets.html |

## Column groups

Core identifiers:

- `county`
- `month`
- `year`
- `month_num`

Clean-energy adoption and conversion:

- `dgstats_interconnected_proxy_count`
- `dgstats_app_received_count`
- `conversion_gap_per_100k_pop`
- `conversion_rate`
- `high_conversion_gap_flag`

Outage and hazard burden:

- `customers_out`
- `hazard_burden`
- `noaa_property_damage_usd`
- `noaa_event_count`
- `calfire_fire_count`
- `psps_event_count`

Vulnerability, risk, and context:

- `county_resilience_class`
- `svi_rpl_themes`
- `svi_rpl_theme1`
- `svi_rpl_theme2`
- `svi_rpl_theme3`
- `svi_rpl_theme4`
- `fema_risk_score`
- `fema_sovi_score`
- `fema_resl_score`
- `ces_ciscorep_pop_weighted`
- `census_population_estimate`

## Preprocessing summary

The curation workflow standardizes each source to a county-month panel before merging:

1. Convert source-specific dates to calendar months.
2. Map tract, ZIP, county, circuit, or event-level records to California counties when needed.
3. Aggregate dynamic event and adoption variables to county-month counts or totals.
4. Align vulnerability, risk, environmental, and population context to the county-month index.
5. Build a complete county-month grid for all 58 counties and 96 months.
6. Fill absent dynamic event counts with zero when no event or record was observed for that county-month.
7. Carry contextual county-level values across applicable months.
8. Preserve undefined conversion rates as missing when there were no applications in the denominator.
9. Export the final public modeling panel and validate row uniqueness, column coverage, county count, and month count.

The script `src/build_resilience_panel.py` documents the released preprocessing logic and can rebuild the public panel format from an assembled input panel with either the current released column names or the earlier internal column names.

Example:

```bash
python src/build_resilience_panel.py --input data/resilience_planning_dataset.csv --output data/resilience_planning_dataset_checked.csv
```

## Missing data and scaling

No feature scaling is applied in the released CSV. Values are stored in interpretable source units or clearly named derived units, such as counts, dollars, percentiles, scores, and rates per 100,000 residents.

Most released columns are complete after county-month alignment. `conversion_rate` may be missing when the rate is not defined, usually because the county-month has no application denominator. Those missing values are intentionally preserved instead of being forced to zero.

## Analysis scripts

### Paper descriptive analysis

`src/paper_descriptive_analysis.py` produces the main dataset-paper summary tables and figures. It focuses on dataset coverage, temporal variation, county heterogeneity, vulnerability-burden relationships, and descriptive lag screening.

Example:

```bash
python src/paper_descriptive_analysis.py --input data/resilience_planning_dataset.csv --outdir paper_outputs
```

Main outputs include:

- `dataset_summary.csv`
- `missing_values.csv`
- `yearly_totals.csv`
- `county_summary.csv`
- `state_transition_check.csv`
- `svi_outage_correlation.csv`
- `lagged_screening_all_lags.csv`
- `lagged_screening_best_lags.csv`
- paper figure PNGs

### Resilience trend analysis

`src/resilience_trend_analysis.py` contains additional trend analyses over the panel, including adoption divergence by vulnerability tier, conversion gaps, hazard-outage co-movement, outage recovery by FEMA resilience, and concentration of repeated conversion-gap flags.

Example:

```bash
python src/resilience_trend_analysis.py --input data/resilience_planning_dataset.csv --outdir trend_outputs
```

## Reproducibility checks

Before using the dataset, the following checks are recommended:

```bash
python src/build_resilience_panel.py --input data/resilience_planning_dataset.csv --output data/resilience_planning_dataset_checked.csv
python src/paper_descriptive_analysis.py --input data/resilience_planning_dataset.csv --outdir paper_outputs
python src/resilience_trend_analysis.py --input data/resilience_planning_dataset.csv --outdir trend_outputs
```

The expected main panel has 5,568 rows, 58 counties, and 96 months.

## Notes for interpretation

The panel is descriptive and harmonized for county-month analysis. It should not be interpreted as a causal estimate of the effect of solar interconnections on outages or hazard burden without additional modeling assumptions.

`customers_out` is an outage burden measure aggregated from reporting data. It should be treated as cumulative customer-outage burden rather than a count of unique people affected.

Static and slowly updated context variables, such as SVI, FEMA risk, CalEnviroScreen, and population estimates, are useful for priority and equity context. Dynamic variables, such as interconnections, applications, outages, storm events, wildfire events, PSPS events, and property damage, are better suited for monthly trend analysis.
