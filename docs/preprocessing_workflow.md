# Preprocessing workflow

This note describes the preprocessing decisions behind `data/resilience_planning_dataset.csv`.

## Target panel

The target unit is a California county-month. The released dataset has one row for each of 58 counties and each month from January 2017 through December 2024.

## Source alignment

Each source was first converted into a county-month file before being merged.

| Source family | Original resolution | County-month treatment |
|---|---|---|
| DGStats interconnection records | Project/application records with location and dates | Count applications and interconnected projects by county and month |
| Power outage records | Time-stamped outage observations | Aggregate customer-outage burden by county and month |
| PSPS reports | Event, circuit, tract, or report-level records depending on year and file | Map reported exposure to counties and count PSPS event months |
| NOAA Storm Events | Event-level records with county and date fields | Count storm events and sum property damage by county and month |
| CAL FIRE records | Incident or perimeter records | Count fire events by county and month after geographic alignment |
| CDC/ATSDR SVI | County or tract-level vintage files | Use county-level overall and theme percentile values as vulnerability context |
| CalEnviroScreen | Tract-level environmental burden scores | Population-weight tract values to county context values |
| FEMA National Risk Index | County-level risk and resilience scores | Align county risk, social vulnerability, and resilience scores to all panel months |
| Census population estimates | County-year population estimates | Align annual county population estimates to months in the same year |

## Cleaning choices

- County names are standardized before merging.
- All date fields are converted to calendar months.
- Dynamic count variables are set to zero when no record is observed for a county-month.
- County context variables are carried across applicable months because they are not monthly event observations.
- `conversion_gap_per_100k_pop` is computed as the application-interconnection gap normalized by population.
- `conversion_rate` is computed only when applications are present. It is left missing when the denominator is zero or unavailable.
- The final panel is validated for duplicate county-month rows, expected county count, expected month count, and required output columns.

## Imputation and scaling

The released panel does not apply statistical imputation or feature scaling.

The only fill decisions are structural:

- event/count fields with no observed event are filled as zero;
- static or slow-moving county context values are aligned across months;
- undefined conversion rates remain missing.

This keeps the released dataset close to interpretable source units and lets downstream users choose their own modeling-specific imputation and scaling.

## Rebuild check

Run:

```bash
python src/build_resilience_panel.py --input data/resilience_planning_dataset.csv --output data/resilience_planning_dataset_checked.csv
```

The expected output is 5,568 rows, 26 columns, 58 counties, and 96 months.
