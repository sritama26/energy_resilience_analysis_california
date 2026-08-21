# Energy Resilience Analysis — California

Analysis of a county-month panel dataset covering solar interconnection adoption,
power outages, hazard events (PSPS, wildfire, storms), and social/community
vulnerability indices across California counties, used to study how energy
resilience and clean-energy adoption diverge across vulnerable communities.

## Project structure

```
.
├── data/
│   ├── resilience_planning_dataset.csv               # county-month panel data
│   └── resilience_planning_dataset_data_dictionary.xlsx  # column definitions
└── src/
    └── data_analyis.py   # analysis script
```

## Data

`data/resilience_planning_dataset.csv` is a county-month panel with columns including:

See `resilience_planning_dataset_data_dictionary.xlsx` for full column definitions.

## Analysis script

`src/data_analyis.py` runs the below analyses over the panel:

1. Solar adoption divergence by SVI** — compares solar interconnection rates
   (per 100k residents) across social-vulnerability terciles and tracks whether
   the gap between low- and high-vulnerability counties is widening.
2. Conversion falling behind (hazard burden / CES) — checks whether
   high-hazard-burden or high-CES counties have lower/worsening solar
   application conversion rates than low-burden counties.
3. Outage recovery by FEMA resilience — builds event-time outage curves
   around hazard events and compares peak outage impact and recovery time
   between low- and high-FEMA-resilience counties.
4. Conversion-gap flag concentration — measures whether counties flagged
   for a high conversion gap are persistently the same ones, increasingly
   concentrated (Gini coefficient), and disproportionately high-SVI.

Each analysis writes PNG figures and CSV tables to the output directory, and a
combined `summary.json` is written at the end of the run.

### Usage

```bash
python src/data_analyis.py --input data/resilience_planning_dataset.csv --outdir trend_outputs
```

- `--input` (required): path to the panel CSV.
- `--outdir` (default `trend_outputs`): directory for figures, tables, and `summary.json`.

### Requirements

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`

```bash
pip install pandas numpy matplotlib
```
