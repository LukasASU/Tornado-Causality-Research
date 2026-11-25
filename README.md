# Tornado Causality: An In-depth Analysis of Potential Variables Causing Tornados

CSE 472 (ASU) course project by Lukas Cereghino and Efaz Arian.

This project asks two questions about tornadoes in the central US (TX, OK, KS, NE):

1. **Which large-scale weather and climate variables drive tornado activity and tornado losses?** We answer this with **PCMCI+** (causal discovery for time series, via the Tigramite library) on monthly data from 1982 to 2024.
2. **Can the discovered causes forecast tornado outcomes better than a simple baseline, or better than using every variable?** We answer this with forecasting models that are scored only on years they never saw.

## Repository layout

```
Data/                      raw input data (not in the repo, see "Getting the data")

PCMCI/                     Part 1: data preparation and causal discovery
    CombineHourly.py       hourly ERA5 -> two CSVs (optional)
    PreProcessData.py      raw files -> clean monthly table
    run-PCMCI.py           causal discovery (PCMCI+)
    download_soil.py       re-download ERA5 soil moisture / precipitation
    output/                tables and graphs written by the scripts above

Models/                    Part 2: forecasting models (reads PCMCI/output)
    forecast_common.py     shared modeling code
    plots.py               the two charts
    model_tornado_days.py  one script per outcome variable
    model_tornado_count.py
    model_outbreak_days.py
    model_sheldus_loss.py
    results/               tables and charts written by the model scripts
```

`Data/`, `PCMCI/` and `Models/` all sit at the same level. The model scripts find the PCMCI results by looking in `../PCMCI/output`, and the PCMCI scripts read raw files from `../Data`, so keep the three folders side by side.

## Getting the data

**The data is not stored in this repository.** The files used for the project are in a shared Google Drive folder. To get access, email **lcereghi@asu.edu** and request the folder.

Once you have it:
- Put the raw files in the top-level `Data/` folder (next to `PCMCI/` and `Models/`).
- If you only want to run the models, you can skip Part 1 and put the PCMCI output files (`preprocessed_*.csv`, `links_*.csv` and the `.meta.json` files) in `PCMCI/output/`.
- You can also rebuild everything from the raw data by running the scripts in order (see below).

## Pipeline at a glance

```
Data/ (raw files)
   |
   |  CombineHourly.py      hourly ERA5 NetCDF  ->  two CSVs      (optional)
   v
   |  PreProcessData.py     raw files -> clean monthly table
   v
PCMCI/output/preprocessed_*.csv
   |
   |  run-PCMCI.py          causal discovery
   v
PCMCI/output/links_*.csv + graph_*.png
   |
   |  Models/model_<outcome>.py    forecast test of the discovered causes
   v
Models/results/<outcome>/regression/
```

---

# Part 1: PCMCI (`PCMCI/`)

Run these from inside the `PCMCI/` folder. Because the raw data lives one level up in `Data/`, the commands below pass `../Data` explicitly.

### `CombineHourly.py` (optional, only for the hourly ERA5 data)
Merges the per-year hourly ERA5 NetCDF files into two CSVs in `Data/`:
- `era5_pressure_winds.csv`: 500 hPa winds (`u500`, `v500`)
- `era5_surface.csv`: `cape`, `u10`, `v10` (10 m winds)

It only averages each time step over the study box (weighted by cos(latitude)); nothing else is computed. Reads `../Data/era5_round2/` and writes to `../Data/`. Each CSV has a `.meta.json` recording the box used.

```
python CombineHourly.py --in-dir ../Data/era5_round2 --out-dir ../Data --box 43 26 -106 -94
```
Box order is North, South, West, East. Use the same box everywhere.

### `PreProcessData.py`
Everything that happens before the causal algorithm:
1. Loads each source (SPC tornadoes, SHELDUS losses, ENSO, Gulf SST, ERA5 monthly files, hourly CSVs).
2. Puts them on a common monthly index and merges them.
3. Removes the seasonal cycle and the linear trend (standardized anomalies).
4. Saves the results, plus quick-look plots.

Outcome variables (`--outcome`):

| Option | Variable | Meaning |
|---|---|---|
| `spc` | `tornado_days` | log(1 + days per month with an EF1+ tornado) |
| `count` | `tornado_count` | log(1 + EF1+ tornadoes per month) |
| `outbreaks` | `outbreak_days` | log(1 + days with 6+ EF1+ tornadoes; set with `--outbreak-min`) |
| `sheldus` | `sheldus_loss` | log(1 + inflation-adjusted property + crop losses from tornadoes) |
| `both` / `all` | | `spc` + `sheldus` / all four |

```
python PreProcessData.py --data-dir ../Data --outcome all --extra soil precip --hourly --box 43 26 -106 -94
```
Other options: `--data-dir` (where the raw files are; default is `Data` inside the current folder, so use `../Data`), `--states` (SPC and SHELDUS states), `--era5-file`, `--start`, `--end`, `--hourly-vars`, `--cape-thr`, `--shear-thr`.

### `run-PCMCI.py`
Reads `output/preprocessed_<outcome>.csv`, runs PCMCI+ with a partial-correlation test, and writes a table of significant links and a graph. It does no data cleaning.

```
python run-PCMCI.py --outcome all --drop t2m
python run-PCMCI.py --outcome spc --tau-max 6 --alpha 0.05
```
- `--tau-max`: longest lag tested, in months (default 3)
- `--alpha`: significance level (default 0.01)
- `--drop`: variables to leave out of a run

### `download_soil.py`
Re-downloads ERA5 monthly soil moisture or total precipitation for 1982 to 2024 over the correct area. Needs a Copernicus account (see Setup). It saves into a folder called `Data` in the current directory, so run it from the repository root:

```
python PCMCI/download_soil.py soil
python PCMCI/download_soil.py precip
```

## PCMCI output folder (`PCMCI/output/`)

| File | What it is |
|---|---|
| `merged_monthly_<outcome>.csv` | Raw monthly values of every variable (before anomalies) |
| `preprocessed_<outcome>.csv` | Standardized, detrended anomalies; this is the input to PCMCI and to the models |
| `preprocessed_<outcome>.meta.json` | States, box and dates used (read for graph captions) |
| `qa_series_<outcome>.png` | Time-series plot of every variable, for checking the data |
| `links_<outcome>.csv` | Significant causal links found by PCMCI+; the models read this to pick their inputs |
| `graph_<outcome>.png` | The causal graph with a legend |

`<outcome>` is one of `tornado_days`, `tornado_count`, `outbreak_days`, `sheldus_loss`.

### Reading `links_<outcome>.csv`
Columns: `source, target, lag, mark, strength, p_value`.
- **lag**: months by which the source leads the target (0 = same month).
- **mark**: `-->` directed (source causes target); `o-o` same-month link, direction undetermined; an `x` on an end means conflicting evidence about direction.
- **strength**: partial correlation (MCI value) between -1 and 1. Negative means the two move in opposite directions. Weak values (about 0.2 or below) explain little variance.
- **p_value**: significance of the link.

### Reading `graph_<outcome>.png`
- The outcome variable is highlighted (orange glow, bold name).
- Arrowhead = directed link; circle (o) = direction not determined; x = conflicting evidence on direction.
- Number on a link = lag in months; no number = same month.
- Red = positive, blue = negative; paler and thinner = weaker.
- Node color = how strongly a variable persists from month to month.
- Text in brackets under a name = the data source (ERA5, NOAA, SPC, etc.).

Only links into the outcome with a lag of at least 1 month can be used for forecasting. Same-month links describe the physics but cannot predict ahead.

---

# Part 2: Forecast models (`Models/`)

Run these from inside the `Models/` folder, after Part 1 has produced `PCMCI/output/`.

Each script tests one outcome and asks: **do the causes PCMCI found forecast the outcome better than simply predicting an average month?**

```
python model_tornado_days.py
python model_tornado_count.py
python model_outbreak_days.py
python model_sheldus_loss.py
```

Optional flags (same for every script):

| Flag | What it does |
|---|---|
| `--forest` | also fit a random forest, to see if a non-linear model helps |
| `--drop t2m` | leave a variable out of the models |
| `--reselect` | re-run PCMCI+ on each training window so the test years never influence the choice of parents (slower, cleaner test; needs tigramite). Use with `--tau-max` and `--alpha` |
| `--first-test-year`, `--block-years` | change the time split (defaults: 2002 and 5-year blocks) |
| `--all-max-lag` | longest lag used by the all-variables model (default 3) |
| `--boot`, `--seed` | bootstrap resamples (default 1000) and random seed |
| `--pcmci-dir`, `--results-dir` | change where inputs are read and results are written |

## The models

All forecast models are ridge regression (linear regression with a penalty that keeps the coefficients small; the penalty is chosen by leave-one-out error inside the training years).

| Model | What it uses |
|---|---|
| Climatology | Always predicts the training-period average. This is the "no skill" baseline. |
| Last month's value | Copies the previous month's outcome. |
| Own past only | The outcome's own previous months. |
| Causal parents | Only the lagged variables PCMCI+ linked directly to the outcome. |
| Causal parents + own past | The causal parents plus the outcome's previous month. |
| All features + own past | Every variable at every lag from 1 to the longest parent lag (at least 3), plus the outcome's previous month. |
| Same-month neighbors | Variables linked to the outcome within the same month. **This is a nowcast, not a forecast**, because those values are not known in advance. It is a reference for how much the current environment explains. |

Outcomes with no lagged parents (`outbreak_days`, `sheldus_loss`) cannot get the causal-parents models; the other models are still scored.

## How the test works
- **Expanding-window split:** train on the early years, test on the next 5-year block, then move forward. The first test block is 2002 to 2006 and the last ends in 2024 (276 test months). A model never sees the months it is scored on.
- **Skill** = 1 - (model squared error / climatology squared error). 0 means no better than an average month; negative means worse; 1 would be perfect.
- **95% ranges** come from a moving-block bootstrap (1000 resamples), which keeps neighboring months together because they are not independent.
- Scores are given for all months and for March to June (peak tornado season).
- **Caveat:** by default the causal parents come from the PCMCI+ run on all years, including the test years, which can slightly favor the causal model. `--reselect` removes this.

## Models output folder (`Models/results/<outcome>/regression/`)

| File | What it is |
|---|---|
| `skill_all_months.csv` | Skill (with 95% range) of every model over all test months |
| `skill_spring.csv` | Same, for March to June only |
| `predictions.csv` | Observed value and every model's forecast, month by month |
| `parents_by_fold.csv` | Which causal parents were used in each training window |
| `figures/skill_summary.png` | Skill of every model with 95% ranges |
| `figures/observed_vs_predicted.png` | Observed anomaly (grey) vs each model's forecast, one panel per model, March to June shaded |

## Main finding
For tornado days and tornado count, the lagged causes found by PCMCI+ forecast no better than climatology (all 95% ranges include zero), and using every variable does not help. The same-month nowcast reaches a skill of about 0.2, so the variables are clearly related to tornado activity within a month, but the lagged links carry almost no forecasting information. `outbreak_days` and `sheldus_loss` had no lagged parents.

---

## Data sources (`Data/`)

| File | Source |
|---|---|
| `1950-2025_SPC.csv` | NOAA Storm Prediction Center tornado records |
| `direct_loss_aggregated_output_*.csv` | SHELDUS county x year x month tornado losses |
| `NOAA_SST.txt` | NOAA monthly Nino indices (ENSO, Nino 3.4 anomaly) |
| `ERDDAP.csv` | ERSST v5 Gulf of Mexico sea surface temperature anomaly |
| `ERA5_core_monthly.nc` | ERA5 monthly CAPE, 2 m dewpoint, 2 m temperature |
| `ERA5_Volumetric_soil_layer_1.nc` | ERA5 monthly soil moisture |
| `ERA5_total_precipitation.nc` | ERA5 monthly precipitation |
| `era5_round2/` | ERA5 hourly files (500 hPa winds, CAPE, 10 m winds), one per year |
| `era5_pressure_winds.csv`, `era5_surface.csv` | Created by `CombineHourly.py` |

ERA5 is a reanalysis: a weather model continuously corrected with real observations. It describes the large-scale environment well but cannot see individual storms.

## Setup

```
python -m pip install -r Requirements.txt
```
Needs Python 3 with pandas, numpy, xarray, netCDF4, matplotlib, networkx, tigramite (and joblib, numba, scipy, scikit-learn). ERA5 downloads also need `cdsapi` and a `.cdsapirc` file with your Copernicus key (never commit it).

## Caveats
- Hidden common causes (confounders) and weak links mean results are suggestive, not proof.
- Same-month link directions are unreliable.
- Many tests are run at once, so a few chance links are expected. Check that a link keeps its strength across different `--tau-max` values and time periods.
- SHELDUS losses are sparse and noisy; tornado counts and days are easier to model.