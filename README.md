# Tornado Causality Research

Finds which large-scale weather and climate variables drive tornado activity and tornado losses in the central US, using **PCMCI+** (causal discovery for time series, via the Tigramite library) on monthly data from 1982 to 2024.

## Pipeline at a glance

```
Data/ (raw files)
   |
   |  CombineHourly.py      hourly ERA5 NetCDF  ->  two CSVs      (optional)
   v
   |  PreProcessData.py     raw files -> clean monthly table
   v
output/preprocessed_*.csv
   |
   |  run-PCMCI.py          causal discovery
   v
output/links_*.csv + output/graph_*.png
```

## Scripts

### `CombineHourly.py` (optional, only for the hourly ERA5 data)
Merges the per-year hourly ERA5 NetCDF files into two CSVs in `Data/`:
- `era5_pressure_winds.csv`: 500 hPa winds (`u500`, `v500`)
- `era5_surface.csv`: `cape`, `u10`, `v10` (10 m winds)

It only averages each time step over the study box (weighted by cos(latitude)); nothing else is computed. Each CSV has a `.meta.json` recording the box used.

```
python CombineHourly.py --box 43 26 -106 -94
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
python PreProcessData.py --outcome all --extra soil precip --hourly --box 43 26 -106 -94
```
Other options: `--states` (SPC and SHELDUS states), `--era5-file`, `--start`, `--end`, `--hourly-vars`, `--cape-thr`, `--shear-thr`.

### `run-PCMCI.py`
Reads `output/preprocessed_<outcome>.csv`, runs PCMCI+ with a partial-correlation test, and writes a table of significant links and a graph. It does no data cleaning.

```
python run-PCMCI.py --outcome all --drop t2m
python run-PCMCI.py --outcome spc --tau-max 6 --alpha 0.05
```
- `--tau-max`: longest lag tested, in months (default 3)
- `--alpha`: significance level (default 0.01)
- `--drop`: variables to leave out of a run

## Output folder

| File | What it is |
|---|---|
| `merged_monthly_<outcome>.csv` | Raw monthly values of every variable (before anomalies) |
| `preprocessed_<outcome>.csv` | Standardized, detrended anomalies; this is the input to PCMCI |
| `preprocessed_<outcome>.meta.json` | States, box and dates used (read for graph captions) |
| `qa_series_<outcome>.png` | Time-series plot of every variable, for checking the data |
| `links_<outcome>.csv` | Significant causal links found by PCMCI+ |
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

Only links into the outcome with lag of at least 1 month can be used for forecasting. Same-month links describe the physics but cannot predict ahead.

## Data (`Data/`)

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