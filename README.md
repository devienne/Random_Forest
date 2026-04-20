# Subduction Zone Fault Slip from Seismic Noise and GPS Data

Reproduction and refactor of Rouet-Leduc et al. (2019) — *Continuous chatter of
the Cascadia subduction zone revealed by machine learning* (Nature Geoscience).

A **Random Forest regressor** is trained to predict GPS displacement rate (a proxy
for tectonic fault slip) from statistical features of ambient seismic noise recorded
near Vancouver Island. The underlying hypothesis is that seismic noise statistics
are a fingerprint of fault slip rate, making seismic data a continuous proxy for
geodetic measurements.

> This work was presented during the course *Big Data to Earth Scientists* at
> Peking University, 2020.
>
> **Cite as:** Devienne, J. A. P. M. "Subduction zone fault slip from seismic noise
> and GPS data." arXiv. https://doi.org/10.48550/arXiv.2304.08316

---

## Pipeline overview

```
IRIS FDSN API                    UNAVCO GPS API
     │                                 │
     ▼                                 ▼
src/data/download_seismic.py    src/data/download_gps.py
     │                                 │
     │  data/raw/seismic/*.mseed        │  data/raw/gps/*.csv
     ▼                                 ▼
src/features/seismic_features.py  src/features/gps_features.py
  · detrend + demean                · total horizontal displacement
  · bandpass (8–13 Hz, 5 bands)     · √(δN² + δE²)  [bug fix]
  · 4 stats per band                · rolling linear regression
  · 60-day rolling mean             · slope = displacement rate
     │                                 │
     └──────────────┬──────────────────┘
                    ▼
           data/processed/{station}_merged.csv
                    │
                    ▼
         src/pipelines/train.py
           · chronological 80/20 split  [leakage fix]
           · RandomForestRegressor (n_estimators=100)
           · R², MAE, RMSE on test set
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
  artifacts/models/     artifacts/figures/
  {station}_rf.joblib   {station}_prediction.png
  {station}_metrics.json {station}_feature_importances.png
```

---

## Repository structure

```
.
├── configs/
│   └── default.yaml          # all tuneable parameters
├── data/
│   ├── raw/                  # downloaded miniSEED and GPS CSV files (gitignored)
│   ├── interim/              # station lists
│   └── processed/            # merged per-station feature+target CSVs
├── src/
│   ├── data/
│   │   ├── download_seismic.py
│   │   └── download_gps.py
│   ├── features/
│   │   ├── seismic_features.py
│   │   └── gps_features.py
│   ├── models/
│   │   └── random_forest.py
│   ├── pipelines/
│   │   └── train.py          # entry point
│   └── utils/
│       ├── config.py         # Pydantic config schema + loader
│       └── plotting.py
├── tests/
│   ├── test_config.py
│   ├── test_features.py
│   └── test_models.py
├── artifacts/
│   ├── figures/              # output plots
│   └── models/               # saved .joblib models and metrics JSON
├── docs/
│   ├── report/               # LaTeX source and PDF
│   └── presentation/         # original PPTX
├── .env.example
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

---

## Installation

Python 3.10+ is required.

```bash
# 1. Clone the repository
git clone https://github.com/devienne/cascadia-fault-slip.git
cd cascadia-fault-slip

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install the package with development dependencies
pip install -e ".[dev]"

# 4. Copy the environment template
cp .env.example .env
# Edit .env if you need to redirect the data directory or override API URLs
```

---

## Usage

### Step 1 — Download raw data

```bash
# Download seismic station list and miniSEED files
python -c "
from src.utils.config import load_config
from src.data.download_seismic import get_station_list, download_seismic_data
cfg = load_config()
stations = get_station_list(cfg.seismic, cfg.paths.interim_dir)
download_seismic_data(stations, cfg.seismic, cfg.paths.raw_seismic_dir)
"

# Download GPS position files
python -c "
from src.utils.config import load_config
from src.data.download_gps import get_gps_station_list, download_gps_data
cfg = load_config()
stations = get_gps_station_list(cfg.gps, cfg.paths.interim_dir)
download_gps_data(stations, cfg.gps, cfg.paths.raw_gps_dir)
"
```

> **Note:** The full download covers ~10 years of daily data and may take several
> hours depending on network speed. Already-downloaded files are skipped on re-runs.

### Step 2 — Build the feature matrix

```bash
# (prepare_data pipeline — coming in Phase 2)
# This step reads raw data and writes data/processed/{station}_merged.csv
python -m src.pipelines.prepare_data
```

### Step 3 — Train the model

```bash
# Train all GPS stations found in data/processed/
python -m src.pipelines.train

# Train a single station
python -m src.pipelines.train --station ALBH

# Use a custom config
python -m src.pipelines.train --config configs/custom.yaml
```

Results are written to `artifacts/models/` and `artifacts/figures/`.

### Step 4 — Run tests

```bash
pytest
# With coverage report
pytest --cov=src --cov-report=term-missing
```

---

## Configuration

All scientific and ML parameters live in [`configs/default.yaml`](configs/default.yaml).
The key parameters are:

| Section | Parameter | Default | Description |
|---|---|---|---|
| `seismic` | `center_lat/lon` | 48.9, -123.9 | Station search centre |
| `seismic` | `max_radius_deg` | 0.6 | Station search radius |
| `seismic` | `freq_bands` | 8–13 Hz (5 bands) | Bandpass filter ranges |
| `seismic` | `rolling_window_days` | 60 | Rolling mean window |
| `gps` | `rolling_window_days` | 60 | Rolling regression window |
| `model` | `n_estimators` | 100 | Number of trees |
| `model` | `train_size` | 0.8 | Chronological train fraction |

---

## Known methodological issues

These issues were identified during the refactor and documented here for scientific
transparency. They explain why the original study's results appeared "too good."

### 1. Data leakage via random train/test split (fixed)

The original scripts used `sklearn.train_test_split` with a random shuffle on a
time-series with 60-day rolling features. Because consecutive rows share ~59/60
days of data, a random split allows the model to see near-duplicate rows of future
data during training.

**Fix:** `chronological_split()` in `src/models/random_forest.py` uses the first
80% of the time-ordered series for training and the remaining 20% for testing.

### 2. GPS displacement: sum instead of magnitude (fixed)

The original `construct_the_matrix_3.py` computed total horizontal displacement as
`δN + δE` (arithmetic sum). The correct quantity is the Euclidean norm
`√(δN² + δE²)`.

**Fix:** `total_horizontal_displacement()` in `src/features/gps_features.py`.

### 3. Rolling regression returned intercept, not slope (fixed)

`rf_1.py` used `regression[:,1]` from `np.polyfit`, which is the **intercept**.
The displacement rate is the **slope** (`regression[:,0]`).

**Fix:** `compute_displacement_rate()` in `src/features/gps_features.py` returns
`coeffs[0]`.

### 4. Channel mismatch between download and feature extraction (fixed)

`download_seismic_3.py` requested the `HHE` channel but `construct_the_matrix_2.py`
filtered for `BHE`, resulting in zero rows being processed.

**Fix:** both modules now read `cfg.seismic.channel` (default `HHE`).

### 5. n_estimators=1 (fixed)

The original used a single decision tree. `n_estimators` is now set to 100 and
exposed as a config parameter.

---

## Data sources

| Source | Description |
|---|---|
| [IRIS FDSN](http://service.iris.edu/fdsnws/) | Seismic waveform data (HHE channel, miniSEED) |
| [UNAVCO](https://web-services.unavco.org/) | GPS daily position data (NAM14 reference frame, USGS-preprocessed) |

---

## References

1. Rogers, G. & Dragert, H. Episodic tremor and slip on the Cascadia subduction
   zone. *Science* 300, 1942–1943 (2003).
2. Rouet-Leduc, B., Hulbert, C. & Johnson, P. A. Continuous chatter of the Cascadia
   subduction zone revealed by machine learning. *Nature Geosci* 12, 75–79 (2019).
3. Murray, J. R. & Svarc, J. GPS data collection, processing, and analysis conducted
   by the USGS Earthquake Hazards Program. *Seismol. Res. Lett.* 88, 916–925 (2017).
