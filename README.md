# Delft noise monitoring

This repository provides analyses for a noise monitoring study in Delft. Specifically, Jupyter Notebooks analyze the following (acoustic) aspects: loudness, sound-source presence, intermittency, survey associations and sensitivity analyses.

## License

The analysis code and notebooks are licensed under the [MIT License](LICENSE). The study data are licensed separately; see their 4TU.ResearchData record.

## Data

The study data, excluding survey responses, are hosted separately: [10.4121/b4f6c7be-4537-4cc2-97fd-1a4c446dd900](https://doi.org/10.4121/b4f6c7be-4537-4cc2-97fd-1a4c446dd900).

Extract `lake.zip`, `marts.zip`, `derived.zip` and `ref.zip` into `data/`, preserving their folder structure, and place `manifest.json` directly in `data/`. See the dataset README for extraction details. The data was originally collected using our low-cost source-aware sound sensors.

Sensor coordinates in the data are rounded to five decimal places.

### Survey access

Survey responses and respondent-to-sensor linkage are not public. The corresponding Notebook will therefore fail without the data present.

## Setup

Create or activate the `delft-sensors` environment from `environment.yml` (Python 3.12). Package versions are pinned in the environment file. Run commands and notebooks from this repository root:

```sh
python -m scripts.prepare_analysis build --data-root data --destination analysis.duckdb --verify-data
```

This creates a new local DuckDB database without requiring the original database. Existing destinations are refused. `--verify-data` checks all lake/mart checksums; derived/reference checksums and all file sizes/file sets are always checked. Moving the data requires rebuilding the database because views use absolute paths.

Optional environment variables: `DELFT_DATA_ROOT`, `DELFT_DATABASE`, `DELFT_OUTPUT_ROOT`, `DELFT_THREADS`, `DELFT_MEMORY_LIMIT`. Set them before opening notebook connections. `DELFT_OUTPUT_ROOT` is the parent of generated `out/` and `figures/` directories. For constrained-memory runs, `DELFT_THREADS=1` and `DELFT_MEMORY_LIMIT=4GB` are useful starting settings.

## Workflows


| Notebook | Purpose                                                         |
| -------- | --------------------------------------------------------------- |
| 1        | Loudness, sampling, intermittency and associated sources        |
| 2        | Source presence, spatial maps, diurnal curves and co-occurrence |
| 4        | Broad overview and day/night graphics                           |
| 5        | Survey associations (data not publicly available)               |
| 8        | Location sensitivity to detection thresholds                    |
| 10       | Common-observation-period sensitivity                           |

Run notebook cells in order. Notebooks 1, 2, 4 and 8 read the prepared database independently. For notebook 10 the required script workflow is as follows:

```sh
python scripts/regenerate_diurnal_plots.py
python scripts/temporal_presence_sensitivity.py
python scripts/temporal_coverage.py
# Then run analysis_10_observation_period_sensitivity.ipynb
```

Map backgrounds are fetched from OpenStreetMap and require internet access.
