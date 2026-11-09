# ChartDensity-Bench

**Benchmarking MLLMs for Numerical Data Reconstruction under Visual Density**

A comprehensive benchmarking framework for evaluating Multimodal Large Language Models (MLLMs) on structured numerical data reconstruction from scientific chart figures under controlled visual density.

---

## 📖 Overview

**ChartDensity-Bench** is a benchmark for evaluating MLLMs on structured numerical data reconstruction from compound chart figures under controlled visual density. Built from charts paired with source-level ground-truth data, ChartDensity-Bench systematically varies the number of simultaneously presented charts (*n*), enabling controlled evaluation of density-induced degradation.

**multi-dimensional evaluation framework** covering:

- **Structural reliability** — does the model return the correct number of charts?
- **Reconstruction completeness** — how much of the ground-truth X-axis is recovered?
- **Parseability** — can the returned output be parsed into usable numerical data?
- **Numerical fidelity** — how accurately are the Y values reconstructed?

---

## 🔗 Chart Data Reconstruction Workflow

If you are interested in a practical, end-to-end workflow for reconstructing numerical data from chart figures, see our [SciFig2Data](https://github.com/issair/SciFig2Data).

---

## 🧩 Pipeline

The pipeline consists of four sequential scripts:

| # | Script | Purpose |
|---|--------|---------|
| 0 | `compound.py` | Builds the density-controlled dataset: packs *n* single-chart images into compound figures for *n* ∈ {3, 6, 9} and records the source-to-compound mapping. |
| 1 | `api.py` | Sends chart images to MLLM via API and saves extracted numerical data as Markdown tables. |
| 2 | `parse_calculate.py` | Parses the extracted Markdown tables, aligns them with source-level ground-truth CSVs, and computes reconstruction metrics per chart and per compound figure. |
| 3 | `summary.py` | Aggregates all evaluation JSON files into summary tables for cross-model, cross-density comparison. |
| 4 | `effect_analysis.py` | Performs chart-level mixed-effects regression and density-sensitivity analysis on the reconstruction metrics. |

---

## 📂 Repository Structure

```
.
├── compound.py   # Script 0: build density-controlled dataset
├── api.py         # Script 1: image → Markdown reconstruction
├── parse_calculate.py           # Script 2: Markdown → reconstruction metrics
├── summary.py        # Script 3: metrics → benchmark tables
├── effect_analysis.py            # Script 4: mixed-effects & density-slope analysis
├── requirements.txt
├── README.md
│
├── ICPR_filtered/                # Example dataset layout
│   ├── test/                     # density = 1 (single chart)
│   ├── test_compound3/           # density = 3
│   ├── test_compound6/           # density = 6
│   ├── test_compound9/           # density = 9
│   │   ├── *.jpg                 # compound chart images
│   │   ├── compound_info_9.json  # compound_id → source images mapping
│   │   └── results_<model>/      # reconstruction outputs
│   │       ├── *.md
│   │       ├── error.json
│   │       └── Processing_Report_*.md
│   ├── test_table/               # source-level ground-truth CSVs
│   └── results/
│       ├── test/                 # evaluation_results_<model>.json
│       └── test_compound9/       # evaluation_results_<model>.json
│
├── benchmark_figures/            # output tables
│   ├── density_benchmark.csv
│   └── density_degradation.csv
│
└── density_regression_results/   # output of effect_analysis.py
    ├── chart_level_analysis_dataset.csv
    ├── mae_mixed_model_summary.txt
    ├── mae_fixed_effects.csv
    ├── mae_density_slopes.csv
    ├── rmse_mixed_model_summary.txt
    ├── rmse_fixed_effects.csv
    └── rmse_density_slopes.csv
```

---

## ⚙️ Environment Setup

### 1. Requirements

Python **3.9+** is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. API Configuration

Script 1 uses an OpenAI-compatible client for example. Configure your API key and base URL in `api.py`.

And choose the model name in the `client.chat.completions.create(...)`.

---

## 🚀 Usage

### Step 0 — Build the density-controlled dataset

ChartDensity-Bench constructs compound figures by packing *n* single-chart images into one canvas, so that visual density is controlled by design. Before running any model inference, generate the compound datasets from a directory of single-chart images.

**Input layout**

```
ICPR_filtered/
├── test/                 # single-chart images: *.jpg, *.jpeg, *.png, *.bmp, ...
└── test_table/           # source-level ground-truth CSVs
```

**Run**

```bash
python compound.py
```

With the default configuration, the script:

- reads all single-chart images from `ICPR_filtered/test/`;
- generates compound images for group sizes `[3, 6, 9]`, i.e. one row per density level;
- writes results into `ICPR_filtered/test_compound3/`, `test_compound6/`, `test_compound9/`.
  
---

### Step 1 — Reconstruct numerical data from chart images

Process every image inside a directory:

```bash
python api.py ICPR_filtered/test_compound9
```

Multiple directories at once:

```bash
python api.py ICPR_filtered/test ICPR_filtered/test_compound3
```

Retry only failed images (reads `results_<model>/error.json`):

```bash
python api.py ICPR_filtered/test_compound9 --recover
```

**Outputs** (per directory, i.e., under `results_gemini/`):

- `<image_name>.md` — reconstructed numerical data as a Markdown table
- `error.json` — list of failed images with error reasons
- `Processing_Report_<dir>_<timestamp>.md` — success/failure summary

---

### Step 2 — Compute reconstruction metrics

Two modes are supported.

**Compound mode** (multiple charts per figure — density > 1):

```bash
python parse_calculate.py \
    --mode compound \
    --compound_info ICPR_filtered/test_compound9/compound_info_9.json \
    --results_dir ICPR_filtered/test_compound9/results_gemini \
    --table_dir ICPR_filtered/test_table \
    --output_json ICPR_filtered/results/test_compound9/evaluation_results_gemini.json
```

**Single-chart mode** (density = 1):

```bash
python parse_calculate.py \
    --mode single \
    --results_dir ICPR_filtered/test/results_gemini \
    --table_dir ICPR_filtered/test_table \
    --output_json ICPR_filtered/results/test/evaluation_results_gemini.json
```

Optional flags: `--top_n N` (limit number of compounds), `--image_ext .jpg`.

**Metrics computed per chart:**

| Metric | Meaning |
|--------|---------|
| `completeness_ratio` | Reconstruction completeness: extracted X-count ÷ ground-truth X-count |
| `point_coverage` | Fraction of ground-truth X values successfully matched by an extracted point |
| `mae` | Mean Absolute Error of Y values (normalized) — numerical fidelity |
| `rmse` | Root Mean Squared Error of Y values — numerical fidelity |

Per-compound **summary** (top of the JSON) additionally reports:

- `mismatch_compound_count` / `mismatch_compound_ratio` — compounds where the number of returned charts deviates from the expected count (structural reliability)
- `avg_completeness_ratio`, `avg_mae`, `avg_rmse`, `avg_point_coverage`
- `null_mae_rmse_source_image_count` / `..._ratio` — charts where no numerical points could be parsed (parseability)

---

### Step 3 — Aggregate benchmark results

After running Step 2 for every model and every density level, place the resulting `evaluation_results_<model>.json` files under `ICPR_filtered/results/<subset>/`, then run:

```bash
python summary.py
```

This produces:

- `evaluation_summary.csv` — flat table of all model × density combinations
- `benchmark_figures/density_benchmark.csv` — cross-model comparison under increasing visual density
- `benchmark_figures/density_degradation.csv` — density-induced degradation vs. the single-chart (density = 1) baseline

### Multi-Dimensional Evaluation Metrics

| Metric | Formula | Dimension |
|--------|---------|-----------|
| **RFV** (Response Format Validity) | `1 − mismatch_compound_ratio` | Structural reliability |
| **CAR** (Chart Alignment Rate) | `total_aligned_charts / expected_charts` | Structural reliability |
| **ACR** (Average Completeness Ratio) | `avg_completeness_ratio` | Reconstruction completeness |
| **EC** (Effective Completeness) | `CAR × ACR` | Structural reliability × Reconstruction completeness |
| **PR** (Parseability Rate) | `1 − null_mae_rmse_source_image_ratio` | Parseability |
| **Point Coverage** | `avg_point_coverage` | Reconstruction completeness |
| **MAE / RMSE** | (see Step 2) | Numerical fidelity |

---

### Step 4 — Chart-level mixed-effects analysis

Script 4 performs a chart-level statistical analysis of density-induced degradation. It reads all per-chart records from the evaluation JSON files, aligns them by `source_image`, and fits mixed-effects models that treat the **chart identity as a random effect**, allowing the same source chart to be compared across densities.

Run from the repository root:

```bash
python effect_analysis.py
```

The script auto-discovers evaluation JSONs under `ICPR_filtered/results/` (configurable via `BASE_DIRECTORY`) and writes to `density_regression_results/`:

**Models and density levels**

- `MODEL_ORDER = [<your_model_list>]`
- `DENSITY_ORDER = [1, 3, 6, 9]` — mapped from directory names via `DIRECTORY_TO_DENSITY`

**What it does**

1. **Collects chart-level records** — one row per (model, density, source_image), carrying the reconstruction metrics (`completeness_ratio`, `point_coverage`, `mae`, `rmse`, etc.).
2. **Validates chart structure** — reports duplicate `(model, density, chart_id)` observations and how many density levels each source chart appears at, since the mixed-effects model assumes chart identity can be tracked across densities.
3. **Fits mixed-effects regressions** on `MAE` and `RMSE` separately, using the formula:

   ```
   metric ~ C(model, Treatment(reference='kimi')) * C(density, Treatment(reference=1))
   ```

   with a random intercept per `chart_id`. Kimi at density 1 serves as the reference cell. Optimizers are tried in order `lbfgs → powell → cg` for numerical robustness.

4. **Estimates density-sensitivity slopes** — a per-model OLS slope of the metric against density, as an auxiliary, model-specific summary of the degradation trend.

**Outputs**

| File | Content |
|------|---------|
| `chart_level_analysis_dataset.csv` | Flat, chart-level dataset actually used by the analysis |
| `mae_mixed_model_summary.txt` | Full `statsmodels` summary of the MAE mixed-effects model |
| `mae_fixed_effects.csv` | Fixed-effect coefficients, SEs, z, p-values, 95% CIs |
| `mae_density_slopes.csv` | Per-model OLS slope of MAE vs. density |
| `rmse_mixed_model_summary.txt` | Same as above, for RMSE |
| `rmse_fixed_effects.csv` | Same as above, for RMSE |
| `rmse_density_slopes.csv` | Same as above, for RMSE |

All numeric outputs are rounded to 4 decimal places.

**Interpreting the fixed effects**

- The `C(density, ...)` terms quantify the average effect of density (relative to density 1, the reference) after accounting for chart identity and model.
- The `C(model, ...) * C(density, ...)` interaction terms quantify model-specific deviations from the average density effect — i.e., **how differently each model degrades as visual density increases**.
- Random intercepts per `chart_id` control for the fact that the same source chart is repeated across models and density levels, enabling the chart-level paired comparisons reported in the paper.

---

## 🗂 Data Format Specifications

### `compound_info_*.json`

Describes how a compound figure maps to its source charts:

```json
{
  "compounds": [
    {
      "compound_id": 1,
      "output_filename": "compound_001.png",
      "source_images": ["chart_a.jpg", "chart_b.jpg", "chart_c.jpg"]
    }
  ]
}
```

### Source-Level Ground-Truth CSVs

One CSV per source chart, named `<source_image_stem>.csv`, containing at minimum a numeric **X** column and a **Y** column. Optional columns (e.g., `Series`, `Label`) are used to split multi-series charts.

### Reconstructed Markdown

Markdown tables produced by the model follow the prompt specification:

```
| Series Name | X Value | Y Value |
|-------------|---------|---------|
| ...         | ...     | ...     |
```

The parser also supports multiple `## Chart N` sections within a single Markdown file, corresponding to compound figures with multiple subplots.

---

## 📜 License

### Code

The original code in this repository is released under the **MIT License**. See [LICENSE](LICENSE) for the full license text.

### Dataset and Benchmark Materials

The dataset and benchmark materials included in this repository are derived from the **ICPR 2022 CHART-Infographics** dataset:

> K. Davila, F. Xu, S. Ahmed, D. A. Mendoza, S. Setlur, and V. Govindaraju, "ICPR 2022: Challenge on Harvesting Raw Tables from Infographics (CHART-Infographics)," *2022 26th International Conference on Pattern Recognition (ICPR)*, Montreal, Canada, 2022, pp. 4995–5001.
> DOI: [10.1109/ICPR56361.2022.9956289](https://doi.org/10.1109/ICPR56361.2022.9956289)

The original third-party data and chart images are **not relicensed by the authors of ChartDensity-Bench**. They remain subject to the licenses and terms under which the original materials were released.

ChartDensity-Bench adds an original benchmark organization and evaluation framework, including density-controlled compound construction, source-to-compound mappings, benchmark metadata, evaluation scripts, and analysis tools. Rights in these original contributions are retained by the authors and are licensed as specified above.

Users are responsible for complying with the licenses and terms applicable to the original CHART-Infographics data and any underlying third-party materials.

If you use ChartDensity-Bench in academic research, please cite the ChartDensity-Bench paper as well as the original CHART-Infographics dataset/paper.

---

## 📚 Citation

```bibtex
@misc{wu2026chartdensitybenchbenchmarkingmllmsnumerical,
      title={ChartDensity-Bench: Benchmarking MLLMs for Numerical Data Reconstruction under Visual Density}, 
      author={Xinhe Wu and Yadong Jin},
      year={2026},
      eprint={2609.38781},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.38781}, 
}
```