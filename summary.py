import os
import json
from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 1. Collect evaluation results
# ============================================================
def collect_evaluation_results(base_dir='.'):
    all_results = []
    base_path = Path(base_dir)
    directories = [d for d in base_path.iterdir() if d.is_dir()]
    directories.insert(0, base_path)

    for directory in directories:
        json_files = list(directory.glob('evaluation_results_*.json'))
        for json_file in json_files:
            model_name = json_file.stem.replace('evaluation_results_', '')
            directory_name = directory.name if directory != base_path else '.'

            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'summary' in data:
                summary = data['summary']
                row = {
                    'model_name': model_name,
                    'directory': directory_name,
                    'total_processed_compounds': summary.get('total_processed_compounds'),
                    'mismatch_compound_count': summary.get('mismatch_compound_count'),
                    'mismatch_compound_ratio': summary.get('mismatch_compound_ratio'),
                    'total_aligned_charts': summary.get('total_aligned_charts'),
                    'avg_completeness_ratio': summary.get('avg_completeness_ratio'),
                    'avg_mae': summary.get('avg_mae'),
                    'avg_rmse': summary.get('avg_rmse'),
                    'avg_point_coverage': summary.get('avg_point_coverage'),
                    'null_mae_rmse_source_image_count': summary.get('null_mae_rmse_source_image_count'),
                    'null_mae_rmse_source_image_ratio': summary.get('null_mae_rmse_source_image_ratio'),
                }
                all_results.append(row)
                print(f"Processed: {json_file}")

    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.sort_values(['model_name', 'directory'])
    return df


# ============================================================
# 2. Configuration
# ============================================================
BASE_DIRECTORY = 'ICPR_filtered/results'
EVALUATION_SUMMARY_CSV = 'evaluation_summary.csv'
OUTPUT_DIR = Path('benchmark_figures')
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_ORDER = ["doubao", "gemini", "glm", "kimi", "qwen"]
DIRECTORY_TO_DENSITY = {
    "test": 1,
    "test_compound3": 3,
    "test_compound6": 6,
    "test_compound9": 9,
}


# ============================================================
# 3. Helper functions
# ============================================================
def fmt_4dec(x):
    if pd.isna(x):
        return np.nan
    return round(float(x), 4)


def weighted_average(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


# ============================================================
# 4. Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 4.1 Collect results and save evaluation_summary.csv
    # --------------------------------------------------------
    df = collect_evaluation_results(BASE_DIRECTORY)
    df.to_csv(EVALUATION_SUMMARY_CSV, index=False, encoding='utf-8-sig')
    print(f"\nResults saved to: {EVALUATION_SUMMARY_CSV}")
    print("\nData preview:")
    print(df.head())

    # --------------------------------------------------------
    # 4.2 Validate columns and map density
    # --------------------------------------------------------
    required_columns = [
        "model_name",
        "directory",
        "total_processed_compounds",
        "mismatch_compound_count",
        "mismatch_compound_ratio",
        "total_aligned_charts",
        "avg_completeness_ratio",
        "avg_mae",
        "avg_rmse",
        "avg_point_coverage",
        "null_mae_rmse_source_image_count",
        "null_mae_rmse_source_image_ratio",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing the following fields: {missing}")

    df["density"] = df["directory"].map(DIRECTORY_TO_DENSITY)
    if df["density"].isna().any():
        unknown = df.loc[df["density"].isna(), "directory"].unique()
        raise ValueError(f"Unknown directory found: {unknown}")

    # --------------------------------------------------------
    # 4.3 Derived metrics
    # --------------------------------------------------------
    # Expected charts: expected_charts = processed_compounds × density
    df["expected_charts"] = df["total_processed_compounds"] * df["density"]

    # Response Format Validity (RFV)
    df["RFV"] = 1 - df["mismatch_compound_ratio"]

    # Chart Alignment Rate (CAR)
    df["CAR"] = df["total_aligned_charts"] / df["expected_charts"]

    # Average Completeness Ratio (ACR)
    df["ACR"] = df["avg_completeness_ratio"]

    # Effective Completeness (EC)
    df["EC"] = df["CAR"] * df["ACR"]

    # Parseability Rate (PR)
    df["PR"] = 1 - df["null_mae_rmse_source_image_ratio"]

    # Point Coverage
    df["Point Coverage"] = df["avg_point_coverage"]

    # --------------------------------------------------------
    # 4.4 Table 2 — Performance under increasing chart density
    # --------------------------------------------------------
    density_rows = []
    for _, row in df.iterrows():
        density_rows.append({
            "Model": row["model_name"],
            "Subplots": int(row["density"]),
            "RFV": row["RFV"],
            "EC": row["EC"],
            "PR": row["PR"],
            "Point Coverage": row["Point Coverage"],
            "MAE": row["avg_mae"],
            "RMSE": row["avg_rmse"],
        })

    table2_raw = pd.DataFrame(density_rows)
    table2_raw["Model"] = pd.Categorical(
        table2_raw["Model"], categories=MODEL_ORDER, ordered=True
    )
    table2_raw = table2_raw.sort_values(["Model", "Subplots"])

    table2_save = table2_raw.copy()
    numeric_cols_2 = ["RFV", "EC", "PR", "Point Coverage", "MAE", "RMSE"]
    for col in numeric_cols_2:
        if col in table2_save.columns:
            table2_save[col] = table2_save[col].map(fmt_4dec)

    table2_save.to_csv(
        OUTPUT_DIR / "density_benchmark.csv",
        index=False,
        float_format="%.4f",
    )
    print(f"\nSaved: {OUTPUT_DIR / 'density_benchmark.csv'}")

    # --------------------------------------------------------
    # 4.5 Density Degradation Analysis — full results
    # --------------------------------------------------------
    density_data = df.copy()
    density_data["Model"] = density_data["model_name"].str.lower()
    density_data = density_data.sort_values(["Model", "density"])

    degradation_rows = []
    for model in MODEL_ORDER:
        sub = density_data[density_data["Model"] == model].copy()
        if sub.empty:
            continue

        baseline = sub[sub["density"] == 1]
        if baseline.empty:
            print(f"Warning: {model} does not have density=1 baseline.")
            continue

        baseline_mae = baseline["avg_mae"].iloc[0]
        baseline_rmse = baseline["avg_rmse"].iloc[0]

        for _, row in sub.iterrows():
            density = int(row["density"])
            mae = row["avg_mae"]
            rmse = row["avg_rmse"]

            delta_mae = mae - baseline_mae
            delta_rmse = rmse - baseline_rmse

            relative_delta_mae = delta_mae / baseline_mae if baseline_mae != 0 else np.nan
            relative_delta_rmse = delta_rmse / baseline_rmse if baseline_rmse != 0 else np.nan

            degradation_rows.append({
                "Model": model,
                "Subplots": density,
                "Baseline MAE (1 subplot)": baseline_mae,
                "MAE": mae,
                "ΔMAE": delta_mae,
                "Relative ΔMAE": relative_delta_mae,
                "Baseline RMSE (1 subplot)": baseline_rmse,
                "RMSE": rmse,
                "ΔRMSE": delta_rmse,
                "Relative ΔRMSE": relative_delta_rmse,
            })

    degradation = pd.DataFrame(degradation_rows)

    degradation_save = degradation.copy()
    numeric_cols_deg = [
        "Baseline MAE (1 subplot)", "MAE", "ΔMAE", "Relative ΔMAE",
        "Baseline RMSE (1 subplot)", "RMSE", "ΔRMSE", "Relative ΔRMSE",
    ]
    for col in numeric_cols_deg:
        if col in degradation_save.columns:
            degradation_save[col] = degradation_save[col].map(fmt_4dec)

    degradation_save.to_csv(
        OUTPUT_DIR / "density_degradation.csv",
        index=False,
        float_format="%.4f",
    )
    print(f"Saved: {OUTPUT_DIR / 'density_degradation.csv'}")

    # --------------------------------------------------------
    # 4.6 Console summary
    # --------------------------------------------------------
    print("\nSaved tables:")
    print(f"  {EVALUATION_SUMMARY_CSV}")
    print(f"  {OUTPUT_DIR / 'density_benchmark.csv'}")
    print(f"  {OUTPUT_DIR / 'density_degradation.csv'}")


if __name__ == '__main__':
    main()