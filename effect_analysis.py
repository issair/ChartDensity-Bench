import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

BASE_DIRECTORY = Path("ICPR_filtered/results")
OUTPUT_DIRECTORY = Path("density_regression_results")
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
MODEL_ORDER = ["doubao", "gemini", "glm", "kimi", "qwen"]
DENSITY_ORDER = [1, 3, 6, 9]
DIRECTORY_TO_DENSITY = {
    ".": 1,
    "test": 1,
    "test_compound3": 3,
    "test_compound6": 6,
    "test_compound9": 9,
}

def print_section(title):
    print("\n")
    print("=" * 100)
    print(title)
    print("=" * 100)

def normalize_model_name(name):
    return str(name).strip().lower()

def get_density(directory_name):
    if directory_name in DIRECTORY_TO_DENSITY:
        return DIRECTORY_TO_DENSITY[directory_name]
    name = str(directory_name).lower()
    if "compound3" in name:
        return 3
    if "compound6" in name:
        return 6
    if "compound9" in name:
        return 9
    if name in {".", "test", "single", "single_chart"}:
        return 1
    raise ValueError(f"Unable to recognize the density of the directory: {directory_name}")

def fmt_4dec(x):
    if pd.isna(x):
        return np.nan
    return round(float(x), 4)

def collect_chart_level_records(base_dir):
    all_records = []
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"BASE_DIRECTORY does not exist: {base_dir.resolve()}")
    directories = [d for d in base_dir.iterdir() if d.is_dir()]
    directories.insert(0, base_dir)
    for directory in directories:
        json_files = sorted(directory.glob("evaluation_results_*.json"))
        if not json_files:
            continue
        directory_name = directory.name if directory != base_dir else "."
        density = get_density(directory_name)
        print_section(f"Reading directory: {directory_name} | density={density}")
        for json_file in json_files:
            model_name = normalize_model_name(json_file.stem.replace("evaluation_results_", ""))
            print(f"Processing: {json_file}")
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            if not isinstance(records, list):
                warnings.warn(f"{json_file}: records is not a list, skipping.")
                continue
            for record_idx, record in enumerate(records):
                source_image = record.get("source_image")
                if source_image is None:
                    warnings.warn(f"{json_file}: record {record_idx} is missing source_image, skipping.")
                    continue
                chart_id = str(source_image).strip()
                row = {
                    "model": model_name,
                    "directory": directory_name,
                    "density": density,
                    "record_idx": record_idx,
                    "source_image": source_image,
                    "chart_id": chart_id,
                    "compound_id": record.get("compound_id"),
                    "output_filename": record.get("output_filename"),
                    "chart": record.get("chart"),
                    "x_is_numeric": record.get("x_is_numeric"),
                    "extracted_x_count": record.get("extracted_x_count"),
                    "original_x_count": record.get("original_x_count"),
                    "completeness_ratio": record.get("completeness_ratio"),
                    "matched_count": record.get("matched_count"),
                    "point_coverage": record.get("point_coverage"),
                    "mae": record.get("mae"),
                    "rmse": record.get("rmse"),
                }
                all_records.append(row)
    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("No chart-level records were read.")
    return df

def prepare_dataframe(df):
    df = df.copy()
    numeric_columns = [
        "density",
        "extracted_x_count",
        "original_x_count",
        "completeness_ratio",
        "matched_count",
        "point_coverage",
        "mae",
        "rmse",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["model"] = df["model"].astype(str).str.lower()
    df["density"] = pd.Categorical(df["density"], categories=DENSITY_ORDER, ordered=True)
    df["chart_id"] = df["source_image"].astype(str).str.strip()
    unknown_models = sorted(set(df["model"].unique()) - set(MODEL_ORDER))
    if unknown_models:
        warnings.warn(f"Models not defined in MODEL_ORDER were found: {unknown_models}")
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    df = df.sort_values(["model", "density", "chart_id"]).reset_index(drop=True)
    return df

def validate_chart_structure(df):
    print_section("Chart-level structure validation")
    duplicate_mask = df.duplicated(subset=["model", "density", "chart_id"], keep=False)
    duplicates = df[duplicate_mask]
    print(f"Duplicate model-density-chart observations: {len(duplicates)}")
    if not duplicates.empty:
        print(duplicates[["model", "density", "chart_id", "record_idx", "compound_id", "chart"]].head(20).to_string(index=False))
        print("\nWARNING: Duplicate observations exist.\nBefore mixed-effects regression, you must confirm whether these are true duplicates.")
    print(f"Total chart-level records: {len(df)}")
    print(f"Unique source charts: {df['chart_id'].nunique()}")
    coverage = df.groupby("chart_id", observed=True)["density"].nunique().value_counts().sort_index()
    print("\nNumber of density levels per chart:")
    print(coverage.to_string())

def fit_mixed_model(df, dependent_variable, model_reference="kimi"):
    print_section(f"Mixed-effects regression: {dependent_variable.upper()}")
    analysis_df = df[df[dependent_variable].notna()].copy()
    analysis_df = analysis_df[analysis_df["model"].notna()]
    analysis_df = analysis_df[analysis_df["density"].notna()]
    print(f"Valid observations: {len(analysis_df)}")
    print(f"Unique charts: {analysis_df['chart_id'].nunique()}")
    analysis_df["chart_id"] = analysis_df["chart_id"].astype(str)
    formula = f"{dependent_variable} ~ C(model, Treatment(reference='{model_reference}')) * C(density, Treatment(reference=1))"
    print("\nFormula:")
    print(formula)
    model = smf.mixedlm(formula, data=analysis_df, groups=analysis_df["chart_id"], re_formula="1")
    fit_result = None
    optimizers = ["lbfgs", "powell", "cg"]
    for optimizer in optimizers:
        try:
            print(f"\nTrying optimizer: {optimizer}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit_result = model.fit(reml=False, method=optimizer, maxiter=2000, disp=False)
            if fit_result is not None:
                print(f"Successfully fitted with {optimizer}")
                break
        except Exception as e:
            print(f"Optimizer {optimizer} failed: {e}")
    if fit_result is None:
        raise RuntimeError(f"{dependent_variable} mixed model fitting failed.")
    print("\nModel summary:")
    print(fit_result.summary())
    with open(OUTPUT_DIRECTORY / f"{dependent_variable}_mixed_model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(fit_result.summary()))
    fixed = pd.DataFrame({
        "term": fit_result.fe_params.index,
        "coef": fit_result.fe_params.values,
        "std_error": fit_result.bse_fe.values,
        "z": fit_result.fe_params.values / fit_result.bse_fe.values,
        "p_value": fit_result.pvalues[fit_result.fe_params.index].values,
    })
    fixed["ci_low"] = fixed["coef"] - 1.96 * fixed["std_error"]
    fixed["ci_high"] = fixed["coef"] + 1.96 * fixed["std_error"]
    
    # Keep 4 decimal places when saving
    fixed_save = fixed.copy()
    numeric_cols_fixed = ["coef", "std_error", "z", "p_value", "ci_low", "ci_high"]
    for col in numeric_cols_fixed:
        if col in fixed_save.columns:
            fixed_save[col] = fixed_save[col].map(fmt_4dec)
    fixed_save.to_csv(OUTPUT_DIRECTORY / f"{dependent_variable}_fixed_effects.csv", index=False, float_format="%.4f")
    
    print("\nFixed effects:")
    print(fixed.to_string(index=False))
    return fit_result, analysis_df, fixed

def calculate_density_slopes(df, dependent_variable):
    print_section(f"Auxiliary density sensitivity slopes: {dependent_variable.upper()}")
    rows = []
    import statsmodels.api as sm
    for model in MODEL_ORDER:
        sub = df[(df["model"] == model) & df[dependent_variable].notna()].copy()
        if len(sub) < 5:
            continue
        X = sm.add_constant(sub["density"].astype(float))
        y = sub[dependent_variable]
        regression = sm.OLS(y, X).fit()
        beta = regression.params["density"]
        se = regression.bse["density"]
        p = regression.pvalues["density"]
        rows.append({
            "model": model,
            "dependent": dependent_variable,
            "slope": beta,
            "std_error": se,
            "p_value": p,
            "ci_low": beta - 1.96 * se,
            "ci_high": beta + 1.96 * se,
            "r_squared": regression.rsquared,
            "n": len(sub),
        })
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    
    # Keep 4 decimal places when saving
    result_save = result.copy()
    numeric_cols_slope = ["slope", "std_error", "p_value", "ci_low", "ci_high", "r_squared"]
    for col in numeric_cols_slope:
        if col in result_save.columns:
            result_save[col] = result_save[col].map(fmt_4dec)
    result_save.to_csv(OUTPUT_DIRECTORY / f"{dependent_variable}_density_slopes.csv", index=False, float_format="%.4f")
    return result

def export_analysis_dataset(df):
    output_columns = [
        "model",
        "directory",
        "density",
        "source_image",
        "chart_id",
        "compound_id",
        "output_filename",
        "chart",
        "x_is_numeric",
        "extracted_x_count",
        "original_x_count",
        "completeness_ratio",
        "matched_count",
        "point_coverage",
        "mae",
        "rmse",
    ]
    # Keep 4 decimal places when exporting
    df_save = df[output_columns].copy()
    numeric_cols_dataset = ["density", "extracted_x_count", "original_x_count", "completeness_ratio", 
                            "matched_count", "point_coverage", "mae", "rmse"]
    for col in numeric_cols_dataset:
        if col in df_save.columns:
            df_save[col] = df_save[col].map(fmt_4dec)
    df_save.to_csv(OUTPUT_DIRECTORY / "chart_level_analysis_dataset.csv", index=False, float_format="%.4f")

def main():
    print_section("Density Regression for Chart Reconstruction Benchmark")
    print(f"Base directory:\n{BASE_DIRECTORY.resolve()}")
    print(f"Output directory:\n{OUTPUT_DIRECTORY.resolve()}")
    df = collect_chart_level_records(BASE_DIRECTORY)
    df = prepare_dataframe(df)
    validate_chart_structure(df)
    export_analysis_dataset(df)
    mae_fit, mae_analysis, mae_fixed = fit_mixed_model(df, dependent_variable="mae", model_reference="kimi")
    rmse_fit, rmse_analysis, rmse_fixed = fit_mixed_model(df, dependent_variable="rmse", model_reference="kimi")
    mae_slopes = calculate_density_slopes(df, "mae")
    rmse_slopes = calculate_density_slopes(df, "rmse")

if __name__ == "__main__":
    main()