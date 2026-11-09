import re
import os
import json
import math
from pathlib import Path
import pandas as pd

CHART_PATTERN = re.compile(
    r"^\s*(?:#{0,6}\s*)?(?:\*\*)?\s*(?:Chart|Table)\s+(\d+)\b",
    re.IGNORECASE,
)
NUMBERED_CHART_PATTERN = re.compile(
    r"^\s*#{1,6}\s*(\d+)\.\s+",
    re.IGNORECASE,
)


def normalize(value):
    return str(value).strip().casefold()


def to_number(value):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def is_table_row(line):
    line = line.strip()
    return line.startswith("|") and line.endswith("|")


def split_row(line):
    return [x.strip() for x in line.strip().strip("|").split("|")]


def is_separator(line):
    if not is_table_row(line):
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in split_row(line))


# ============================================================
# 1. Markdown Table Parsing (Single Table + Chart Splitting + Continuous Table Splitting)
# ============================================================

def get_chart_markers(lines):
    markers = []
    for i, line in enumerate(lines):
        m = CHART_PATTERN.match(line)
        if m:
            markers.append((i, int(m.group(1))))
    if markers:
        return markers
    for i, line in enumerate(lines):
        m = NUMBERED_CHART_PATTERN.match(line)
        if m:
            markers.append((i, int(m.group(1))))
    return markers


def has_chart_marker(lines):
    return bool(get_chart_markers(lines))


def split_charts(lines):
    markers = get_chart_markers(lines)
    if not markers:
        return []
    charts = []
    first_marker, first_number = markers[0]
    if first_marker > 0 and first_number != 1:
        charts.append(("Chart 1", lines[:first_marker]))
    for i, (start, number) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else len(lines)
        charts.append((f"Chart {number}", lines[start + 1:end]))
    return charts


def extract_first_table(lines):
    """Extract the first Markdown table from lines, returning a DataFrame or None."""
    first_table = None
    for i in range(len(lines) - 1):
        if is_table_row(lines[i]) and is_separator(lines[i + 1]):
            first_table = i
            break
    if first_table is None:
        return None

    header = split_row(lines[first_table])
    rows = []
    i = first_table + 2
    while i < len(lines):
        # Skip nested header+separator
        if i + 1 < len(lines) and is_table_row(lines[i]) and is_separator(lines[i + 1]):
            i += 2
            continue
        if is_table_row(lines[i]):
            row = split_row(lines[i])
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            rows.append(row[:len(header)])
        i += 1
    return pd.DataFrame(rows, columns=header) if rows else None


def extract_raw_tables(lines):
    starts = [i for i in range(len(lines) - 1)
              if is_table_row(lines[i]) and is_separator(lines[i + 1])]
    tables = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        header = split_row(lines[start])
        rows = []
        for line in lines[start + 2:end]:
            if not is_table_row(line):
                continue
            row = split_row(line)
            if len(row) < len(header):
                row += [""] * (len(header) - len(row))
            rows.append(row[:len(header)])
        tables.append(pd.DataFrame(rows, columns=header))
    return tables


# ---------- Automatic splitting of continuous tables (for script 1) ----------

def find_x_column(df):
    for col in df.columns:
        name = normalize(col)
        if "x value" in name:
            return col
        if name == "x" or name.startswith("x "):
            return col
    if len(df.columns) >= 2:
        return df.columns[1]
    return None


def find_y_column(df):
    for col in df.columns:
        name = normalize(col)
        if "y value" in name:
            return col
        if name == "y" or name.startswith("y "):
            return col
    if len(df.columns) >= 3:
        return df.columns[2]
    return None


def get_structure_columns(df):
    columns = []
    for col in df.columns[:2]:
        values = [to_number(x) for x in df[col]]
        ratio = sum(x is not None for x in values) / max(len(values), 1)
        if ratio < 0.8:
            columns.append(col)
    return columns


def find_x_reset_points(df):
    x_col = find_x_column(df)
    if x_col is None:
        return []
    values = [to_number(x) for x in df[x_col]]
    if any(x is None for x in values):
        return []
    return [i for i in range(1, len(values)) if values[i] < values[i - 1]]


def is_structure_boundary(df, point, window=3):
    if point <= 1 or point >= len(df) - 1:
        return False
    columns = get_structure_columns(df)
    if not columns:
        return False
    x_col = find_x_column(df)
    series_col = columns[0]

    if x_col is not None:
        left_series = normalize(df.iloc[point - 1][series_col])
        right_series = normalize(df.iloc[point][series_col])
        if left_series != right_series:
            left_x = [normalize(x) for x in df.loc[
                df[series_col].map(normalize) == left_series, x_col]]
            right_x = [normalize(x) for x in df.loc[
                df[series_col].map(normalize) == right_series, x_col]]
            return left_x != right_x

    left = df.iloc[max(0, point - window):point]
    right = df.iloc[point:min(len(df), point + window)]
    changed = 0
    for col in columns:
        lv = {normalize(x) for x in left[col]}
        rv = {normalize(x) for x in right[col]}
        if lv and rv and lv.isdisjoint(rv):
            changed += 1
    return changed == len(columns)


def split_continuous_table(df):
    if len(df) < 6:
        return [df.reset_index(drop=True)]
    boundaries = set(find_x_reset_points(df))
    for point in range(3, len(df) - 2):
        if is_structure_boundary(df, point):
            boundaries.add(point)
    boundaries = sorted(boundaries)
    if not boundaries:
        return [df.reset_index(drop=True)]
    valid = []
    last = 0
    for p in boundaries:
        if p - last < 3:
            continue
        valid.append(p)
        last = p
    if len(df) - last < 3:
        return [df.reset_index(drop=True)]
    bounds = [0, *valid, len(df)]
    return [df.iloc[s:e].reset_index(drop=True) for s, e in zip(bounds, bounds[1:])]


# ---------- Unified entry point: given md text -> [(chart_name, df), ...] ----------

def parse_md_tables(md_text, mode="compound"):
    """
    mode = "compound": supports Chart markers / automatic continuous table splitting (script 1 behavior)
    mode = "single":   only takes the first table (script 2 behavior)
    """
    lines = md_text.splitlines()

    if mode == "single":
        df = extract_first_table(lines)
        return [("Chart 1", df)] if df is not None else []

    # compound mode
    if has_chart_marker(lines):
        results = []
        for chart_name, chart_lines in split_charts(lines):
            df = extract_first_table(chart_lines)
            if df is not None:
                results.append((chart_name, df))
        return results

    results = []
    for raw_table in extract_raw_tables(lines):
        for i, df in enumerate(split_continuous_table(raw_table), start=1):
            results.append((f"Chart {i}", df))
    return results


# ============================================================
# 2. Original Data Loading
# ============================================================

def get_original_csv_path(source_image, table_dir):
    csv_filename = os.path.splitext(source_image)[0] + ".csv"
    return os.path.join(table_dir, csv_filename)


def load_original_chart(csv_path, separator=",", encoding="utf-8"):
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path, sep=separator, encoding=encoding)
    except Exception as e:
        print(f"  ✗ Error reading {csv_path}: {e}")
        return None


# ============================================================
# 3. Completeness Ratio Calculation
# ============================================================

def is_numeric_series(series, threshold=0.8):
    values = [to_number(v) for v in series]
    if not values:
        return False
    return sum(v is not None for v in values) / len(values) >= threshold


def count_x_values(df, numeric_only=True):
    x_col = find_x_column(df)
    if x_col is None:
        return 0, False
    series = df[x_col]
    numeric = is_numeric_series(series)
    if numeric and numeric_only:
        count = sum(to_number(v) is not None for v in series)
    else:
        count = sum(str(v).strip() != "" for v in series)
    return count, numeric


def compute_completeness_ratio(extracted_df, original_df):
    orig_count, orig_is_numeric = count_x_values(original_df)
    ext_count, _ = count_x_values(extracted_df)
    ratio = (ext_count / orig_count) if orig_count else None
    return {
        "x_is_numeric": orig_is_numeric,
        "extracted_x_count": ext_count,
        "original_x_count": orig_count,
        "completeness_ratio": ratio,
    }


# ============================================================
# 4. Y Value Error Calculation (MAE + RMSE)
# ============================================================

def find_series_column(df):
    for col in df.columns:
        values = [to_number(x) for x in df[col]]
        ratio = sum(x is not None for x in values) / max(len(values), 1)
        if ratio < 0.8:
            return col
    return None


def edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def match_series_names(ext_names, orig_names):
    mapping = {}
    for ext_name in ext_names:
        best_orig, best_dist = None, float("inf")
        for orig_name in orig_names:
            d = edit_distance(ext_name, orig_name)
            if d < best_dist:
                best_dist, best_orig = d, orig_name
        threshold = max(len(ext_name), len(best_orig) if best_orig else 0) / 2
        if best_dist > threshold:
            return None
        mapping[ext_name] = best_orig
    return mapping


def get_series_groups(df):
    series_col = find_series_column(df)
    if series_col is None:
        return {"": df}
    groups = {}
    for val in df[series_col].map(normalize).unique():
        groups[val] = df[df[series_col].map(normalize) == val].reset_index(drop=True)
    return groups


def get_xy_pairs(df):
    x_col = find_x_column(df)
    y_col = find_y_column(df)
    if x_col is None or y_col is None:
        return []
    return [(xr, to_number(xr), to_number(yr))
            for xr, yr in zip(df[x_col], df[y_col])]


def _pairs_by_nearest_x(ext_pairs, orig_pairs):
    orig_numeric = [(xr, xn, yn) for xr, xn, yn in orig_pairs if xn is not None]
    if not orig_numeric:
        return []
    candidates = []
    for ext_idx, (_, ext_x, ext_y) in enumerate(ext_pairs):
        if ext_x is None or ext_y is None:
            continue
        best_idx, best_d = None, float("inf")
        for orig_idx, (_, orig_x, orig_y) in enumerate(orig_numeric):
            if orig_y is None:
                continue
            d = abs(orig_x - ext_x)
            if d < best_d:
                best_d, best_idx = d, orig_idx
        if best_idx is not None:
            candidates.append((ext_idx, best_idx, best_d, ext_y,
                               orig_numeric[best_idx][2]))
    candidates.sort(key=lambda x: x[2])
    used, matches = set(), []
    for _, orig_idx, _, ext_y, orig_y in candidates:
        if orig_idx not in used:
            matches.append((ext_y, orig_y))
            used.add(orig_idx)
    return matches


def _pairs_by_order(ext_pairs, orig_pairs):
    matches = []
    for (_, _, ext_y), (_, _, orig_y) in zip(ext_pairs, orig_pairs):
        if ext_y is None or orig_y is None:
            continue
        matches.append((ext_y, orig_y))
    return matches


def compute_y_error_by_series(extracted_df, original_df, x_is_numeric, max_error=100):
    ext_groups = get_series_groups(extracted_df)
    orig_groups = get_series_groups(original_df)

    if list(ext_groups.keys()) == [""] and list(orig_groups.keys()) == [""]:
        series_mapping = {"": ""}
    else:
        series_mapping = match_series_names(list(ext_groups.keys()),
                                            list(orig_groups.keys()))
        if series_mapping is None:
            series_mapping = {"": ""}

    all_matches = []
    total_original_x_count = 0

    for ext_series, orig_series in series_mapping.items():
        if ext_series == "" and orig_series == "":
            ext_sub, orig_sub = extracted_df, original_df
        else:
            if ext_series not in ext_groups or orig_series not in orig_groups:
                continue
            ext_sub = ext_groups[ext_series]
            orig_sub = orig_groups[orig_series]

        ext_pairs = get_xy_pairs(ext_sub)
        orig_pairs = get_xy_pairs(orig_sub)
        total_original_x_count += len([p for p in orig_pairs if p[1] is not None])

        if x_is_numeric:
            all_matches.extend(_pairs_by_nearest_x(ext_pairs, orig_pairs))
        else:
            all_matches.extend(_pairs_by_order(ext_pairs, orig_pairs))

    point_coverage = (len(all_matches) / total_original_x_count) \
        if total_original_x_count > 0 else None

    if not all_matches:
        return {"matched_count": 0, "point_coverage": point_coverage,
                "mae": None, "rmse": None}

    errors_relative, errors_normalized = [], []
    for ext_y, orig_y in all_matches:
        if abs(orig_y) > 1e-6:
            errors_relative.append(
                min(abs(ext_y - orig_y) / abs(orig_y) * 100, max_error))

    if len(all_matches) > 1:
        orig_ys = [y for _, y in all_matches]
        ext_ys = [y for y, _ in all_matches]
        orig_min, orig_max = min(orig_ys), max(orig_ys)
        if orig_max - orig_min > 1e-6:
            for ext_y, orig_y in all_matches:
                ne = (ext_y - orig_min) / (orig_max - orig_min) * 100
                no = (orig_y - orig_min) / (orig_max - orig_min) * 100
                errors_normalized.append(abs(ne - no))

    if errors_normalized:
        errors = errors_normalized
    elif errors_relative:
        errors = errors_relative
    else:
        errors = [min(abs(e - o), max_error) for e, o in all_matches]

    if not errors:
        return {"matched_count": len(all_matches),
                "point_coverage": point_coverage, "mae": None, "rmse": None}

    mae = sum(errors) / len(errors)
    rmse = math.sqrt(sum(e ** 2 for e in errors) / len(errors))
    if mae > 100 or rmse > 100:
        mae = rmse = None

    return {"matched_count": len(all_matches), "point_coverage": point_coverage,
            "mae": mae, "rmse": rmse}


# ============================================================
# 5. Unified Evaluation Core: given a set of (chart_name, extracted_df, source_image)
# ============================================================

def evaluate_chart(chart_name, extracted_df, source_image, table_dir):
    csv_path = get_original_csv_path(source_image, table_dir)
    original_df = load_original_chart(csv_path)
    if original_df is None:
        print(f"  ✗ Original data does not exist or failed to read: {csv_path}")
        return None

    completeness = compute_completeness_ratio(extracted_df, original_df)
    y_error = compute_y_error_by_series(
        extracted_df, original_df, completeness["x_is_numeric"])

    return {
        "source_image": source_image,
        "chart": chart_name,
        **completeness,
        "matched_count": y_error["matched_count"],
        "point_coverage": y_error["point_coverage"],
        "mae": y_error["mae"],
        "rmse": y_error["rmse"],
    }


# ---------- Compound Chart Entry Point ----------

def load_compound_info(compound_info_path, top_n=None):
    with open(compound_info_path, "r", encoding="utf-8") as f:
        compound_info = json.load(f)
    compounds = compound_info["compounds"]
    return compounds[:top_n] if top_n else compounds


def get_extracted_md_path(output_filename, results_dir):
    md_filename = os.path.splitext(output_filename)[0] + ".md"
    return os.path.join(results_dir, md_filename)


def evaluate_compound(compound, results_dir, table_dir):
    compound_id = compound["compound_id"]
    output_filename = compound["output_filename"]
    source_images = compound["source_images"]

    md_path = get_extracted_md_path(output_filename, results_dir)
    if not os.path.exists(md_path):
        print(f"  ✗ Extraction result does not exist: {md_path}")
        return [], None

    md_text = Path(md_path).read_text(encoding="utf-8")
    charts = parse_md_tables(md_text, mode="compound")

    if len(charts) != len(source_images):
        mismatch_info = {
            "compound_id": compound_id,
            "output_filename": output_filename,
            "expected_chart_count": len(source_images),
            "extracted_chart_count": len(charts),
            "chart_count_diff": len(charts) - len(source_images),
        }
        print(f"  ! {output_filename}: chart count mismatch "
              f"(expected {len(source_images)}, extracted {len(charts)}), skipping metric calculation")
        return [], mismatch_info

    records = []
    for (chart_name, ext_df), source_image in zip(charts, source_images):
        rec = evaluate_chart(chart_name, ext_df, source_image, table_dir)
        if rec is not None:
            rec["compound_id"] = compound_id
            rec["output_filename"] = output_filename
            records.append(rec)
    return records, None


def evaluate_all_compounds(compound_info_path, results_dir, table_dir,
                           top_n=None, only_existing_md=True):
    compounds = load_compound_info(compound_info_path, top_n=top_n)
    all_records, all_mismatches, processed_count = [], [], 0

    for compound in compounds:
        md_path = get_extracted_md_path(compound["output_filename"], results_dir)
        if only_existing_md and not os.path.exists(md_path):
            continue
        processed_count += 1
        print(f"Compound {compound['compound_id']}:")
        records, mismatch = evaluate_compound(compound, results_dir, table_dir)
        all_records.extend(records)
        if mismatch is not None:
            all_mismatches.append(mismatch)
        print()

    return pd.DataFrame(all_records), all_mismatches, processed_count


# ---------- Single Image Entry Point ----------

def evaluate_single_image(source_image, results_dir, table_dir):
    md_filename = os.path.splitext(source_image)[0] + ".md"
    md_path = os.path.join(results_dir, md_filename)
    if not os.path.exists(md_path):
        print(f"  ✗ Extraction result does not exist: {md_path}")
        return None

    md_text = Path(md_path).read_text(encoding="utf-8")
    charts = parse_md_tables(md_text, mode="single")
    if not charts:
        print(f"  ✗ Failed to extract a table from the MD: {md_path}")
        return None

    chart_name, ext_df = charts[0]
    return evaluate_chart(chart_name, ext_df, source_image, table_dir)


def evaluate_all_images(results_dir, table_dir, image_extensions=(".jpg",)):
    all_records, all_mismatches, processed_count = [], [], 0

    md_files = [f for f in os.listdir(results_dir) if f.endswith(".md")]
    for md_file in sorted(md_files):
        base_name = os.path.splitext(md_file)[0]
        source_image = None
        for ext in image_extensions:
            candidate = base_name + ext
            if os.path.exists(get_original_csv_path(candidate, table_dir)):
                source_image = candidate
                break
        if source_image is None:
            print(f"  ! Unable to find the source image or CSV file corresponding to {base_name}")
            continue

        processed_count += 1
        print(f"Image {source_image}:")
        rec = evaluate_single_image(source_image, results_dir, table_dir)
        if rec:
            all_records.append(rec)
        print()

    return pd.DataFrame(all_records), all_mismatches, processed_count


# ============================================================
# 6. Result Serialization (identical in both scripts, keeping one)
# ============================================================

def _to_jsonable(value):
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_result_json(result_df, mismatches, processed_count):
    records = [{k: _to_jsonable(v) for k, v in row.items()}
               for row in result_df.to_dict(orient="records")]

    def _mean(key):
        values = [r[key] for r in records if r.get(key) is not None]
        return sum(values) / len(values) if values else None

    mismatch_count = len(mismatches)
    mismatch_ratio = (mismatch_count / processed_count) if processed_count else None

    null_mae_rmse = [r for r in records
                     if r.get("mae") is None and r.get("rmse") is None]
    total = len(records)

    summary = {
        "total_processed_compounds": processed_count,
        "mismatch_compound_count": mismatch_count,
        "mismatch_compound_ratio": mismatch_ratio,
        "total_aligned_charts": total,
        "avg_completeness_ratio": _mean("completeness_ratio"),
        "avg_mae": _mean("mae"),
        "avg_rmse": _mean("rmse"),
        "avg_point_coverage": _mean("point_coverage"),
        "null_mae_rmse_source_image_count": len(null_mae_rmse),
        "null_mae_rmse_source_image_ratio":
            (len(null_mae_rmse) / total) if total else None,
    }
    return {"summary": summary,
            "chart_count_mismatches": mismatches,
            "records": records}


def save_results_json(result_df, mismatches, processed_count, output_path):
    result_json = build_result_json(result_df, mismatches, processed_count)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)
    return result_json


# ============================================================
# 7. Unified CLI Entry Point
# ============================================================

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d)


def run_compound_mode(compound_info_path, results_dir, table_dir,
                      output_json_path, top_n=None):
    _ensure_dir(output_json_path)
    result_df, mismatches, processed = evaluate_all_compounds(
        compound_info_path, results_dir, table_dir, top_n=top_n)

    print("=" * 60)
    print(f"Total compounds processed: {processed}")
    print(f"Chart count mismatches: {len(mismatches)}")
    print(f"Successfully aligned charts: {len(result_df)}")
    print("=" * 60)

    result_json = save_results_json(result_df, mismatches, processed,
                                    output_json_path)
    s = result_json["summary"]
    print(f"\nResults saved to: {output_json_path}")
    print(f"Mismatch compound ratio: {s['mismatch_compound_ratio']}")
    print(f"Average completeness ratio: {s['avg_completeness_ratio']}")
    print(f"Average Point Coverage: {s['avg_point_coverage']}")
    print(f"Average MAE: {s['avg_mae']}")
    print(f"Average RMSE: {s['avg_rmse']}")


def run_single_mode(results_dir, table_dir, output_json_path,
                    image_extensions=(".jpg",)):
    _ensure_dir(output_json_path)
    result_df, mismatches, processed = evaluate_all_images(
        results_dir, table_dir, image_extensions=image_extensions)

    print("=" * 60)
    print(f"Total images processed: {processed}")
    print(f"Mismatch count: {len(mismatches)}")
    print(f"Successfully extracted tables: {len(result_df)}")
    print("=" * 60)

    result_json = save_results_json(result_df, mismatches, processed,
                                    output_json_path)
    s = result_json["summary"]
    print(f"\nResults saved to: {output_json_path}")
    print(f"Mismatch ratio: {s['mismatch_compound_ratio']}")
    print(f"Average completeness ratio: {s['avg_completeness_ratio']}")
    print(f"Average MAE: {s['avg_mae']}")
    print(f"Average RMSE: {s['avg_rmse']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["compound", "single"], required=True)
    parser.add_argument("--results_dir", default="ICPR_filtered/test_compound9/results_doubao")
    parser.add_argument("--table_dir", default="ICPR_filtered/test_table")
    parser.add_argument("--output_json", default="ICPR_filtered/results/test_compound9/evaluation_results_doubao.json")
    parser.add_argument("--compound_info", help="required in compound mode", default="ICPR_filtered/test_compound9/compound_info_9.json")
    parser.add_argument("--top_n", type=int, default=None)
    parser.add_argument("--image_ext", default=".jpg")

    args = parser.parse_args()

    if args.mode == "compound":
        assert args.compound_info, "--compound_info is required in compound mode"
        run_compound_mode(args.compound_info, args.results_dir, args.table_dir,
                          args.output_json, top_n=args.top_n)
    else:
        run_single_mode(args.results_dir, args.table_dir, args.output_json,
                        image_extensions=(args.image_ext,))