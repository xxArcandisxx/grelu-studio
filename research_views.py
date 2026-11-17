"""Evaluation and interval checks for researcher-supplied tabular data."""
import csv
import io
import re

from flask import Blueprint, abort, make_response, render_template, request
import numpy as np

from core.stats import auroc, expected_calibration_error, pearson, spearman

research = Blueprint("research", __name__)
PAGES = {
    "benchmark": ("Benchmark Evaluation", "Compare prediction scores against measured values. Upload CSV with observed and predicted columns."),
    "tracks": ("Track Analysis", "Compare numerical tracks across matched rows. Upload a CSV with two to sixteen numeric columns."),
    "bed": ("BED Validation", "Check the syntax and interval ordering of a tab-separated BED file. Coordinates are zero-based and end-exclusive."),
}
EXAMPLES = {
    "benchmark": "observed,predicted\n0,0.15\n1,0.8\n0,0.3\n1,0.7",
    "tracks": "track_a,track_b,track_c\n1,3,9\n2,6,7\n3,9,4\n4,12,1",
    "bed": "chr7\t5530500\t5530610\tACTB\nchr12\t6534450\t6534540\tGAPDH",
}


def numeric_table(text):
    with io.StringIO(text) as source:
        reader = csv.reader(source, strict=True)
        headers = [value.strip() for value in next(reader, [])]
        if not 2 <= len(headers) <= 16 or not all(headers) or len(set(headers)) != len(headers):
            raise ValueError("Use two to sixteen nonempty, unique column headers.")
        rows = []
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != len(headers):
                raise ValueError(f"Line {reader.line_num}: expected {len(headers)} values.")
            try:
                values = [float(field) for field in row]
            except ValueError as error:
                raise ValueError(f"Line {reader.line_num}: all values must be numeric.") from error
            if not all(np.isfinite(values)) or any(abs(value) > 1e100 for value in values):
                raise ValueError(f"Line {reader.line_num}: use finite values with magnitude at most 1e100.")
            rows.append(values)
            if len(rows) > 5000:
                raise ValueError("Use at most 5,000 data rows.")
    if len(rows) < 3:
        raise ValueError("Provide at least three paired data rows.")
    return headers, np.array(rows, dtype=float)


def benchmark(text, kind):
    headers, values = numeric_table(text)
    if set(headers) != {"observed", "predicted"}:
        raise ValueError("Use exactly the observed and predicted columns.")
    y = values[:, headers.index("observed")]
    pred = values[:, headers.index("predicted")]
    count = len(y)
    if kind == "binary":
        if not np.isin(y, [0, 1]).all() or not ((pred >= 0) & (pred <= 1)).all():
            raise ValueError("Binary evaluation requires 0/1 labels and predictions between 0 and 1.")
        from sklearn.metrics import average_precision_score

        positives = int(np.sum(y))
        ece, bins = expected_calibration_error(y, pred)
        both = 0 < positives < count
        metrics = {"Samples": count, "Positive fraction": positives / count, "AUROC": auroc(y, pred),
                   "Average precision": float(average_precision_score(y, pred)) if both else None,
                   "Brier score": float(np.mean((pred - y) ** 2)), "Calibration error (10 bins)": ece}
        return {"mode": kind, "metrics": metrics, "calibration_bins": bins}, [
            ("Benchmark metrics", ["Metric", "Value"], metrics.items()),
            ("Calibration bins", ["Bin", "Samples", "Mean prediction", "Observed fraction"],
             [(entry["bin"] + 1, entry["n"], entry["mean_pred"], entry["frac_pos"]) for entry in bins]),
        ]
    if kind != "regression":
        raise ValueError("Choose regression or binary classification.")
    mse = float(np.mean((pred - y) ** 2))
    metrics = {"Samples": count, "MAE": float(np.mean(np.abs(pred - y))), "RMSE": mse ** 0.5,
               "Pearson r": pearson(y, pred), "Spearman rho": spearman(y, pred)}
    return {"mode": kind, "metrics": metrics}, [("Benchmark metrics", ["Metric", "Value"], metrics.items())]


def track_analysis(text):
    headers, data = numeric_table(text)
    tracks = [{"name": name, "mean": float(values.mean()), "sd": float(values.std()),
               "min": float(values.min()), "max": float(values.max())}
              for name, values in zip(headers, data.T, strict=True)]
    matrix = [[pearson(first, second) for second in data.T] for first in data.T]
    return {"samples": len(data), "tracks": tracks, "pearson": matrix}, [
        ("Track summaries", ["Track", "Mean", "Population SD", "Minimum", "Maximum"],
         [(r["name"], r["mean"], r["sd"], r["min"], r["max"]) for r in tracks]),
        ("Pearson correlation", ["Track", *headers], [(name, *row) for name, row in zip(headers, matrix, strict=True)]),
    ]


def bed_validation(text):
    valid = 0
    invalid = []
    total = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith(("#", "track ", "browser ")):
            continue
        total += 1
        if total > 5000:
            raise ValueError("Use at most 5,000 BED records.")
        columns = line.split("\t")
        if len(columns) < 3:
            problem = "Expected at least three tab-separated columns."
        elif not re.fullmatch(r"[A-Za-z0-9_.-]+", columns[0]):
            problem = "Invalid chromosome or contig name."
        elif not all(re.fullmatch(r"[0-9]{1,10}", value) for value in columns[1:3]):
            problem = "Start and end must be nonnegative integers."
        elif int(columns[1]) >= int(columns[2]):
            problem = "End must be greater than start."
        elif len(columns) >= 6 and columns[5] not in {"+", "-", "."}:
            problem = "Strand must be +, -, or ."
        else:
            valid += 1
            continue
        invalid.append({"line": line_no, "issue": problem})
    if not total:
        raise ValueError("Provide at least one BED interval.")
    sections = [("Validation summary", ["Metric", "Records"],
                 [("Total", total), ("Valid", valid), ("Invalid", len(invalid))])]
    if invalid:
        sections.append(("Records to fix", ["Line", "Issue"], [(r["line"], r["issue"]) for r in invalid]))
    return {"total": total, "valid": valid, "issues": invalid}, sections


@research.route("/research/<mode>", methods=["GET", "POST"])
def run(mode):
    if mode not in PAGES:
        abort(404)
    title, description = PAGES[mode]
    sample = request.form.get("text", "")
    kind = request.form.get("kind", "regression")
    result, tables, error, status = None, [], None, 200
    if request.method == "POST":
        try:
            upload = request.files.get("file")
            if "example" in request.form:
                sample = EXAMPLES[mode]
            elif upload and upload.filename:
                sample = upload.read().decode("utf-8-sig")
            if mode == "benchmark":
                result, tables = benchmark(sample, kind)
            elif mode == "tracks":
                result, tables = track_analysis(sample)
            else:
                result, tables = bed_validation(sample)
            result["analysis"] = mode
        except (ValueError, csv.Error) as exc:
            error, status = str(exc), 400
    if result is not None:
        from experiment_store import save_experiment
        result["saved_run_id"] = save_experiment(title, mode, result, request.path,
            {"text": sample, "kind": kind}, {"tables": [
                {"title": name, "headings": headings, "rows": list(rows)} for name, headings, rows in tables]})
    page = render_template("research.html", mode=mode, title=title, description=description,
                           sample=sample, kind=kind, result=result, tables=tables, error=error)
    response = make_response(page, status)
    response.headers["Cache-Control"] = "no-store"
    return response
