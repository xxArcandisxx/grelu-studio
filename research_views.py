"""Evaluation and numerical summaries for researcher-supplied CSV data."""
import csv
import io

from flask import Blueprint, abort, make_response, render_template, request
import numpy as np

from core.stats import auroc, expected_calibration_error, pearson, spearman
from core.dataset import dataset_overview

research = Blueprint("research", __name__)
PAGES = {
    "benchmark": ("Benchmark Evaluation", "Compare prediction scores against measured values. Upload CSV with observed and predicted columns."),
    "tracks": ("Track Analysis", "Compare numerical tracks across matched rows. Upload a CSV with two to sixteen numeric columns."),
    "dataset": ("Dataset Overview", "Understand your sequences before analysis: inspect lengths, GC content, duplicates, unknown bases, and numeric labels."),
}
EXAMPLES = {
    "benchmark": "observed,predicted\n0,0.15\n1,0.8\n0,0.3\n1,0.7",
    "tracks": "track_a,track_b,track_c\n1,3,9\n2,6,7\n3,9,4\n4,12,1",
    "dataset": ">region_a\nACGTACGT\n>region_b\nGGCCGGCCNN\n>region_a_repeat\nACGTACGT\n>unknown_region\nNNNN\n",
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


@research.route("/research/<mode>", methods=["GET", "POST"])
def run(mode):
    if mode not in PAGES:
        abort(404)
    title, description = PAGES[mode]
    sample = request.form.get("text", "")
    kind = request.form.get("kind", "fasta" if mode == "dataset" else "regression")
    result, tables, error, status = None, [], None, 200
    if request.method == "POST":
        try:
            upload = request.files.get("file")
            if "example" in request.form:
                sample = EXAMPLES[mode]
                if mode == "dataset":
                    kind = "fasta"
            elif upload and upload.filename:
                sample = upload.read().decode("utf-8-sig")
            if mode == "benchmark":
                result, tables = benchmark(sample, kind)
            elif mode == "dataset":
                result, tables = dataset_overview(sample, kind)
            else:
                result, tables = track_analysis(sample)
            result["analysis"] = mode
        except (ValueError, csv.Error, UnicodeError) as exc:
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
