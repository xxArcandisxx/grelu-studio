"""Additional browser views backed by the Studio engine."""
import random
from tempfile import TemporaryDirectory

from flask import Blueprint, abort, make_response, render_template, request

from variant_views import MODES, comparisons, validate_pairs

workbench = Blueprint("workbench", __name__)
TITLES = {"predict": "Sequence prediction", "qc": "Sequence quality control", "models": "Model inspector"}
TITLES.update({key: details[0] for key, details in MODES.items()})


def parse_batch(text, parse_sequence):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Enter DNA or a FASTA batch.")
    if not lines[0].startswith(">"):
        return [{"id": "sequence_1", "sequence": parse_sequence(text, "input")}]
    records = []
    for line in lines:
        if line.startswith(">"):
            if len(records) == 12:
                raise ValueError("Provide up to 12 FASTA records per run.")
            parts = line[1:].split(maxsplit=1)
            records.append((parts[0] if parts else f"sequence_{len(records) + 1}", []))
        else:
            records[-1][1].append(line)
    if len({name for name, _ in records}) != len(records):
        raise ValueError("Each FASTA record needs a distinct identifier.")
    return [{"id": name, "sequence": parse_sequence("".join(parts), f"record {index}")}
            for index, (name, parts) in enumerate(records, 1)]


def quality_report(records, length):
    report = []
    for record in records:
        seq = record["sequence"]
        unknown = seq.count("N")
        known = len(seq) - unknown
        gc = (seq.count("G") + seq.count("C")) / known if known else None
        report.append({"id": record["id"], "length": len(seq), "unknown": unknown, "gc_content": gc,
                       "matches_model_length": len(seq) == length})
    return report


@workbench.route("/<mode>", methods=["GET", "POST"])
def analyze(mode):
    from studio_web import MODELS, PROVENANCE, inference_lock, model_choices, parse_sequence, resolve_model

    if mode not in TITLES:
        abort(404)
    choices = model_choices()
    selected = request.form.get("model", "demo")
    model = resolve_model(selected)
    fasta = request.form.get("fasta", "")
    shuffle = "shuffle" in request.form
    columns, rows, result, error, status = [], [], None, None, 200
    if request.method == "POST":
        try:
            if model is None:
                raise ValueError("Choose an available model.")
            spec = MODELS[model]
            length = spec["seq_len"]
            if mode != "models":
                upload = request.files.get("file")
                if upload and upload.filename:
                    fasta = upload.read().decode("utf-8-sig")
                if mode not in MODES and "example" in request.form:
                    rng = random.Random(13)
                    fasta = "\n".join(f">example_{i}\n" + "".join(rng.choices("ACGT", k=length))
                                      for i in range(1, 4))
                records = parse_batch(fasta, parse_sequence)
                checks = quality_report(records, length)
            provenance = PROVENANCE[model]
            if mode == "qc":
                result = {"model": provenance, "records": checks}
                columns = ["Record", "Length", "GC", "Unknown bases", "Window"]
                rows = [[item["id"], item["length"], "—" if item["gc_content"] is None
                         else f"{item['gc_content']:.1%}", item["unknown"],
                         "Exact match" if item["matches_model_length"] else f"Expected {length:,} bp"]
                        for item in checks]
            else:
                from engine import execute_platform_workflow

                params = {"model": spec, "device": "cpu"}
                with inference_lock, TemporaryDirectory(prefix="grelu-studio-") as path:
                    if mode == "predict" or mode in MODES:
                        if any(not item["matches_model_length"] for item in checks):
                            raise ValueError(f"Each record must be {length:,} bases. Check the file in Sequence QC.")
                        if mode in MODES:
                            counts = validate_pairs(mode, records)
                            params["batch_size"] = 1
                        params.update(sequences=records, output_dir=path,
                                      n_shuffles=5 if mode == "predict" and shuffle else 0,
                                      center_bins=0, track_bins=100, seed=0)
                    outcome = execute_platform_workflow("info" if mode == "models" else "predict", params)
                if not outcome["ok"]:
                    raise RuntimeError("The engine could not load the model or finish the analysis.")
                result = outcome
                result["provenance"] = provenance
                if mode == "models":
                    card = result["results"]["model"]
                    columns = ["Property", "Value"]
                    rows = [["Backend", card["backend"]], ["Device", card["device"]],
                            ["Input length", card["seq_len"]], ["Output tracks", card["n_tasks"]],
                            ["Tracks", ", ".join(card["tasks"])],
                            ["Training assembly", card.get("train_genome", "Unavailable")]]
                else:
                    result["results"].pop("csv", None)
                    result["model"].pop("path", None)
                    result["inputs"] = records
                    result["settings"] = {"n_shuffles": params["n_shuffles"], "track_bins": 100, "seed": 0}
                    columns = ["Record", "Track", "Score", "Reverse-complement score", "GC content"]
                    if mode == "predict" and shuffle:
                        columns += ["Shuffle mean", "Score − shuffle mean"]
                    for record in result["results"]["predictions"]:
                        for track, score in record["scores"].items():
                            row = [record["id"], track, f"{score:.6g}",
                                   f"{record['reverse_complement_scores'][track]:.6g}",
                                   f"{record['gc_content']:.1%}" if "gc_content" in record else "—"]
                            if mode == "predict" and shuffle:
                                row += [f"{record[key][track]:.6g}" for key in ("shuffled_baseline", "delta_vs_baseline")]
                            rows.append(row)
                    if mode in MODES:
                        effects = comparisons(mode, result["results"]["predictions"], counts)
                        result["results"]["comparisons"] = effects
                        result["analysis"] = mode
                        columns = ["Track", "Baseline", "Alternate", "Differences", "Baseline score",
                                   "Alternate score", "Delta", "Reverse-complement delta"]
                        rows = [[item["track"], item["reference"], item["alternate"], item["differences"],
                                 f"{item['reference_score']:.6g}", f"{item['alternate_score']:.6g}",
                                 f"{item['delta']:+.6g}", f"{item['reverse_complement_delta']:+.6g}"]
                                for item in effects]
                        if mode == "prioritize":
                            columns.insert(0, "Rank")
                            for row, item in zip(rows, effects, strict=True):
                                row.insert(0, item["rank_within_track"])
        except ValueError as exc:
            error, status = str(exc), 400
        except RuntimeError as exc:
            error, status = str(exc), 500
    if result is not None:
        result["selected_model"] = selected
        result["actual_model"] = choices["local"][model]
    response = make_response(render_template("analysis.html", mode=mode, title=TITLES[mode],
                            choices=choices, selected=selected, fasta=fasta, shuffle=shuffle,
                            variant_help=MODES[mode][1] if mode in MODES else None,
                            result=result, columns=columns, rows=rows, error=error), status)
    response.headers["Cache-Control"] = "no-store"
    return response

