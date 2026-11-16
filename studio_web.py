"""Browser interface for reference and alternate regulatory predictions."""
import hashlib
import os
import random
import re
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

from flask import Flask, jsonify, render_template, request

from curated_scan import curated_scan
from research_views import research
from workbench import workbench

app = Flask(__name__)
app.register_blueprint(workbench)
app.register_blueprint(curated_scan)
app.register_blueprint(research)
app.config["MAX_CONTENT_LENGTH"] = 600_000
inference_lock = Lock()

MODELS = {"demo": {"source": "demo", "seq_len": 200, "n_tasks": 3, "seed": 0}}
checkpoint = os.environ.get("GRELU_HUMAN_CHECKPOINT")
PROVENANCE = {"demo": {"model_id": "studio-demo-cnn-v1", "torch_version": version("torch")}}
if checkpoint:
    length = int(os.environ["GRELU_HUMAN_INPUT_LENGTH"])
    if not 1 <= length <= 200_000:
        raise ValueError("GRELU_HUMAN_INPUT_LENGTH must be between 1 and 200,000 bases.")
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    with checkpoint_path.open("rb") as model_file:
        digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    MODELS["human"] = {"source": "checkpoint", "path": str(checkpoint_path), "seq_len": length}
    PROVENANCE["human"] = {
        "model_id": os.environ.get("GRELU_HUMAN_MODEL_ID", "Lab human regulatory model"),
        "checkpoint_sha256": digest,
        "grelu_version": version("grelu"),
        "torch_version": version("torch"),
    }


ZOO_MODELS = (
    "Genentech/human-atac-catlas-model",
    "Genentech/human-chromhmm-fullstack-model",
    "Genentech/enformer-model",
    "Genentech/decima-model",
    "Genentech/borzoi-model",
    "Genentech/GM12878_dnase-model",
    "Genentech/human-mpra-gosai-2024-model",
    "Genentech/test-model",
)


def resolve_model(selection):
    if isinstance(selection, str):
        if selection in MODELS:
            return selection
        if selection in ZOO_MODELS:
            return "human" if "human" in MODELS else "demo"
    return None


def model_choices():
    active = resolve_model(ZOO_MODELS[0])
    return {
        "local": {key: "Demo CNN · random weights" if key == "demo"
                  else info["model_id"] for key, info in PROVENANCE.items()},
        "zoo": ZOO_MODELS,
        "actual_label": PROVENANCE[active]["model_id"] if active != "demo" else "Demo CNN",
        "actual_length": MODELS[active]["seq_len"],
    }


def parse_sequence(raw, label):
    if not isinstance(raw, str):
        raise ValueError(f"Provide a {label} sequence.")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    sequence = "".join(lines).upper()
    if not re.fullmatch(r"[ACGTN]+", sequence) or len(sequence) > 200_000:
        raise ValueError(f"Enter one {label} sequence of up to 200,000 bases using A, C, G, T or N.")
    return sequence


@app.get("/")
@app.get("/variants")
def index():
    return render_template("studio.html", mode="variants", choices=model_choices(), selected="demo")


@app.get("/api/example")
def example():
    model = resolve_model(request.args.get("model", "demo"))
    if model is None:
        return jsonify(error="Choose an available model."), 400
    length = MODELS[model]["seq_len"]
    position = length // 2
    rng = random.Random(9)
    ref = "".join(rng.choices("ACGT", k=length))
    alt_base = "A" if ref[position] != "A" else "C"
    return {"reference": ref, "alternate": ref[:position] + alt_base + ref[position + 1:]}


@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object."), 400
    selection = body.get("model")
    model = resolve_model(selection)
    if model is None:
        return jsonify(error="Choose an available model."), 400
    try:
        ref = parse_sequence(body.get("reference"), "reference")
        alt = parse_sequence(body.get("alternate"), "alternate")
        if len(ref) != len(alt):
            raise ValueError("The reference and alternate must have the same length.")
        differences = [i for i, (a, b) in enumerate(zip(ref, alt)) if a != b]
        if len(differences) != 1 or "N" in (ref[differences[0]], alt[differences[0]]):
            raise ValueError("Provide exactly one substitution between A, C, G and T.")
        offset = differences[0]
        expected = MODELS[model]["seq_len"]
        if model == "human" and expected and len(ref) != expected:
            raise ValueError(f"Enter the model's full {expected:,}-base human regulatory window.")
        if model == "demo" and len(ref) > expected:
            raise ValueError(f"The demo accepts windows up to {expected} bases.")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    from engine import execute_platform_workflow

    with inference_lock, TemporaryDirectory(prefix="grelu-studio-") as path:
        outcome = execute_platform_workflow("predict", {
            "model": MODELS[model],
            "sequences": [{"id": "reference", "sequence": ref},
                          {"id": "alternate", "sequence": alt}],
            "device": "cpu", "output_dir": path, "center_bins": 0,
            "n_shuffles": 0, "track_bins": 100, "seed": 0,
        })
    if not outcome["ok"]:
        app.logger.error("Prediction failed for model %s", model)
        return jsonify(error="Prediction failed. Check the model installation on the server."), 500
    outcome["results"].pop("csv", None)
    outcome["model"].pop("path", None)
    outcome["inputs"] = {"reference": ref, "alternate": alt,
                         "position": offset + 1, "substitution": ref[offset] + ">" + alt[offset]}
    outcome["settings"] = {"aggregation": "Mean across output bins", "track_bins": 100, "seed": 0}
    outcome["model"]["provenance"] = PROVENANCE[model]
    outcome["demo"] = model == "demo"
    outcome["selected_model"] = selection
    outcome["model"]["catalog_selection"] = selection in ZOO_MODELS
    response = jsonify(outcome)
    response.headers["Cache-Control"] = "no-store"
    return response

