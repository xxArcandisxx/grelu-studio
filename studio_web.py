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

from example_views import examples
from exploration_views import exploration
from experiment_views import experiments
from lab_views import lab
from advanced_views import advanced
from account_views import init_accounts
from curated_scan import curated_scan
from research_views import research
from workbench import workbench
from core.model_lengths import ModelLengthError, contract_for_spec, required_length

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.url_defaults
def version_static_assets(endpoint, values):
    """Use a new asset URL after a stylesheet changes in the local checkout."""
    if endpoint == 'static' and 'filename' in values:
        asset = Path(app.static_folder) / values['filename']
        if asset.is_file():
            values.setdefault('v', str(asset.stat().st_mtime_ns))


init_accounts(app)
app.register_blueprint(workbench)
app.register_blueprint(curated_scan)
app.register_blueprint(research)
app.register_blueprint(advanced)
app.register_blueprint(lab)
app.register_blueprint(experiments)
app.register_blueprint(exploration)
app.register_blueprint(examples)
app.config["MAX_CONTENT_LENGTH"] = 600_000
app.config["TRAIN_BATCH_SIZE"] = int(os.environ.get("GRELU_TRAIN_BATCH_SIZE", "8"))
app.config["DISPLAY_TIMEZONE"] = os.environ.get("GRELU_DISPLAY_TIMEZONE", "Asia/Kolkata")
inference_lock = Lock()

MODELS = {"demo": {"source": "demo", "seq_len": 200, "n_tasks": 3, "seed": 0,
                   "input_policy": "allow_short_with_n_pad"}}
checkpoint = os.environ.get("GRELU_HUMAN_CHECKPOINT")
PROVENANCE = {"demo": {"model_id": "studio-demo-cnn-v1", "torch_version": version("torch")}}
if checkpoint:
    declared_length = os.environ.get("GRELU_HUMAN_INPUT_LENGTH")
    spec = {"source": "checkpoint", "path": str(Path(checkpoint).resolve(strict=True))}
    if declared_length is not None:
        spec['seq_len'] = declared_length
    contract = contract_for_spec(spec, require_known=False)
    length = contract['seq_len']
    if length is not None and not 1 <= length <= 200_000:
        raise ValueError("GRELU_HUMAN_INPUT_LENGTH must be between 1 and 200,000 bases.")
    checkpoint_path = Path(checkpoint).resolve(strict=True)
    with checkpoint_path.open("rb") as model_file:
        digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    MODELS["human"] = {**spec, "seq_len": length if length is not None else declared_length}
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
    from model_registry import sync_models
    sync_models()
    # Catalog names are references, never aliases for a different model.
    return selection if isinstance(selection, str) and selection in MODELS else None


def model_choices():
    from model_registry import sync_models
    sync_models()
    active = "human" if "human" in MODELS else "demo"
    lengths, length_errors = {}, {}
    for key, spec in MODELS.items():
        try:
            contract = contract_for_spec(spec)
            lengths[key] = contract['seq_len']
        except (ModelLengthError, OSError, KeyError) as exc:
            lengths[key] = None
            length_errors[key] = str(exc)
    return {
        "local": {key: "Compact CNN · 200 bp" if key == "demo"
                  else info["model_id"] for key, info in PROVENANCE.items()},
        "zoo": ZOO_MODELS,
        "lengths": lengths, "length_errors": length_errors,
        "actual_label": PROVENANCE[active]["model_id"] if active != "demo" else "Compact CNN",
        "actual_length": lengths[active],
    }


@app.errorhandler(ModelLengthError)
def incompatible_model_length(error):
    return jsonify(error=str(error), seq_len=None, input_length_contract=error.contract,
                   verified_compatibility=False), 400


def parse_sequence(raw, label):
    from core.dna_inputs import parse_single
    return parse_single(raw, label)


@app.route("/", methods=["GET", "POST"])
@app.route("/variants", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        from lab_views import check_csrf
        from flask import redirect
        check_csrf()
        response = app.make_response(compare())
        if response.status_code == 200:
            return redirect("/experiments/" + response.get_json()["saved_run_id"])
        return render_template("studio.html", mode="variants", choices=model_choices(),
            selected=request.form.get("model", "demo"), error=response.get_json().get("error")), response.status_code
    return render_template("studio.html", mode="variants", choices=model_choices(), selected="demo")


@app.get("/api/example")
def example():
    model = resolve_model(request.args.get("model", "demo"))
    if model is None:
        return jsonify(error="Choose an available model."), 400
    length = required_length(MODELS[model])
    position = length // 2
    rng = random.Random(9)
    ref = "".join(rng.choices("ACGT", k=length))
    alt_base = "A" if ref[position] != "A" else "C"
    return {"reference": ref, "alternate": ref[:position] + alt_base + ref[position + 1:]}


@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True)
    if body is None and request.form and request.path in ("/", "/variants"):
        body = request.form.to_dict()
    if not isinstance(body, dict):
        return jsonify(error="Send a JSON object."), 400
    extras = set(body) - {'model','reference','alternate','csrf_token'}
    if extras:
        return jsonify(error='Unsupported input fields: '+', '.join(sorted(extras))+
            '. This endpoint compares supplied sequences; genomic coordinates and assemblies are not extracted.'), 400
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
        expected = required_length(MODELS[model])
        if model != "demo" and expected and len(ref) != expected:
            raise ValueError(f"Enter the model's full {expected:,}-base model window.")
        if model == "demo" and len(ref) > expected:
            raise ValueError(f"The compact CNN accepts windows up to {expected} bases.")
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
    from experiment_store import save_experiment
    outcome["saved_run_id"] = save_experiment("Predicted regulatory effects", "variants", outcome, "/variants",
        {"model": selection, "reference": ref, "alternate": alt})
    response = jsonify(outcome)
    response.headers["Cache-Control"] = "no-store"
    return response

