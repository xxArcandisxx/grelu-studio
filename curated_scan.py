"""Live model interpretation on the bundled human reference loci."""
import json
import math
from pathlib import Path
from time import monotonic

from flask import Blueprint, current_app, make_response, render_template, request

curated_scan = Blueprint("curated_scan", __name__)
LOCI = json.loads(Path(__file__).with_name("curated_loci.json").read_text(encoding="utf-8"))
GM12878_SHA256 = "f266b296f49b4e97e3ac52e94594da0aaae1d467778ea58b421e1ee7c4482bea"
BASES = "ACGT"
POSITIONS = range(1053, 1061)


def scan_locus(locus, spec):
    from core.models import load_model

    started = monotonic()
    adapter, _ = load_model(spec, "cpu", batch_size=1)
    reference = locus["dna"]
    reference_score = float(adapter.predict([reference])[0, 0].mean())
    grid = []
    for position in POSITIONS:
        base = reference[position]
        substitutions = [alt for alt in BASES if alt != base]
        sequences = [reference[:position] + alt + reference[position + 1:] for alt in substitutions]
        values = adapter.predict(sequences).mean(axis=-1)
        changes = {alt: float(delta) - reference_score for alt, (delta,) in zip(substitutions, values, strict=True)}
        if not math.isfinite(reference_score) or not all(map(math.isfinite, changes.values())):
            raise ValueError("The model returned a non-finite score.")
        grid.append({"position": locus["start"] + position + 1, "reference": base, "deltas": changes})
    return {
        "assembly": "hg38", "chromosome": locus["chrom"], "track": adapter.task_names[0],
        "reference_score": reference_score, "positions": grid,
        "runtime_s": round(monotonic() - started, 2),
    }


@curated_scan.route("/saturation", methods=["GET", "POST"])
def saturation():
    from studio_web import MODELS, PROVENANCE, inference_lock

    provenance = PROVENANCE.get("human", {})
    ready = provenance.get("checkpoint_sha256") == GM12878_SHA256
    selected = request.form.get("locus", next(iter(LOCI)))
    result, error, status = None, None, 200
    if request.method == "POST":
        if (set(request.form) - {"csrf_token"}) != {"locus"} or len(request.form.getlist("locus")) != 1 or request.files or selected not in LOCI:
            error, status = "Choose one of the listed reference loci.", 400
        elif not ready:
            error, status = "Start Studio with the pinned Genentech GM12878 DNase checkpoint.", 503
        else:
            locus = LOCI[selected]
            try:
                with inference_lock:
                    result = scan_locus(locus, MODELS["human"])
            except Exception:
                current_app.logger.exception("Curated scan failed")
                error, status = "The model could not complete the scan.", 500
            if result:
                result.update(locus=selected, source=locus["source"], model=provenance)
    if result is not None:
        from experiment_store import save_experiment
        result["inputs"] = {"locus": selected, "sequence": LOCI[selected]["dna"], "chromosome": LOCI[selected]["chrom"], "start_0based": LOCI[selected]["start"]}
        result["saved_run_id"] = save_experiment("Curated saturation scan", "saturation", result,
            "/saturation", {"locus": selected})
    scale = 0.0
    if result:
        scale = max(abs(delta) for row in result["positions"] for delta in row["deltas"].values())
    response = make_response(render_template("saturation.html", mode="saturation", available=ready,
                             loci=LOCI, selected=selected, result=result, error=error,
                             bases=BASES, scale=scale), status)
    response.headers["Cache-Control"] = "no-store"
    return response
