"""
engine.py - GRELU Studio entry point (router + CLI only).

    python engine.py --workflow vep --params params.json
  - prints ONE JSON object to stdout (logs go to stderr)
  - exit code 0 on success, 1 on failure ({"ok": false, "error": ...})

Quick test (no downloads):  python engine.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from core.sequences import BASES
from core.utils import to_py
from workflows import WORKFLOWS


def execute_platform_workflow(workflow: str, params: dict) -> dict:
    """Central task router. Never raises: always returns a JSON-able dict."""
    t0 = time.time()
    try:
        handler = WORKFLOWS.get(workflow)
        if handler is None:
            raise ValueError(f"Unknown workflow '{workflow}'. Options: {sorted(WORKFLOWS)}")
        seed = int(params.get("seed", 0))
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
        except ImportError:
            pass
        result = handler(params)
        out = {"ok": True, "workflow": workflow, **result, "runtime_s": round(time.time() - t0, 2)}
        if params.get("output_dir"):
            Path(params["output_dir"]).mkdir(parents=True, exist_ok=True)
            (Path(params["output_dir"]) / f"{workflow}_result.json").write_text(json.dumps(to_py(out), indent=2))
        return to_py(out)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "workflow": workflow, "error": str(e),
                "traceback": traceback.format_exc(), "runtime_s": round(time.time() - t0, 2)}


def selftest() -> int:
    out = Path("runs/selftest")
    out.mkdir(parents=True, exist_ok=True)
    rnd, L = random.Random(1), 200
    csv_path = out / "variants.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant_id", "sequence", "pos", "ref", "alt"])
        for i in range(6):
            seq = "".join(rnd.choice(BASES) for _ in range(L))
            pos = 101
            ref = seq[pos - 1]
            alt = rnd.choice([b for b in BASES if b != ref])
            w.writerow([f"rs_demo_{i}", seq, pos, ref, alt])

    model = {"source": "demo", "seq_len": L, "n_tasks": 3}
    r1 = execute_platform_workflow("vep", {"model": model, "variants_csv": str(csv_path),
                                           "center_bins": 21, "n_shuffles": 10,
                                           "output_dir": str(out / "vep")})
    r2 = execute_platform_workflow("design", {"model": model, "n_random": 2, "on_targets": [0],
                                              "off_targets": [1, 2], "max_iter": 15,
                                              "output_dir": str(out / "design")})
    for r in (r1, r2):
        if r["ok"]:
            print(f"[OK ] {r['workflow']:7s} device={r['device']} {r['runtime_s']}s")
            for c in r["evidence"]:
                print(f"      - [{c['status']}] {c['name']}: {c['message']}")
        else:
            print(f"[FAIL] {r['workflow']}: {r['error']}")
            print(r["traceback"])
    print(f"Outputs in: {out.resolve()}")
    return 0 if r1["ok"] and r2["ok"] else 1


def main():
    ap = argparse.ArgumentParser(description="GRELU Studio engine")
    ap.add_argument("--workflow", choices=sorted(WORKFLOWS))
    ap.add_argument("--params", help="path to params JSON")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.workflow or not args.params:
        ap.error("--workflow and --params are required (or use --selftest)")
    params = json.loads(Path(args.params).read_text())
    result = execute_platform_workflow(args.workflow, params)
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()