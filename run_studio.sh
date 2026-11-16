#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

if [[ -z "${GRELU_HUMAN_CHECKPOINT:-}" ]]; then
  if checkpoint=$(.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import LocalEntryNotFoundError

try:
    print(hf_hub_download(
        "Genentech/GM12878_dnase-model",
        "model.ckpt",
        revision="9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e",
        local_files_only=True,
    ))
except LocalEntryNotFoundError:
    raise SystemExit(1)
PY
  ); then
    export GRELU_HUMAN_CHECKPOINT="$checkpoint"
    export GRELU_HUMAN_INPUT_LENGTH=2114
    export GRELU_HUMAN_MODEL_ID="Genentech GM12878 DNase"
  fi
fi

exec .venv/bin/python -m flask --app studio_web run --host 127.0.0.1 --port "${PORT:-5000}"
