"""Shared model metadata and input guidance for browser workflows."""
import json
from functools import lru_cache


def length_error(expected, records=None):
    detail = ''
    if records:
        detail = ' Received ' + ', '.join(f"{row['id']}: {len(row['sequence']):,} bp" for row in records[:4]) + '.'
    suggestion = ' Select Load example for matching inputs, or choose a model that accepts your sequence length.'
    return f'Each record must be {expected:,} bases for the selected model.' + detail + suggestion


@lru_cache(maxsize=8)
def _metadata(spec_json):
    from core.models import load_model, model_card
    from studio_web import inference_lock

    with inference_lock:
        adapter, warnings = load_model(json.loads(spec_json), 'cpu', batch_size=1)
        card = model_card(adapter)
    return card, warnings


def model_metadata(model):
    from studio_web import MODELS

    try:
        return _metadata(json.dumps(MODELS[model], sort_keys=True))
    except Exception as exc:
        raise RuntimeError('Could not load model metadata. Check the configured checkpoint and server installation.') from exc
