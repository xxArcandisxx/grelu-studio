"""Shared model metadata and input guidance for browser workflows."""
import json
from functools import lru_cache
from pathlib import Path
from core.model_lengths import ModelLengthError, required_length


def length_error(expected, records=None):
    detail = ''
    if records:
        detail = ' Received ' + ', '.join(f"{row['id']}: {len(row['sequence']):,} bp" for row in records[:4]) + '.'
    suggestion = ' Select Load example for matching inputs, or choose a model that accepts your sequence length.'
    return f'Each record must be {expected:,} bases for the selected model.' + detail + suggestion


@lru_cache(maxsize=8)
def _metadata(spec_json, checkpoint_stamp=None):
    from core.models import load_model, model_card
    from studio_web import inference_lock

    with inference_lock:
        spec = json.loads(spec_json)
        required_length(spec)
        adapter, warnings = load_model(spec, 'cpu', batch_size=1)
        card = model_card(adapter)
    return card, warnings


def model_metadata(model):
    from studio_web import MODELS

    try:
        spec = MODELS[model]
        required_length(spec)
        stamp = None
        if spec.get('path'):
            stat = Path(spec['path']).stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
        return _metadata(json.dumps(spec, sort_keys=True), stamp)
    except ModelLengthError:
        raise
    except Exception as exc:
        raise RuntimeError('Could not load model metadata. Check the configured checkpoint and server installation.') from exc
