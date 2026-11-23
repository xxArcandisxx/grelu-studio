"""Model-specific workflow windows, kept distinct from architectural flexibility.

Training metadata defines a checkpoint's reference window for Studio's existing
fixed-window workflows. It does NOT prove that its architecture rejects every
other width. Arbitrary website configuration is never a metadata substitute.
"""
from functools import lru_cache
from pathlib import Path
import operator


class ModelLengthError(ValueError):
    def __init__(self, message, contract):
        super().__init__(message)
        self.contract = {**contract, 'verified_compatibility': False, 'error': message}


def numpy_metadata():
    import numpy as np
    return [np.dtype, np.ndarray,
            (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct')] + [
        getattr(np.dtypes, name) for name in ['BoolDType', 'Float32DType', 'Float64DType',
        'Int32DType', 'Int64DType', 'ObjectDType', 'StrDType']]


def _length(value, source):
    try:
        if isinstance(value, bool):
            raise ValueError
        number = int(value) if isinstance(value, str) and value.isdecimal() else operator.index(value)
        if number <= 0:
            raise ValueError
        return number
    except (TypeError, ValueError, OverflowError):
        raise ModelLengthError(f'Input length in {source} must be a positive integer; got {value!r}.',
                               {'status': 'unknown', 'seq_len': None, 'source': source}) from None


def length_contract(data_params=None, model_params=None, *, demo_window=None):
    data, params = data_params or {}, model_params or {}
    model_type = params.get('model_type')
    architecture = ('variable' if model_type in {'ConvModel', 'DilatedConvModel'} else
                    'length_dependent' if model_type == 'ConvMLPModel' else 'unknown')
    contract = {'status': 'unknown', 'seq_len': None, 'source': None,
                'architecture_length_mode': architecture, 'workflow_length_mode': 'unknown',
                'verified_compatibility': False,
                'compatibility_scope': 'Authoritative Studio workflow window only; not general inference or arbitrary-width certification.'}
    candidates = []
    train = data.get('train') or {}
    for source, value in [('checkpoint.data_params.train.seq_len', train.get('seq_len')),
                          ('checkpoint.data_params.train_seq_len', data.get('train_seq_len'))]:
        if value is not None:
            candidates.append((source, _length(value, source)))
    # ConvMLPModel uses constructor seq_len to size its flattened head. Pooling
    # can admit nearby widths with the same flattened size; do not call it an
    # exact universal tensor-width constraint. This is its canonical window.
    if model_type == 'ConvMLPModel' and params.get('seq_len') is not None:
        candidates.append(('checkpoint.model_params.seq_len (ConvMLPModel head)',
                           _length(params['seq_len'], 'checkpoint.model_params.seq_len')))
    if demo_window is not None:
        candidates.append(('builtin demo workflow specification', _length(demo_window, 'demo.seq_len')))
        contract['architecture_length_mode'] = 'variable'
    if not candidates:
        return contract
    if len({n for _, n in candidates}) != 1:
        raise ModelLengthError(f'Conflicting native input-length metadata: {candidates}.', contract)
    return {**contract, 'status': 'known', 'seq_len': candidates[0][1],
            'source': candidates[0][0], 'workflow_length_mode': 'fixed_window',
            'verified_compatibility': True}


def validate_length(contract, configured=None, *, require_known=True):
    result = {**contract, 'configured_seq_len': configured}
    if configured is not None:
        configured = _length(configured, 'website model.seq_len')
        result['configured_seq_len'] = configured
    if result['seq_len'] is None:
        if require_known:
            raise ModelLengthError('Required input length is unknown: no authoritative workflow window '
                                   'was found in checkpoint metadata or a supported model specification. '
                                   'A website-configured length alone cannot verify compatibility.', result)
        return result
    if configured is not None and configured != result['seq_len']:
        raise ModelLengthError(f'Input length mismatch: website configured {configured:,} bases, but '
                               f"{result['source']} specifies {result['seq_len']:,} bases for this workflow. "
                               'Rejecting incompatible configuration; neither value was silently overridden.',
                               {**result, 'status': 'incompatible_configuration'})
    return result


@lru_cache(maxsize=32)
def _checkpoint_contract(path, mtime_ns, size):
    """Read trusted metadata without constructing a model or performing inference."""
    import torch
    with torch.serialization.safe_globals(numpy_metadata()):
        payload = torch.load(path, map_location='cpu', weights_only=True)
    if 'base_model' in payload:
        return contract_for_spec(payload['base_model'], require_known=False)
    return length_contract(payload.get('data_params'), payload.get('hyper_parameters', {}).get('model_params'))


def contract_for_spec(spec, *, require_known=True):
    source = spec.get('source', 'demo')
    if source == 'demo':
        contract = length_contract(demo_window=spec.get('seq_len', 200))
    elif source in {'checkpoint', 'studio_trained'}:
        path = Path(spec['path']).resolve(strict=True)
        stat = path.stat()
        contract = _checkpoint_contract(str(path), stat.st_mtime_ns, stat.st_size)
    else:
        # Remote models acquire native metadata when load_model loads them.
        # Catalog references alone cannot establish a website window.
        contract = length_contract()
    return validate_length(contract, spec.get('seq_len'), require_known=require_known)


def required_length(spec):
    return contract_for_spec(spec)['seq_len']
