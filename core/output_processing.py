"""Explicit application prediction contract; no biological rescaling is inferred."""
import numpy as np


def prediction_tensor(values, count=None, task_names=None):
    array = np.asarray(values)
    if array.ndim != 3 or any(dimension <= 0 for dimension in array.shape):
        raise ValueError('Prediction output must be a nonempty (sequence, task, output position) tensor.')
    if count is not None and array.shape[0] != count:
        raise ValueError('Prediction output sequence count does not match the input records.')
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError('Prediction output must contain only finite numeric values; NaN/Infinity are invalid.')
    if task_names is not None:
        if (not isinstance(task_names, (list, tuple)) or not task_names
                or any(not isinstance(name, str) or not name.strip() for name in task_names)
                or len(set(task_names)) != len(task_names) or len(task_names) != array.shape[1]):
            raise ValueError('Prediction task metadata is missing, duplicated, or incompatible with the output tensor.')
    return array


def predict_checked(adapter, sequences):
    if not sequences:
        raise ValueError('At least one input sequence is required for predictions.')
    names = getattr(adapter, 'task_names', None)
    if names is None:
        raise ValueError('Prediction task metadata is unavailable; output compatibility is unknown.')
    return prediction_tensor(adapter.predict(sequences), len(sequences), names)


def output_semantics(values):
    array = prediction_tensor(values)
    return {'label': 'Model prediction score', 'units': 'not established',
            'scale': 'native model output; no probability or logarithm conversion assumed',
            'output_positions': array.shape[2],
            'score_operation': 'pass through single native output' if array.shape[2] == 1 else 'arithmetic mean across model output positions',
            'chart_operation': 'native values; optional mean downsampling for multiple positions; display formatting only',
            'spatial_mapping': 'not established'}
