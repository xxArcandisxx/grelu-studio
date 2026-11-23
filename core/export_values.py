"""Lossless finite JSON values and explicitly spreadsheet-safe CSV text."""
import math
import numpy as np


def json_value(value):
    if isinstance(value,np.generic):
        item=value.item()
        if isinstance(item,np.generic):raise TypeError('NumPy scalar has no supported JSON scalar representation.')
        return json_value(item)
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,dict):
        if any(not isinstance(key,str) for key in value):
            raise TypeError('Snapshot JSON dictionary keys must be strings.')
        return {key:json_value(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_value(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value):
        raise ValueError('Snapshot/export values must be finite; NaN and Infinity are not saved as null or zero.')
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f'Unsupported snapshot value type: {type(value).__name__}.')


def csv_cell(value):
    """Escape risky text only; numeric signs stay numeric. JSON retains exact IDs."""
    if isinstance(value,str) and value not in ('+','-'):
        stripped=value.lstrip()
        if value.startswith(('\t','\r','\n')) or stripped.startswith(('=','+','-','@','＝','＋','－','＠')):
            return "'"+value
    return value
