import sys

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def card(name, status, message):
    """One evidence card for the UI's Evidence tab. status: pass | warn | info"""
    return {"name": name, "status": status, "message": message}


def to_py(o):
    """Make numpy / NaN values JSON-safe."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and (o != o):
        return None
    return o