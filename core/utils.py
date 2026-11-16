import sys

import numpy as np


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def card(name, status, message):
    """One evidence card for the UI's Evidence tab. status: pass | warn | info"""
    return {"name": name, "status": status, "message": message}


def to_py(o):
    """Make anything JSON-safe: numpy -> python, NaN -> None, unknown objects -> str."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return to_py(o.item())
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)