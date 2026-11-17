"""Persistent, workspace-managed model registrations."""
import hashlib
import json
from pathlib import Path
from threading import RLock

from flask import current_app

registry_lock = RLock()


def lab_root():
    path = Path(current_app.config.get('LAB_ROOT') or Path(current_app.instance_path) / 'lab')
    path.mkdir(parents=True, exist_ok=True)
    return path


def registrations():
    path = lab_root() / 'models.json'
    with registry_lock:
        return json.loads(path.read_text()) if path.exists() else {}


def save_registration(key, spec, label, origin):
    from importlib.metadata import version
    with Path(spec['path']).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    entry = {'spec': spec, 'provenance': {'model_id': label, 'checkpoint_sha256': digest,
             'torch_version': version('torch'), 'origin': origin}}
    with registry_lock:
        entries = registrations()
        entries[key] = entry
        path = lab_root() / 'models.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(entries))
        temporary.replace(path)
    return key


def remove_registration(key):
    with registry_lock:
        entries = registrations()
        if key not in entries:
            raise ValueError('Choose a workspace-managed model.')
        del entries[key]
        path = lab_root() / 'models.json'
        path.write_text(json.dumps(entries))


def sync_models():
    from studio_web import MODELS, PROVENANCE
    with registry_lock:
        entries = registrations()
        for key in list(MODELS):
            if key.startswith('local-') and key not in entries:
                MODELS.pop(key, None)
                PROVENANCE.pop(key, None)
        for key, value in entries.items():
            MODELS[key] = value['spec']
            PROVENANCE[key] = value['provenance']
