"""Local worker jobs with real status, metrics, and session ownership."""
import json
import os
import secrets
import threading
import time

from flask import current_app, session
from model_registry import lab_root

lock = threading.RLock()
workers = {}


def owner():
    if 'lab_owner' not in session:
        session['lab_owner'] = secrets.token_hex(20)
    return session['lab_owner']


def job_path(job_id):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Invalid run identifier.')
    folder = lab_root() / 'jobs' / job_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_job(job_id):
    with lock:
        path = job_path(job_id) / 'status.json'
        state = json.loads(path.read_text()) if path.exists() else None
        if state and state['state'] in ('queued','running') and state.get('worker_pid'):
            try:
                os.kill(state['worker_pid'],0)
            except ProcessLookupError:
                state.update(state='failed',message='The worker stopped before this run finished. Start a new run with the same inputs.',updated=time.time())
                path.write_text(json.dumps(state,allow_nan=False))
        return state


def update_job(job_id, **changes):
    with lock:
        state = read_job(job_id)
        state.update(changes, updated=time.time())
        path = job_path(job_id) / 'status.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, allow_nan=False))
        temporary.replace(path)
        return state


def submit_job(kind, task, replay=None):
    app = current_app._get_current_object()
    from experiment_store import current_owner
    archive_owner = current_owner()
    job_id = secrets.token_hex(16)
    state = {'id': job_id, 'owner': owner(), 'kind': kind, 'state': 'queued', 'message': 'Waiting for the compute worker.',
             'worker_pid':os.getpid(), 'history': [], 'created': time.time(), 'updated': time.time(), 'result': None}
    (job_path(job_id) / 'status.json').write_text(json.dumps(state))
    def work():
        with app.app_context():
            try:
                update_job(job_id, state='running', message='Preparing inputs.')
                result = task(job_id)
                from experiment_store import save_experiment
                result['saved_run_id'] = save_experiment(kind, (replay or {}).get('path','/model-installation').strip('/'),
                    result, (replay or {}).get('path',''), (replay or {}).get('form',{}), owner_id=archive_owner)
                update_job(job_id, state='complete', message='Run complete. Results are ready.', result=result)
            except Exception as exc:
                app.logger.exception('Lab job %s failed', job_id)
                update_job(job_id, state='failed', message=str(exc))
            finally:
                with lock:
                    workers.pop(job_id, None)
    with lock:
        # Keep local CPU work serial; inference uses the same lock inside each task.
        thread = threading.Thread(target=work, daemon=True, name=f'grelu-{job_id[:8]}')
        workers[job_id] = thread
    thread.start()
    return job_id
