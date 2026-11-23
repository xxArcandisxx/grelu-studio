"""Local worker jobs with real status, metrics, and session ownership."""
import json
import os
import secrets
import threading
import time
import math

from flask import current_app, session
from model_registry import lab_root
from core.export_values import json_value

lock = threading.RLock()
workers = {}


class JobStateError(ValueError):
    pass


def _write_state(path,state):
    temporary=path.with_suffix('.tmp')
    try:
        temporary.write_text(json.dumps(json_value(state),allow_nan=False))
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()


def owner():
    if 'lab_owner' not in session:
        session['lab_owner'] = secrets.token_hex(20)
    return session['lab_owner']


def job_path(job_id,create=False):
    if len(job_id) != 32 or any(c not in '0123456789abcdef' for c in job_id):
        raise ValueError('Invalid run identifier.')
    folder = lab_root() / 'jobs' / job_id
    if create:folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_job(job_id):
    with lock:
        path = job_path(job_id) / 'status.json'
        try:
            def invalid(value):raise ValueError('Nonstandard JSON constant')
            state = json.loads(path.read_text(),parse_constant=invalid) if path.exists() else None
            if state is not None:json_value(state)
            if state is not None and (not isinstance(state,dict) or state.get('id')!=job_id
                    or not isinstance(state.get('owner'),str) or not state['owner']
                    or state.get('state') not in ('queued','running','complete','failed')
                    or not isinstance(state.get('history'),list)
                    or not all(isinstance(state.get(k),(int,float)) and math.isfinite(state[k]) for k in ('created','updated'))
                    or (state['state']=='complete' and not isinstance(state.get('result'),dict))):
                raise ValueError('Invalid job state schema')
        except (ValueError,TypeError) as error:
            raise JobStateError('Stored job state is invalid; original data was retained.') from error
        if state and state['state'] in ('queued','running') and state.get('worker_pid'):
            try:
                os.kill(state['worker_pid'],0)
            except ProcessLookupError:
                state.update(state='failed',message='The worker stopped before this run finished. Start a new run with the same inputs.',updated=time.time())
                _write_state(path,state)
        return state


def update_job(job_id, **changes):
    with lock:
        state = read_job(job_id)
        if state is None:raise JobStateError('No job exists for this identifier.')
        state.update(changes, updated=time.time())
        path = job_path(job_id) / 'status.json'
        _write_state(path,state)
        return state


def submit_job(kind, task, replay=None):
    app = current_app._get_current_object()
    from experiment_store import current_owner
    archive_owner = current_owner()
    root=lab_root()/'jobs';root.mkdir(exist_ok=True)
    for attempt in range(8):
        job_id=secrets.token_hex(16)
        try:(root/job_id).mkdir()
        except FileExistsError:continue
        break
    else:raise JobStateError('Could not allocate a distinct job identifier; no existing run was overwritten.')
    state = {'id': job_id, 'owner': owner(), 'kind': kind, 'state': 'queued', 'message': 'Waiting for the compute worker.',
             'worker_pid':os.getpid(), 'history': [], 'created': time.time(), 'updated': time.time(), 'result': None}
    try:_write_state(job_path(job_id) / 'status.json',state)
    except Exception:
        folder=job_path(job_id)
        if folder.exists() and not any(folder.iterdir()):folder.rmdir()
        raise
    def work():
        with app.app_context():
            saved_id=None
            try:
                update_job(job_id, state='running', message='Preparing inputs.')
                result = task(job_id)
                from experiment_store import save_experiment,set_experiment_status
                saved_id = save_experiment(kind, (replay or {}).get('path','/model-installation').strip('/'),
                    result, (replay or {}).get('path',''), (replay or {}).get('form',{}), owner_id=archive_owner,snapshot_status='pending')
                result['saved_run_id']=saved_id
                update_job(job_id, state='complete', message='Run complete. Results are ready.', result=result)
                set_experiment_status(saved_id,'complete',archive_owner)
            except Exception as exc:
                app.logger.exception('Lab job %s failed', job_id)
                cleanup_failed=False
                if saved_id:
                    try:
                        from experiment_store import delete_experiment
                        delete_experiment(saved_id,owner_id=archive_owner)
                    except Exception:
                        cleanup_failed=True
                        app.logger.exception('Could not remove incomplete job archive %s',saved_id)
                update_job(job_id, state='failed',result=None,
                    message='The worker could not complete this run. Check server logs; no successful job result is available.',
                    archive_cleanup_pending=cleanup_failed)
            finally:
                with lock:
                    workers.pop(job_id, None)
    with lock:
        # Keep local CPU work serial; inference uses the same lock inside each task.
        thread = threading.Thread(target=work, daemon=True, name=f'grelu-{job_id[:8]}')
        workers[job_id] = thread
    thread.start()
    return job_id
