"""Private, durable analysis snapshots and repeatable input settings."""
import json
from pathlib import Path
import secrets
import sqlite3
import hashlib
import platform
from importlib.metadata import version, PackageNotFoundError
from datetime import datetime, timezone

from flask import current_app, session
from core.export_values import json_value


class StoredExperimentError(ValueError):
    pass


def _json(value):
    return json.dumps(json_value(value),ensure_ascii=False,allow_nan=False)


def _read_json(value):
    def invalid(constant):raise ValueError('Nonstandard JSON constant')
    result=json.loads(value,parse_constant=invalid)
    if not isinstance(result,dict):raise ValueError('Snapshot fields must be JSON objects')
    # Valid JSON exponents such as 1e309 can overflow Python float parsing to
    # infinity without invoking parse_constant. Validate every nested value.
    json_value(result)
    return result


def _fingerprint(value):
    canonical=json.dumps(json_value(value),ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _resolved_owner(db,owner_id):
    alias=db.execute('SELECT account FROM owner_aliases WHERE guest=?',(owner_id,)).fetchone()
    return alias[0] if alias else owner_id


def _metadata(payload,created):
    versions={}
    for package in ['numpy','torch','grelu','Flask','pandas']:
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]=None
    model=payload.get('model',{})
    if not isinstance(model,dict):model={}
    provenance=model.get('provenance') or payload.get('provenance') or payload.get('model_provenance') or model
    sources=provenance if isinstance(provenance,list) else [provenance] if provenance else []
    comparisons=payload.get('results',{}).get('models',[]) if isinstance(payload.get('results'),dict) else []
    models=[]
    for index,p in enumerate(sources):
        if not isinstance(p,dict):continue
        compared=comparisons[index] if index<len(comparisons) and isinstance(comparisons[index],dict) else {}
        models.append(dict(model_id=p.get('model_id'),checkpoint_sha256=p.get('checkpoint_sha256'),
            tasks=model.get('tasks') or compared.get('tasks') or ([payload['track']] if payload.get('track') else None)))
    inputs=payload.get('inputs')
    status=payload.get('status') if payload.get('status') in ('complete','partial') else 'unknown' if 'status' in payload else 'complete'
    settings=payload.get('settings') if isinstance(payload.get('settings'),dict) else {}
    return dict(schema_version=1,status=status,saved_at_utc=created,
        python=platform.python_version(),versions=versions,models=models,
        version_scope='Installed versions at snapshot save; original computation provenance remains in payload.',
        payload_sha256=_fingerprint(payload),
        genome_assembly=payload.get('assembly') or model.get('train_genome'),
        random_seed=settings.get('seed'),
        input_sha256=_fingerprint(inputs) if inputs is not None else None,
        raw_tensor_policy='Stored result snapshot, not a raw inference-tensor/checkpoint archive.',
        retention='Inputs and replay form are retained locally until explicit record deletion; checkpoints and job files are separate.')

REPEAT_PATHS = {'/variants', '/predict', '/qc', '/models', '/allele-specific', '/prioritize',
    '/multi-allelic', '/saturation', '/design', '/batch-variants', '/mutagenesis', '/attribution',
    '/motifs', '/training', '/fine-tuning', '/pwm-scan', '/motif-simulation', '/model-comparison',
    '/research/benchmark', '/research/tracks', '/research/dataset'}


def database():
    path = Path(current_app.config.get('HISTORY_DATABASE') or Path(current_app.instance_path) / 'experiments.sqlite3')
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=20)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('''CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
        workflow TEXT NOT NULL, created TEXT NOT NULL, payload TEXT NOT NULL,
        path TEXT NOT NULL, form TEXT NOT NULL, presentation TEXT NOT NULL)''')
    db.execute('CREATE TABLE IF NOT EXISTS owner_aliases (guest TEXT PRIMARY KEY, account TEXT NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS experiments_owner_created ON experiments(owner,created DESC)')
    if 'metadata' not in {row[1] for row in db.execute('PRAGMA table_info(experiments)')}:
        db.execute("ALTER TABLE experiments ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")
    db.commit()
    return db


def current_owner():
    from account_views import user
    researcher = user()
    if researcher:
        return f"user:{researcher['id']}"
    from lab_jobs import owner
    return 'guest:' + owner()


def claim_guest(token, user_id):
    if not token:
        return
    db = database()
    try:
        db.execute('INSERT OR REPLACE INTO owner_aliases VALUES (?,?)',('guest:'+token,f'user:{user_id}'))
        db.execute('UPDATE experiments SET owner=? WHERE owner=?', (f'user:{user_id}', 'guest:' + token))
        db.commit()
    finally:
        db.close()


def save_experiment(title, workflow, payload, path, form, presentation=None, owner_id=None, snapshot_status=None):
    experiment_id = secrets.token_hex(16)
    # Never persist a CSRF token, file path from a request, or an executable replay URL.
    fields = {str(key): ('on' if value is True else str(value)) for key, value in form.items()
              if key.lower() not in {'csrf_token', 'password', 'confirm_password', 'example', 'file', 'upload',
                  'output_dir', 'checkpoint_path', 'genome_fasta', 'path'} and value is not None and value is not False}
    if path not in REPEAT_PATHS:
        path, fields = '', {}
    if not isinstance(payload,dict) or payload.get('ok') is False or payload.get('status') in ('failed','error','queued','running'):
        raise ValueError('Only completed or explicitly partial result snapshots may be saved.')
    normalized=json_value(payload)
    data = _json(normalized)
    view = _json(presentation or {})
    created=datetime.now(timezone.utc).isoformat(timespec='seconds')
    metadata_value=_metadata(normalized,created)
    if snapshot_status is not None:
        if snapshot_status not in ('pending','complete','partial','failed'):raise ValueError('Unsupported snapshot status.')
        metadata_value['status']=snapshot_status
    metadata=_json(metadata_value)
    db = database()
    try:
        resolved_owner = _resolved_owner(db,owner_id or current_owner())
        db.execute('INSERT INTO experiments (id,owner,title,workflow,created,payload,path,form,presentation,metadata) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (experiment_id, resolved_owner, title, workflow,
             created, data, path, _json(fields), view,metadata))
        db.commit()
    finally:
        db.close()
    return experiment_id


def set_experiment_status(experiment_id,status,owner_id):
    if status not in ('complete','partial','failed'):raise ValueError('Unsupported snapshot status.')
    db=database()
    try:
        resolved=_resolved_owner(db,owner_id)
        row=db.execute('SELECT metadata FROM experiments WHERE id=? AND owner=?',(experiment_id,resolved)).fetchone()
        if not row:raise StoredExperimentError('No owned experiment snapshot exists.')
        metadata=_read_json(row[0]);metadata['status']=status
        db.execute('UPDATE experiments SET metadata=? WHERE id=? AND owner=?',(_json(metadata),experiment_id,resolved))
        db.commit()
    finally:db.close()


def experiment(experiment_id, owner_id=None):
    if len(experiment_id) != 32 or any(char not in '0123456789abcdef' for char in experiment_id):
        return None
    db = database()
    try:
        row = db.execute('SELECT * FROM experiments WHERE id=? AND owner=?',
                         (experiment_id, owner_id or current_owner())).fetchone()
    finally:
        db.close()
    if not row:
        return None
    result = dict(row)
    try:
        for field in ('payload', 'form', 'presentation','metadata'):
            result[field] = _read_json(result[field])
        if result['metadata'].get('payload_sha256') and _fingerprint(result['payload'])!=result['metadata']['payload_sha256']:
            raise ValueError('Stored payload fingerprint mismatch')
    except (ValueError,TypeError) as error:
        raise StoredExperimentError('Stored experiment data is invalid; the original record was retained.') from error
    result.pop('owner', None)
    return result


def list_experiments(query='', workflow='', page=1):
    owner_id = current_owner()
    conditions, params = ['owner=?'], [owner_id]
    if query:
        conditions.append('(title LIKE ? OR workflow LIKE ?)')
        params.extend(['%' + query + '%'] * 2)
    if workflow:
        conditions.append('workflow=?')
        params.append(workflow)
    clause = ' AND '.join(conditions)
    db = database()
    try:
        total = db.execute('SELECT COUNT(*) FROM experiments WHERE ' + clause, params).fetchone()[0]
        rows = db.execute('SELECT id,title,workflow,created,metadata,payload FROM experiments WHERE ' + clause +
                          ' ORDER BY created DESC,rowid DESC LIMIT 30 OFFSET ?', [*params, (page - 1) * 30]).fetchall()
        workflows = [row[0] for row in db.execute('SELECT DISTINCT workflow FROM experiments WHERE owner=? ORDER BY workflow', (owner_id,))]
    finally:
        db.close()
    entries=[]
    for row in rows:
        entry=dict(row)
        raw_metadata=entry.pop('metadata');raw_payload=entry.pop('payload')
        try:
            metadata=_read_json(raw_metadata);payload=_read_json(raw_payload)
            if metadata.get('payload_sha256') and _fingerprint(payload)!=metadata['payload_sha256']:
                raise ValueError('Stored payload fingerprint mismatch')
            entry['status']=metadata.get('status','unknown (legacy snapshot)')
        except (ValueError,TypeError):entry['status']='unreadable snapshot data'
        entries.append(entry)
    return entries, total, workflows


def delete_experiment(experiment_id,owner_id=None):
    db = database()
    try:
        cursor = db.execute('DELETE FROM experiments WHERE id=? AND owner=?', (experiment_id, _resolved_owner(db,owner_id or current_owner())))
        db.commit()
        return bool(cursor.rowcount)
    finally:
        db.close()
