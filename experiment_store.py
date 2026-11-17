"""Private, durable analysis snapshots and repeatable input settings."""
import json
from pathlib import Path
import secrets
import sqlite3
from datetime import datetime, timezone

from flask import current_app, session

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


def save_experiment(title, workflow, payload, path, form, presentation=None, owner_id=None):
    experiment_id = secrets.token_hex(16)
    # Never persist a CSRF token, file path from a request, or an executable replay URL.
    fields = {str(key): ('on' if value is True else str(value)) for key, value in form.items()
              if key not in {'csrf_token', 'password', 'confirm_password', 'example'} and value is not None and value is not False}
    if path not in REPEAT_PATHS:
        path, fields = '', {}
    data = json.dumps(payload, allow_nan=False)
    view = json.dumps(presentation or {}, allow_nan=False)
    db = database()
    try:
        resolved_owner = owner_id or current_owner()
        alias = db.execute('SELECT account FROM owner_aliases WHERE guest=?',(resolved_owner,)).fetchone()
        if alias:
            resolved_owner = alias[0]
        db.execute('INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?)',
            (experiment_id, resolved_owner, title, workflow,
             datetime.now(timezone.utc).isoformat(timespec='seconds'), data, path,
             json.dumps(fields), view))
        db.commit()
    finally:
        db.close()
    return experiment_id


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
    for field in ('payload', 'form', 'presentation'):
        result[field] = json.loads(result[field])
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
        rows = db.execute('SELECT id,title,workflow,created FROM experiments WHERE ' + clause +
                          ' ORDER BY created DESC,rowid DESC LIMIT 30 OFFSET ?', [*params, (page - 1) * 30]).fetchall()
        workflows = [row[0] for row in db.execute('SELECT DISTINCT workflow FROM experiments WHERE owner=? ORDER BY workflow', (owner_id,))]
    finally:
        db.close()
    return [dict(row) for row in rows], total, workflows


def delete_experiment(experiment_id):
    db = database()
    try:
        cursor = db.execute('DELETE FROM experiments WHERE id=? AND owner=?', (experiment_id, current_owner()))
        db.commit()
        return bool(cursor.rowcount)
    finally:
        db.close()
