"""Optional local researcher accounts; analysis tools remain available to guests."""
import hashlib
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

from flask import Blueprint, current_app, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

accounts = Blueprint('accounts', __name__)


def database():
    if 'account_db' not in g:
        path = Path(current_app.config.get('ACCOUNT_DATABASE', Path(current_app.instance_path) / 'accounts.sqlite3'))
        path.parent.mkdir(parents=True, exist_ok=True)
        g.account_db = sqlite3.connect(path)
        g.account_db.row_factory = sqlite3.Row
        g.account_db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
                session_version INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS account_tokens (
                digest TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                purpose TEXT NOT NULL, expires INTEGER NOT NULL);
        ''')
    return g.account_db


@accounts.teardown_app_request
def close_database(error):
    db = g.pop('account_db', None)
    if db is not None:
        db.close()


def user():
    uid = session.get('user_id')
    if not uid:
        return None
    return database().execute('SELECT * FROM users WHERE id=? AND active=1 AND session_version=?',
                              (uid, session.get('session_version', 0))).fetchone()


@accounts.app_context_processor
def account_context():
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return {'current_user': user(), 'csrf_token': session['csrf_token']}


def issue_token(uid, purpose):
    token = secrets.token_urlsafe(32)
    db = database()
    db.execute('DELETE FROM account_tokens WHERE user_id=? AND purpose=?', (uid, purpose))
    db.execute('INSERT INTO account_tokens VALUES (?, ?, ?, ?)',
               (hashlib.sha256(token.encode()).hexdigest(), uid, purpose, int(time.time()) + 3600))
    db.commit()
    return token


@accounts.route('/account', methods=['GET', 'POST'])
@accounts.route('/account/<page>', methods=['GET', 'POST'])
def account(page='profile'):
    allowed = {'profile', 'register', 'login', 'activate', 'forgot', 'reset', 'logout'}
    if page not in allowed:
        from flask import abort
        abort(404)
    if page == 'profile' and not user():
        return redirect(url_for('accounts.account', page='login'))
    error = None
    message = None
    preview_link = None
    state = None
    status = 200
    if page == 'logout' and request.method == 'GET':
        return redirect(url_for('accounts.account'))
    if request.method == 'POST':
        csrf = session.get('csrf_token', '')
        if not csrf or not secrets.compare_digest(csrf, request.form.get('csrf_token', '')):
            return render_template('account.html', page=page, error='Your form expired. Please try again.'), 400
        db = database()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        try:
            if page == 'register':
                name = request.form.get('name', '').strip()
                if not 1 <= len(name) <= 80 or len(email) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
                    raise ValueError('Enter your name and a valid email address.')
                if not 10 <= len(password) <= 256:
                    raise ValueError('Use a password between 10 and 256 characters.')
                try:
                    cursor = db.execute('INSERT INTO users(name,email,password) VALUES (?,?,?)',
                                        (name, email, generate_password_hash(password)))
                    db.commit()
                except sqlite3.IntegrityError:
                    db.rollback()
                    raise ValueError('An account already uses this email. Sign in or reset your password.') from None
                token = issue_token(cursor.lastrowid, 'activate')
                session['pending_account'] = cursor.lastrowid
                preview_link = url_for('accounts.account', page='activate', token=token)
                state = 'pending'
            elif page == 'login':
                record = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
                # Hash even unknown passwords to avoid a cheap account-existence timing signal.
                stored_hash = record['password'] if record else generate_password_hash(secrets.token_urlsafe(32))
                if not check_password_hash(stored_hash, password):
                    raise ValueError('Email or password is incorrect.')
                if not record or not record['active']:
                    raise ValueError('Activate your account before signing in. Use the activation page to create a new link.')
                session.clear()
                session['user_id'] = record['id']
                session['session_version'] = record['session_version']
                return redirect(url_for('accounts.account'))
            elif page in {'activate', 'reset'}:
                token = request.form.get('token', '')
                purpose = 'activate' if page == 'activate' else 'reset'
                digest = hashlib.sha256(token.encode()).hexdigest()
                record = db.execute('SELECT * FROM account_tokens WHERE digest=? AND purpose=? AND expires>?',
                                    (digest, purpose, time.time())).fetchone()
                if not record:
                    raise ValueError('This link is invalid or has expired. Request a new link below.')
                if page == 'reset':
                    if not 10 <= len(password) <= 256:
                        raise ValueError('Use a password between 10 and 256 characters.')
                    if password != request.form.get('confirm_password'):
                        raise ValueError('The passwords do not match.')
                    db.execute('UPDATE users SET password=?, session_version=session_version+1 WHERE id=?',
                               (generate_password_hash(password), record['user_id']))
                else:
                    db.execute('UPDATE users SET active=1 WHERE id=?', (record['user_id'],))
                db.execute('DELETE FROM account_tokens WHERE user_id=? AND purpose=?', (record['user_id'], purpose))
                db.commit()
                state = 'success'
            elif page == 'forgot':
                record = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
                message = 'If an account matches this email, a recovery link has been created.'
                if record:
                    purpose = 'reset' if record['active'] else 'activate'
                    token = issue_token(record['id'], purpose)
                    preview_link = url_for('accounts.account', page=purpose, token=token)
            elif page == 'logout':
                session.clear()
                return redirect('/')
        except ValueError as exc:
            error, status = str(exc), 400
    # Links are displayed only in explicit local development mode, never an email-delivered claim.
    if not current_app.config['ACCOUNT_LOCAL_PREVIEW']:
        preview_link = None
    response = current_app.make_response((render_template('account.html', page=page, error=error, message=message,
                               state=state, preview_link=preview_link, token=request.values.get('token', '')), status))
    response.headers['Cache-Control'] = 'no-store'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


def init_accounts(app):
    configured_key = os.environ.get('GRELU_SECRET_KEY')
    if configured_key:
        app.secret_key = configured_key
    else:
        # A private persistent local key keeps accounts signed in across process restarts.
        path = Path(app.instance_path)
        path.mkdir(parents=True, exist_ok=True)
        key_file = path / 'session.key'
        try:
            fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, 'w') as stream:
                stream.write(secrets.token_hex(32))
        app.secret_key = key_file.read_text()
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                      SESSION_COOKIE_SECURE=os.environ.get('GRELU_HTTPS') == '1',
                      ACCOUNT_LOCAL_PREVIEW=os.environ.get('GRELU_ACCOUNT_LOCAL_PREVIEW', '1') == '1')
    app.register_blueprint(accounts)
