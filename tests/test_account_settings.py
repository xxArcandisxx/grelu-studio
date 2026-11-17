"""Account edits, credential/session protection, image handling, and legacy migration."""
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from PIL import Image, PngImagePlugin
from werkzeug.security import check_password_hash, generate_password_hash

from account_views import database, issue_token
from studio_web import app


class AccountSettingsTests(unittest.TestCase):
    password = 'original secure password'

    @classmethod
    def setUpClass(cls):
        cls.password_hash = generate_password_hash(cls.password)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.previous = {key: app.config.get(key) for key in
                         ('TESTING', 'ACCOUNT_DATABASE', 'HISTORY_DATABASE', 'ACCOUNT_LOCAL_PREVIEW')}
        app.config.update(TESTING=True, ACCOUNT_DATABASE=str(self.root / 'accounts.db'),
                          HISTORY_DATABASE=str(self.root / 'history.db'), ACCOUNT_LOCAL_PREVIEW=True)
        with app.app_context():
            db = database()
            db.executemany('INSERT INTO users(name,email,password,active) VALUES (?,?,?,1)',
                          [('Researcher', 'lab@example.org', self.password_hash),
                           ('Other researcher', 'other@example.org', self.password_hash)])
            db.commit()
        self.client = app.test_client()
        self.login(self.client)

    def tearDown(self):
        app.config.update(self.previous)
        self.tmp.cleanup()

    def login(self, client, email='lab@example.org', password=None):
        client.get('/account/login')
        with client.session_transaction() as state:
            csrf = state['csrf_token']
        return client.post('/account/login', data={'csrf_token': csrf, 'email': email,
                                                  'password': password or self.password})

    def post(self, action, **fields):
        self.client.get('/account')
        with self.client.session_transaction() as state:
            fields['csrf_token'] = state['csrf_token']
        return self.client.post('/account', data={'action': action, **fields})

    def account(self):
        with app.app_context():
            return dict(database().execute('SELECT * FROM users WHERE id=1').fetchone())

    def test_settings_require_login_and_csrf(self):
        guest = app.test_client()
        self.assertEqual(guest.get('/account').status_code, 302)
        self.assertEqual(guest.post('/account', data={'action': 'name', 'name': 'Intruder'}).status_code, 302)
        page = self.client.get('/account')
        for title in ('Profile picture', 'Display name', 'Email address', 'Password'):
            self.assertIn(title, page.text)
        self.assertEqual(page.headers['Cache-Control'], 'no-store')
        self.assertEqual(self.client.post('/account', data={'action': 'name', 'name': 'Intruder'}).status_code, 400)
        self.assertEqual(self.account()['name'], 'Researcher')

    def test_name_update_is_persistent_and_escaped(self):
        self.assertEqual(self.post('name', name='  Alex <Lab>  ').status_code, 303)
        page = self.client.get('/account')
        self.assertIn('Your name has been updated.', page.text)
        self.assertIn('Alex &lt;Lab&gt;', page.text)
        self.assertEqual(self.account()['name'], 'Alex <Lab>')
        self.assertEqual(self.post('name', name='   ').status_code, 400)
        self.assertEqual(self.post('name', name='a' * 81).status_code, 400)
        self.assertEqual(self.post('unknown').status_code, 400)
        self.assertEqual(self.account()['name'], 'Alex <Lab>')

    def test_email_change_authentication_uniqueness_and_session_invalidation(self):
        other_session = app.test_client()
        self.login(other_session)
        for email, password in [('new@example.org', 'wrong'), ('invalid', self.password),
                                ('other@example.org', self.password)]:
            with self.subTest(email=email):
                self.assertEqual(self.post('email', email=email, current_password=password).status_code, 400)
                self.assertEqual(self.account()['email'], 'lab@example.org')
        with app.app_context():
            issue_token(1, 'reset')
        self.assertEqual(self.post('email', email=' NEW@EXAMPLE.ORG ', current_password=self.password).status_code, 303)
        self.assertEqual(self.account()['email'], 'new@example.org')
        self.assertEqual(self.client.get('/account').status_code, 200)
        self.assertEqual(other_session.get('/account').status_code, 302)
        with app.app_context():
            self.assertEqual(database().execute('SELECT COUNT(*) FROM account_tokens').fetchone()[0], 0)
        fresh = app.test_client()
        self.assertEqual(self.login(fresh).status_code, 400)
        self.assertEqual(self.login(fresh, email='new@example.org').status_code, 302)

    def test_password_change_revokes_other_sessions_and_old_recovery_tokens(self):
        other_session = app.test_client()
        self.login(other_session)
        new = 'a different secure password'
        for current, password, confirmation in [('wrong', new, new), (self.password, 'short', 'short'),
                                                (self.password, new, 'mismatch'),
                                                (self.password, self.password, self.password)]:
            self.assertEqual(self.post('password', current_password=current, new_password=password,
                                       confirm_password=confirmation).status_code, 400)
        self.assertTrue(check_password_hash(self.account()['password'], self.password))
        with app.app_context():
            issue_token(1, 'reset')
        with self.client.session_transaction() as state:
            state['lab_owner'] = 'retained-guest-owner'
        self.assertEqual(self.post('password', current_password=self.password, new_password=new,
                                   confirm_password=new).status_code, 303)
        saved = self.account()
        self.assertTrue(check_password_hash(saved['password'], new))
        self.assertFalse(check_password_hash(saved['password'], self.password))
        self.assertEqual(self.client.get('/account').status_code, 200)
        with self.client.session_transaction() as state:
            self.assertEqual(state['lab_owner'], 'retained-guest-owner')
        self.assertEqual(other_session.get('/account').status_code, 302)
        with app.app_context():
            self.assertEqual(database().execute('SELECT COUNT(*) FROM account_tokens').fetchone()[0], 0)
        fresh = app.test_client()
        self.assertEqual(self.login(fresh).status_code, 400)
        self.assertEqual(self.login(fresh, password=new).status_code, 302)

    def test_picture_normalization_privacy_header_and_removal(self):
        image = io.BytesIO()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('private-note', 'This metadata must not be stored')
        Image.new('RGB', (80, 40), 'green').save(image, format='PNG', pnginfo=metadata)
        image.seek(0)
        self.assertEqual(self.post('picture', picture=(image, 'portrait.png')).status_code, 303)
        avatar = self.client.get('/account/avatar/1')
        self.assertEqual(avatar.status_code, 200)
        self.assertEqual(avatar.mimetype, 'image/png')
        self.assertEqual(avatar.headers['Cache-Control'], 'private, no-store')
        self.assertEqual(avatar.headers['X-Content-Type-Options'], 'nosniff')
        with Image.open(io.BytesIO(avatar.data)) as stored:
            self.assertEqual(stored.size, (256, 256))
            self.assertNotIn('private-note', stored.info)
        self.assertGreaterEqual(self.client.get('/account').text.count('src="/account/avatar/1"'), 4)
        self.assertEqual(app.test_client().get('/account/avatar/1').status_code, 404)
        another_user = app.test_client()
        self.login(another_user, email='other@example.org')
        self.assertEqual(another_user.get('/account/avatar/1').status_code, 404)
        self.assertEqual(self.post('remove-picture').status_code, 303)
        self.assertIsNone(self.account()['avatar'])
        self.assertEqual(self.client.get('/account/avatar/1').status_code, 404)
        self.assertNotIn('src="/account/avatar/1"', self.client.get('/account').text)

    def test_invalid_or_oversized_picture_is_not_stored(self):
        for payload in (b'<svg xmlns="http://www.w3.org/2000/svg"></svg>', b'not an image', b'x' * 400_001):
            response = self.post('picture', picture=(io.BytesIO(payload), 'picture.png'))
            self.assertEqual(response.status_code, 400)
            self.assertIsNone(self.account()['avatar'])
        self.assertEqual(self.post('picture').status_code, 400)

    def test_existing_database_migrates_without_losing_account(self):
        path = self.root / 'legacy.db'
        with closing(sqlite3.connect(path)) as legacy:
            legacy.execute('''CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE, password TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
                session_version INTEGER NOT NULL DEFAULT 0)''')
            legacy.execute('INSERT INTO users VALUES (1,?,?,?,?,0)',
                           ('Existing researcher', 'legacy@example.org', self.password_hash, 1))
            legacy.commit()
        app.config['ACCOUNT_DATABASE'] = str(path)
        with app.app_context():
            db = database()
            researcher = db.execute('SELECT * FROM users WHERE id=1').fetchone()
            self.assertEqual(researcher['name'], 'Existing researcher')
            self.assertTrue(check_password_hash(researcher['password'], self.password))
            self.assertIsNone(researcher['avatar'])
        with app.app_context():
            self.assertIn('avatar', {row['name'] for row in database().execute('PRAGMA table_info(users)')})
