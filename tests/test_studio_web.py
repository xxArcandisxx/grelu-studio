"""Route, account lifecycle, and existing scientific workflow regression checks."""
import re
import tempfile
import unittest
from pathlib import Path

from account_views import database
from studio_web import app


class StudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = {k: app.config.get(k) for k in ('TESTING', 'ACCOUNT_DATABASE', 'ACCOUNT_LOCAL_PREVIEW')}
        app.config.update(TESTING=True, ACCOUNT_DATABASE=str(Path(self.tmp.name) / 'accounts.db'), ACCOUNT_LOCAL_PREVIEW=True)
        self.client = app.test_client()

    def tearDown(self):
        app.config.update(self.previous)
        self.tmp.cleanup()

    def post(self, page, **data):
        self.client.get('/account/login')
        with self.client.session_transaction() as session:
            data['csrf_token'] = session['csrf_token']
        return self.client.post('/account/' + page, data=data)

    def register(self):
        response = self.post('register', name='Researcher', email='lab@example.org', password='a strong password')
        self.assertEqual(response.status_code, 200)
        return re.search(r'href="(/account/activate\?token=[^"&]+)', response.text).group(1)

    def test_routes_and_navigation(self):
        for path in ('/', '/predict', '/qc', '/models', '/allele-specific', '/prioritize', '/multi-allelic',
                     '/saturation', '/research/benchmark', '/research/tracks', '/research/bed',
                     '/account/login', '/account/register', '/account/activate', '/account/forgot', '/account/reset'):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn('Skip to content', response.text)

    def test_full_account_lifecycle(self):
        activation = self.register()
        self.assertEqual(self.post('login', email='lab@example.org', password='a strong password').status_code, 400)
        # Following a link alone must not mutate account state.
        self.client.get(activation)
        self.assertEqual(self.post('login', email='lab@example.org', password='a strong password').status_code, 400)
        token = activation.split('token=')[1]
        self.assertIn('You’re activated', self.post('activate', token=token).text)
        self.assertEqual(self.post('activate', token=token).status_code, 400)
        self.assertEqual(self.post('login', email='lab@example.org', password='a strong password').status_code, 302)
        self.assertIn('Researcher', self.client.get('/account').text)
        recovery = self.post('forgot', email='lab@example.org')
        token = re.search(r'/account/reset\?token=([^"&]+)', recovery.text).group(1)
        response = self.post('reset', token=token, password='new strong password', confirm_password='new strong password')
        self.assertIn('Password updated', response.text)
        self.assertEqual(self.client.get('/account').status_code, 302)
        self.assertEqual(self.post('login', email='lab@example.org', password='a strong password').status_code, 400)
        self.assertEqual(self.post('login', email='lab@example.org', password='new strong password').status_code, 302)
        self.assertEqual(self.post('logout').status_code, 302)
        self.assertEqual(self.client.get('/account').status_code, 302)

    def test_expiry_csrf_and_preview_mode(self):
        activation = self.register()
        self.assertEqual(self.client.post('/account/activate', data={'token': activation.split('token=')[1]}).status_code, 400)
        with app.app_context():
            db = database()
            db.execute('UPDATE account_tokens SET expires=0')
            db.commit()
            stored = db.execute('SELECT password FROM users').fetchone()['password']
            self.assertNotEqual(stored, 'a strong password')
        self.assertEqual(self.post('activate', token=activation.split('token=')[1]).status_code, 400)
        app.config['ACCOUNT_LOCAL_PREVIEW'] = False
        self.assertNotIn('Open activation link', self.post('forgot', email='lab@example.org').text)

    def test_existing_workflows(self):
        for mode in ('benchmark', 'tracks', 'bed'):
            with self.subTest(mode=mode):
                response = self.client.post('/research/' + mode, data={'example': '1'})
                self.assertEqual(response.status_code, 200)
                self.assertIn('run-record', response.text)
        for mode in ('predict', 'models', 'allele-specific', 'prioritize', 'multi-allelic'):
            with self.subTest(mode=mode):
                response = self.client.post('/' + mode, data={'example': '1', 'model': 'demo'})
                self.assertEqual(response.status_code, 200)
                self.assertIn('run-record', response.text)
        response = self.client.post('/qc', data={'example': '1', 'model': 'demo'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('Exact match', response.text)
        example = self.client.get('/api/example?model=demo').json
        response = self.client.post('/api/compare', json={**example, 'model': 'demo'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json['results']['predictions']), 2)
        self.assertEqual(self.client.post('/api/compare', json={'model':'demo', 'reference':'AAAA', 'alternate':'AAAA'}).status_code, 400)


if __name__ == '__main__':
    unittest.main()
