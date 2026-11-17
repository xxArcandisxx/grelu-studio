"""Real local training, checkpoint reuse, motifs, and workspace isolation."""
import tempfile
import unittest
from pathlib import Path

from studio_web import app


class LabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = {key:app.config.get(key) for key in ('LAB_ROOT','ACCOUNT_DATABASE','TESTING','HISTORY_DATABASE')}
        app.config.update(HISTORY_DATABASE=str(Path(self.tmp.name)/'history.db'),TESTING=True,LAB_ROOT=str(Path(self.tmp.name)/'lab'),ACCOUNT_DATABASE=str(Path(self.tmp.name)/'accounts.db'))
        self.client = app.test_client()
        self.client.get('/training')
        with self.client.session_transaction() as state:
            self.csrf = state['csrf_token']

    def tearDown(self):
        app.config.update(self.previous)
        from model_registry import sync_models
        with app.app_context():
            sync_models()
        self.tmp.cleanup()

    def post(self,path,**data):
        return self.client.post(path,data={'csrf_token':self.csrf,**data})

    def complete(self,response):
        from lab_jobs import workers
        self.assertEqual(response.status_code,302)
        job_id = response.location.split('/')[-1]
        worker = workers.get(job_id)
        if worker:
            worker.join(timeout=30)
        result = self.client.get('/api/jobs/'+job_id).json
        self.assertEqual(result['state'],'complete',result)
        self.assertNotIn('owner',result)
        return result

    def test_training_finetuning_and_prediction_roundtrip(self):
        trained = self.complete(self.post('/training',example='1',length='100',epochs='2',learning_rate='.001'))
        key = trained['result']['model_key']
        self.assertEqual(len(trained['history']),2)
        self.assertLessEqual(trained['result']['best_validation_mse'],trained['result']['initial_validation_mse'])
        downloaded = self.client.get('/jobs/'+trained['id']+'/checkpoint')
        self.assertEqual(downloaded.status_code,200)
        downloaded.close()
        prediction = self.client.post('/predict',data={'model':key,'example':'1'})
        self.assertEqual(prediction.status_code,200)
        fine = self.complete(self.post('/fine-tuning',example='1',model=key,epochs='1',learning_rate='.0001',target='0'))
        self.assertNotEqual(fine['result']['model_key'],key)
        other = app.test_client()
        self.assertEqual(other.get('/api/jobs/'+trained['id']).status_code,404)
        self.assertEqual(self.post('/model-zoo',action='remove',key=key).status_code,200)
        self.assertEqual(self.client.get('/api/model-info?model='+key).status_code,400)

    def test_motif_coordinates_and_strands(self):
        response = self.post('/motifs',text='>region\nAAATATAAATTT',patterns='TATA:TATAAA',k='3')
        self.assertEqual(response.status_code,200)
        from test_advanced import record
        result = record(response)
        hit = next(row for row in result['hits'] if row['strand']=='+')
        self.assertEqual((hit['start'],hit['end']),(4,9))
        self.assertTrue(result['discovery'])

    def test_csrf_and_validation(self):
        self.assertEqual(self.client.post('/training',data={'example':'1'}).status_code,400)
        self.assertEqual(self.post('/training',text='sequence,label\nAAAA,1',length='100').status_code,400)
        self.assertEqual(self.post('/model-zoo',action='install',repo='unknown/untrusted').status_code,400)
        for path in ('/training','/fine-tuning','/motifs','/model-zoo'):
            self.assertEqual(self.client.get(path).status_code,200)


if __name__ == '__main__':
    unittest.main()
