"""Reference calculations for PWM scores and end-to-end scientific/history workflows."""
import io
import json
import math
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from advanced_views import example_sequence
from core.pwm import EXAMPLE_MATRIX, parse_motifs, scan_pwm
from studio_web import app
from test_advanced import record


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.previous={key:app.config.get(key) for key in ('TESTING','LAB_ROOT','HISTORY_DATABASE','ACCOUNT_DATABASE')}
        app.config.update(TESTING=True,LAB_ROOT=str(Path(self.tmp.name)/'lab'),
            HISTORY_DATABASE=str(Path(self.tmp.name)/'history.db'),ACCOUNT_DATABASE=str(Path(self.tmp.name)/'accounts.db'))
        self.client=app.test_client()
        self.client.get('/experiments')
        with self.client.session_transaction() as state:
            self.csrf=state['csrf_token']

    def tearDown(self):
        app.config.update(self.previous)
        self.tmp.cleanup()

    def post(self,path,**data):
        return self.client.post(path,data={'csrf_token':self.csrf,**data})

    def test_pwm_matches_hand_calculation_and_both_strands(self):
        motif=parse_motifs('>test\nA [9 1 1]\nC [1 9 1]\nG [1 1 9]\nT [1 1 1]')[0]
        result=scan_pwm([{'id':'region','sequence':'ACGNNNCGT'}],[motif],threshold=.99)
        self.assertEqual([(row['start'],row['end'],row['strand']) for row in result['hits']],[(1,3,'+'),(7,9,'-')])
        expected=3*math.log2(((9/12+.1*.25)/1.1)/.25)
        for hit in result['hits']:
            self.assertAlmostEqual(hit['score'],expected)
            self.assertAlmostEqual(hit['relative_score'],1)
            self.assertEqual(hit['matched_sequence'],'ACG')
        self.assertEqual(result['motifs'][0]['skipped_unknown_windows'],5)

    def test_meme_and_jaspar_produce_equivalent_scores(self):
        jaspar=parse_motifs('>test\nA [9 1 1]\nC [1 9 1]\nG [1 1 9]\nT [1 1 1]')
        meme=parse_motifs('MEME version 5\nALPHABET= ACGT\nMOTIF test\nletter-probability matrix: alength= 4 w= 3\n0.75 0.083333333333 0.083333333333 0.083333333333\n0.083333333333 0.75 0.083333333333 0.083333333333\n0.083333333333 0.083333333333 0.75 0.083333333333')
        sequences=[{'id':'region','sequence':'ACGTTACGTCGT'}]
        first=scan_pwm(sequences,jaspar,.7)
        second=scan_pwm(sequences,meme,.7)
        self.assertEqual(len(first['hits']),len(second['hits']))
        for a,b in zip(first['hits'],second['hits'],strict=True):
            self.assertEqual((a['start'],a['strand']),(b['start'],b['strand']))
            self.assertAlmostEqual(a['score'],b['score'])

    def test_examples_save_and_export_reports(self):
        for path in ('/pwm-scan','/motif-simulation','/model-comparison'):
            with self.subTest(path=path):
                response=self.post(path,example='1')
                self.assertEqual(response.status_code,200)
                result=record(response)
                run_id=result['saved_run_id']
                self.assertEqual(self.client.get('/experiments/'+run_id).status_code,200)
                report=self.client.get('/experiments/'+run_id+'/report.html')
                self.assertEqual(report.status_code,200)
                self.assertIn('attachment',report.headers['Content-Disposition'])
                self.assertNotIn('<script src=',report.text)
                self.assertNotIn('/static/',report.text)
                self.assertIn('Complete inputs',report.text)
                data=self.client.get('/experiments/'+run_id+'/record.json').json
                self.assertEqual(data['payload']['workflow'],path[1:])

    def test_simulation_preserves_context_and_matches_prediction(self):
        sequence=example_sequence(200)
        response=self.post('/motif-simulation',model='demo',text=sequence,motif='TATAAA',positions='50',orientation='both',
            partner='GATA',gaps='0,3',controls='on',target='0')
        self.assertEqual(response.status_code,200)
        result=record(response)
        conditions=result['results']['conditions']
        self.assertEqual(len(conditions),4)
        from core.models import load_model
        from studio_web import MODELS
        adapter,_=load_model(MODELS['demo'],'cpu')
        reference=float(adapter.predict([sequence])[0,0].mean())
        for row in conditions:
            self.assertEqual(len(row['sequence']),200)
            self.assertEqual(row['sequence'][:49],sequence[:49])
            after=49+6+row['partner_gap']+4
            self.assertEqual(row['sequence'][after:],sequence[after:])
            score=float(adapter.predict([row['sequence']])[0,0].mean())
            self.assertAlmostEqual(row['score'],score)
            self.assertAlmostEqual(row['delta'],score-reference)
            self.assertAlmostEqual(row['motif_minus_control'],row['score']-row['control_score'])

    def test_comparison_explicit_transforms_and_same_model_check(self):
        sequence=example_sequence(200)
        response=self.post('/model-comparison',text=sequence,model_a='demo',model_b='demo',policy='strict')
        self.assertEqual(response.status_code,200)
        row=record(response)['results']['comparisons'][0]
        self.assertEqual(row['raw_difference'],0)
        self.assertEqual(row['transformation_a'],'Unchanged')
        self.assertEqual(self.post('/model-comparison',text=sequence[:100],model_a='demo',model_b='demo',policy='strict').status_code,400)
        response=self.post('/model-comparison',text=sequence[:100],model_a='demo',model_b='demo',policy='center')
        self.assertEqual(response.status_code,200)
        effective=record(response)['results']['effective_inputs'][0][0]
        self.assertEqual(effective['sequence'],'N'*50+sequence[:100]+'N'*50)
        self.assertEqual(effective['offset'],50)
        self.assertIn('N-padded',effective['transformation'])
        long_sequence=example_sequence(400)
        response=self.post('/model-comparison',text=long_sequence,model_a='demo',model_b='demo',policy='center')
        cropped=record(response)['results']['effective_inputs'][0][0]
        self.assertEqual(cropped['sequence'],long_sequence[100:300])
        self.assertEqual(cropped['offset'],-100)

    def test_private_history_rerun_and_safe_standalone_report(self):
        pair=self.client.get('/api/example?model=demo').json
        result=self.client.post('/api/compare',json={'model':'demo',**pair}).json
        run_id=result['saved_run_id']
        other=app.test_client()
        for ending in ('','/record.json','/report.html'):
            self.assertEqual(other.get('/experiments/'+run_id+ending).status_code,404)
        snapshot=self.client.get('/experiments/'+run_id+'/record.json').json
        self.assertEqual(self.client.post('/api/compare',data=snapshot['form']).status_code,400)
        repeated=self.post('/variants',**snapshot['form'])
        self.assertEqual(repeated.status_code,302)
        self.assertNotEqual(repeated.location,'/experiments/'+run_id)
        self.assertIn('Predicted regulatory effects',self.client.get('/experiments?q=regulatory').text)
        self.assertEqual(self.client.post('/experiments/'+run_id+'/delete').status_code,400)
        self.assertEqual(self.post('/experiments/'+run_id+'/delete').status_code,302)
        self.assertEqual(self.client.get('/experiments/'+run_id).status_code,404)

    def test_login_claims_guest_history(self):
        response=self.post('/pwm-scan',example='1')
        run_id=record(response)['saved_run_id']
        registration=self.post('/account/register',name='Researcher',email='researcher@example.org',password='long password 123')
        token=re.search(r'/account/activate\?token=([^"&]+)',registration.text)[1]
        self.assertEqual(self.post('/account/activate',token=token).status_code,200)
        self.assertEqual(self.post('/account/login',email='researcher@example.org',password='long password 123').status_code,302)
        self.client.get('/account')
        with self.client.session_transaction() as state:
            self.csrf=state['csrf_token']
        self.assertEqual(self.client.get('/experiments/'+run_id).status_code,200)
        self.post('/account/logout')
        self.assertEqual(self.client.get('/experiments/'+run_id).status_code,404)
        self.client.get('/account/login')
        with self.client.session_transaction() as state:
            self.csrf=state['csrf_token']
        self.post('/account/login',email='researcher@example.org',password='long password 123')
        self.assertEqual(self.client.get('/experiments/'+run_id).status_code,200)

    def test_history_claim_includes_background_jobs_completed_after_login(self):
        from experiment_store import claim_guest, current_owner, experiment, save_experiment
        with app.test_request_context('/'):
            guest=current_owner()
            claim_guest(guest.removeprefix('guest:'),17)
            run_id=save_experiment('Completed background run','training',{'history':[]},'/training',{},owner_id=guest)
            self.assertIsNotNone(experiment(run_id,owner_id='user:17'))
            self.assertIsNone(experiment(run_id,owner_id=guest))

    def test_report_escapes_supplied_names_and_disallows_scripts(self):
        response=self.post('/pwm-scan',text='>region<script>alert(1)</script>\nTATAAA',
            matrices=EXAMPLE_MATRIX.replace('Synthetic_AT_pattern','<script>alert(2)</script>'),threshold='0.8')
        self.assertEqual(response.status_code,200)
        run_id=record(response)['saved_run_id']
        exported=self.client.get('/experiments/'+run_id+'/report.html')
        self.assertNotIn('<script>alert(',exported.text)
        self.assertIn("default-src 'none'",exported.headers['Content-Security-Policy'])
        self.assertIn('script',exported.text)

    def test_malformed_matrices_and_invalid_run_inputs(self):
        for value in ('', '>bad\nA [1 2 3]', '>bad\nA [0 0 0]\nC [0 0 0]\nG [0 0 0]\nT [0 0 0]',
                      'MEME version 5\nALPHABET= ACGT\nMOTIF bad\nletter-probability matrix: alength=4 w=3\n.5 .5 0 0'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_motifs(value)
        invalids=[('/pwm-scan',{'example':'1','background':'0,0.5,0.25,0.25'}),
                  ('/pwm-scan',{'example':'1','threshold':'nan'}),
                  ('/motif-simulation',{'text':example_sequence(200),'positions':'199','motif':'TATAAA'}),
                  ('/motif-simulation',{'text':example_sequence(200),'positions':'1,1'}),
                  ('/model-comparison',{'example':'1','policy':'unknown'})]
        for path,values in invalids:
            with self.subTest(path=path,values=values):
                self.assertEqual(self.post(path,**values).status_code,400)


if __name__=='__main__':
    unittest.main()
