"""Input preparation checks only: no inference, training, browser, or worker jobs."""
import csv
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from example_views import WORKFLOWS
from studio_web import app, MODELS, PROVENANCE


class FormControls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.buttons=[]
        self.fields=set()
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='button' and 'data-load-example' in attrs:self.buttons.append(attrs)
        if tag in ('input','select','textarea'):
            self.fields.update(filter(None,(attrs.get('name'),attrs.get('id'))))


class ExampleLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.previous={key:app.config.get(key) for key in ('TESTING','LAB_ROOT','ACCOUNT_DATABASE','HISTORY_DATABASE')}
        root=Path(self.temporary.name)
        app.config.update(TESTING=True,LAB_ROOT=str(root/'lab'),ACCOUNT_DATABASE=str(root/'accounts.db'),HISTORY_DATABASE=str(root/'history.db'))
        self.client=app.test_client()

    def tearDown(self):
        app.config.update(self.previous)
        self.temporary.cleanup()

    def example(self,workflow,**query):
        response=self.client.get('/api/examples/'+workflow,query_string=query)
        self.assertEqual(response.status_code,200,response.text)
        return response.json

    def test_every_tool_has_non_submitting_load_and_download_controls(self):
        paths={workflow:'/'+workflow for workflow in WORKFLOWS}
        paths.update(benchmark='/research/benchmark',tracks='/research/tracks',bed='/research/bed')
        for workflow,path in sorted(paths.items()):
            with self.subTest(workflow=workflow):
                response=self.client.get(path)
                self.assertEqual(response.status_code,200)
                parser=FormControls();parser.feed(response.text)
                self.assertTrue(parser.buttons)
                self.assertTrue(all(button.get('type')=='button' for button in parser.buttons))
                self.assertTrue(all(button['data-load-example']==workflow for button in parser.buttons))
                self.assertIn('data-download-example',response.text)
                self.assertNotIn('name="example"',response.text)
                self.assertIn('/static/examples.js',response.text)
                fields=self.example(workflow)['fields']
                self.assertTrue(set(fields)<=parser.fields,(workflow,set(fields)-parser.fields))

    def test_loading_all_examples_never_loads_models_or_starts_jobs(self):
        with patch('core.models.load_model',side_effect=AssertionError('Loading samples must not load a model')), \
             patch('lab_jobs.submit_job',side_effect=AssertionError('Loading samples must not start a job')):
            for workflow in sorted(WORKFLOWS):
                with self.subTest(workflow=workflow):
                    example=self.example(workflow)
                    self.assertTrue(example['files'])
                    self.assertTrue(example['message'])
        self.assertFalse((Path(self.temporary.name)/'lab'/'jobs').exists())
        self.assertFalse((Path(self.temporary.name)/'history.db').exists())

    def test_training_and_finetuning_csv_have_correct_lengths_and_real_gc_labels(self):
        with patch.dict(MODELS,{'fixture-human':{'source':'demo','seq_len':2114,'n_tasks':1}}), \
             patch.dict(PROVENANCE,{'fixture-human':{'model_id':'Fixture model'}}):
            for workflow,query,length,count in [('training',{'length':'350'},350,40),
                    ('fine-tuning',{'model':'fixture-human'},2114,8)]:
                data=self.example(workflow,**query)
                rows=list(csv.DictReader(io.StringIO(data['fields']['text'])))
                self.assertEqual(len(rows),count)
                for row in rows:
                    self.assertEqual(len(row['sequence']),length)
                    expected=(row['sequence'].count('G')+row['sequence'].count('C'))/length
                    self.assertAlmostEqual(float(row['label']),expected)
                self.assertEqual(data['fields']['epochs'],'2' if workflow=='training' else '1')

    def test_fasta_examples_fit_selected_model_and_pairs_validate(self):
        from workbench import parse_batch
        from studio_web import parse_sequence
        from variant_views import validate_pairs
        with patch.dict(MODELS,{'fixture-human':{'source':'demo','seq_len':2114,'n_tasks':1}}), \
             patch.dict(PROVENANCE,{'fixture-human':{'model_id':'Fixture model'}}):
            for selection,length in [('demo',200),('fixture-human',2114)]:
                for workflow in ('predict','allele-specific','prioritize','multi-allelic'):
                    data=self.example(workflow,model=selection)
                    rows=parse_batch(data['fields']['fasta'],parse_sequence)
                    self.assertTrue(all(len(row['sequence'])==length for row in rows))
                    if workflow!='predict':validate_pairs(workflow,rows)
                pair=self.example('variants',model=selection)['fields']
                ref=parse_sequence(pair['reference'],'reference');alt=parse_sequence(pair['alternate'],'alternate')
                self.assertEqual(len(ref),length)
                self.assertEqual(sum(a!=b for a,b in zip(ref,alt)),1)

    def test_model_comparison_prepares_explicit_length_handling(self):
        with patch.dict(MODELS,{'fixture-human':{'source':'demo','seq_len':2114,'n_tasks':1}}), \
             patch.dict(PROVENANCE,{'fixture-human':{'model_id':'Fixture model'}}):
            data=self.example('model-comparison',model_a='demo',model_b='fixture-human')
            self.assertEqual(data['fields']['policy'],'center')
            self.assertIn('different window lengths',data['message'])
            from workbench import parse_batch
            from studio_web import parse_sequence
            self.assertTrue(all(len(row['sequence'])==2114 for row in parse_batch(data['fields']['text'],parse_sequence)))

    def test_downloads_match_loaded_examples(self):
        for workflow in ('training','variants','pwm-scan','motifs','model-zoo'):
            with self.subTest(workflow=workflow):
                data=self.example(workflow)
                response=self.client.get('/api/examples/'+workflow+'/download')
                self.assertEqual(response.status_code,200)
                self.assertIn('attachment',response.headers['Content-Disposition'])
                if len(data['files'])>1:
                    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
                        self.assertEqual(archive.namelist(),[item['name'] for item in data['files']])
                        for item in data['files']:self.assertEqual(archive.read(item['name']).decode(),item['content'])
                else:self.assertEqual(response.data.decode(),data['files'][0]['content'])
                response.close()

    def test_invalid_model_and_lengths_return_actionable_errors(self):
        for workflow,query in [('training',{'length':'999999'}),('training',{'length':'no'}),
                               ('predict',{'model':'not-installed'}),('model-zoo',{'repo':'unknown/model'})]:
            self.assertEqual(self.client.get('/api/examples/'+workflow,query_string=query).status_code,400)
        self.assertEqual(self.client.get('/api/examples/unknown').status_code,404)


if __name__=='__main__':unittest.main()
