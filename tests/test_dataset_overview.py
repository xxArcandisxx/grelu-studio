"""Dataset statistics, edge cases, and file-to-report flows without model execution."""
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from core.dataset import dataset_overview
from studio_web import app


class DatasetOverviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = {key: app.config.get(key) for key in ('TESTING', 'ACCOUNT_DATABASE', 'HISTORY_DATABASE')}
        root = Path(self.tmp.name)
        app.config.update(TESTING=True, ACCOUNT_DATABASE=str(root/'accounts.db'), HISTORY_DATABASE=str(root/'history.db'))
        self.client = app.test_client()

    def tearDown(self):
        app.config.update(self.previous)
        self.tmp.cleanup()

    def record(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        match = re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>', response.text, re.S)
        self.assertIsNotNone(match)
        return json.loads(match[1])

    def test_hand_calculated_gc_duplicates_lengths_and_unknowns(self):
        data, _ = dataset_overview('>a\nACGT\n>b\nGGCCNN\n>c\nACGT\n>d\nNNN')
        summary = data['summary']
        self.assertEqual(summary['records'], 4)
        self.assertEqual(summary['total_bases'], 17)
        self.assertEqual(summary['unique_sequences'], 3)
        self.assertEqual(summary['duplicate_records'], 1)
        self.assertEqual(summary['duplicate_groups'], 1)
        self.assertEqual(summary['unknown_bases'], 5)
        self.assertEqual(summary['all_unknown_records'], 1)
        self.assertEqual((summary['length_min'], summary['length_max'], summary['length_median']), (3, 6, 4))
        self.assertAlmostEqual(summary['mean_record_gc_percent'], 200/3)
        self.assertEqual(data['base_counts'], {'A':2, 'C':4, 'G':4, 'T':2, 'N':5})
        self.assertIsNone(data['records'][-1]['gc_percent'])
        self.assertTrue(data['records'][0]['duplicate_sequence'])
        weighted, _ = dataset_overview('>a\nAAAAAA\n>b\nGC')
        self.assertEqual(weighted['summary']['mean_record_gc_percent'], 50)
        self.assertEqual(weighted['summary']['pooled_gc_percent'], 25)

    def test_labels_balance_conflicts_and_constant_targets(self):
        data, tables = dataset_overview('sequence,label\nAAAA,0\nCCCC,1\nAAAA,1\nTTTT,0\n', 'csv')
        labels = data['labels']
        self.assertTrue(labels['binary'])
        self.assertEqual(labels['class_counts'], {'0':2, '1':2})
        self.assertEqual(labels['mean'], .5)
        self.assertEqual(labels['population_sd'], .5)
        self.assertEqual(labels['conflicting_sequence_labels'], 1)
        self.assertIn('Binary label balance', [title for title, _, _ in tables])
        constant, _ = dataset_overview('sequence,label\nAAAA,2\nCCCC,2', 'csv')
        self.assertFalse(constant['labels']['binary'])
        self.assertTrue(any('identical' in note for note in constant['warnings']))

    def test_invalid_inputs_and_workload_limits(self):
        invalid = [('', 'fasta'), ('>empty', 'fasta'), ('>\nACGT', 'fasta'), ('ACGX', 'fasta'),
                   ('ACGT', 'invalid'), ('label,sequence\n0,ACGT', 'csv'),
                   ('sequence,label\nACGT,nan', 'csv'), ('sequence,label\nACGT,inf', 'csv'),
                   ('sequence,label\nACGT,1000001', 'csv'), ('sequence,label\nACGT', 'csv'),
                   ('sequence,label\nACGT,0,extra', 'csv'), ('A'*200001, 'fasta'),
                   (''.join(f'>r{i}\nA\n' for i in range(5001)), 'fasta')]
        for text, kind in invalid:
            with self.subTest(kind=kind, preview=text[:35]):
                with self.assertRaises(ValueError):
                    dataset_overview(text, kind)

    def test_histograms_preserve_counts_and_degenerate_values_are_json_safe(self):
        for text, kind in [('>all_n\nNNNN', 'fasta'), ('ACGT', 'fasta'),
                           ('sequence,label\nAAAA,0\nCCCC,5e-324', 'csv'),
                           ('sequence,label\nAAAA,-1000000\nCCCC,1000000', 'csv')]:
            data, _ = dataset_overview(text, kind)
            json.dumps(data, allow_nan=False)
            for chart in data['distributions']:
                expected = data['summary']['records']
                if chart['title'] == 'GC content per sequence':
                    expected -= data['summary']['all_unknown_records']
                self.assertEqual(sum(row['count'] for row in chart['bins']), expected)
                self.assertTrue(all(0 <= row['width'] <= 100 for row in chart['bins']))

    def test_file_examples_save_and_export_without_models_or_jobs(self):
        with patch('core.models.load_model', side_effect=AssertionError('No model should load')), \
             patch('lab_jobs.submit_job', side_effect=AssertionError('No job should start')):
            page = self.client.get('/research/dataset')
            self.assertEqual(page.status_code, 200)
            self.assertIn('Dataset overview', page.text)
            self.assertNotIn('/research/bed', page.text)
            for kind in ('fasta', 'csv'):
                example = self.client.get('/api/examples/dataset', query_string={'kind':kind}).json
                result = self.record(self.client.post('/research/dataset', data=example['fields']))
                run_id = result['saved_run_id']
                snapshot = self.client.get('/experiments/'+run_id+'/record.json').json
                self.assertEqual(snapshot['form']['kind'], kind)
                detail = self.client.get('/experiments/'+run_id)
                self.assertEqual(detail.status_code, 200)
                self.assertIn('action="/research/dataset"', detail.text)
                report = self.client.get('/experiments/'+run_id+'/report.html')
                self.assertEqual(report.status_code, 200)
                self.assertIn('Dataset summary', report.text)
                upload = io.BytesIO(('\ufeff'+example['fields']['text']).encode())
                uploaded = self.record(self.client.post('/research/dataset', data={'kind':kind,'file':(upload, 'dataset.txt')}))
                self.assertEqual(uploaded['summary'], result['summary'])

    def test_display_limit_preserves_complete_json_and_escaped_identifiers(self):
        text = ''.join(f'>r{i}\nACGT\n' for i in range(250)) + '><script>alert(1)</script>\nACGT\n'
        response = self.client.post('/research/dataset', data={'kind':'fasta', 'text':text})
        result = self.record(response)
        self.assertEqual(len(result['records']), 251)
        self.assertIn('Showing the first 200 records', response.text)
        self.assertNotIn('<script>alert(1)</script>', response.text)
        self.assertEqual(self.client.post('/research/dataset', data={'kind':'csv','text':'sequence,label\nACGT,nan'}).status_code, 400)
