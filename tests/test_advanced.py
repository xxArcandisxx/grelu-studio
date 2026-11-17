"""End-to-end checks for scientific consistency, exports, and input validation."""
import io
import json
import tempfile
import unittest
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from advanced_views import example_sequence, variant_example
from core.sequences import dinuc_shuffle
from studio_web import app


class RecordParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.capture = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        self.capture = tag == 'script' and dict(attrs).get('id') == 'run-record'

    def handle_endtag(self, tag):
        if tag == 'script':
            self.capture = False

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)


def record(response):
    parser = RecordParser()
    parser.feed(response.text)
    return json.loads(''.join(parser.parts))


class AdvancedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.previous = {key: app.config.get(key) for key in ('TESTING', 'ACCOUNT_DATABASE', 'HISTORY_DATABASE')}
        app.config.update(HISTORY_DATABASE=str(Path(self.tmp.name) / 'history.db'), TESTING=True, ACCOUNT_DATABASE=str(Path(self.tmp.name) / 'accounts.db'))
        self.client = app.test_client()

    def tearDown(self):
        app.config.update(self.previous)
        self.tmp.cleanup()

    def post(self, mode, **values):
        return self.client.post('/' + mode, data={'model': 'demo', **values})

    def test_new_pages_examples_and_model_guidance(self):
        for mode in ('design', 'batch-variants', 'mutagenesis', 'attribution'):
            with self.subTest(mode=mode):
                self.assertEqual(self.client.get('/' + mode).status_code, 200)
                response = self.post(mode, example='1')
                self.assertEqual(response.status_code, 200)
                result = record(response)
                self.assertTrue(result['ok'])
                self.assertTrue(result['demo'])
                self.assertEqual(result['model']['provenance']['model_id'], 'studio-demo-cnn-v1')
                self.assertNotIn('path', result['model'])
                self.assertNotIn('csv', result['results'])
                self.assertNotIn('fasta', result['results'])
                self.assertIn('download-csv', response.text)
        details = self.client.get('/api/model-info?model=demo').json
        self.assertEqual(details['seq_len'], 200)
        self.assertEqual(len(details['tasks']), 3)
        # A catalog reference must never silently run a different local model.
        self.assertEqual(self.client.get('/api/model-info?model=Genentech/borzoi-model').status_code, 400)
        response = self.post('design', text='ACGT')
        self.assertEqual(response.status_code, 400)
        self.assertIn('Load example', response.text)

    def test_design_history_reconstructs_output_and_objective(self):
        response = self.post('design', example='1', target='0', off_target='1', iterations='5', candidates='12', edit_start='81', edit_end='120')
        self.assertEqual(response.status_code, 200)
        result = record(response)
        design = result['results']['designs'][0]
        sequence = list(design['seed_sequence'])
        previous = design['score_before']
        for step in design['history'][1:]:
            self.assertGreater(step['score'], previous)
            previous = step['score']
            mutation = step['mutation']
            index = mutation['pos_1based'] - 1
            self.assertTrue(80 <= index <= 119)
            self.assertEqual(sequence[index], mutation['from'])
            sequence[index] = mutation['to']
        self.assertEqual(''.join(sequence), design['designed_sequence'])
        self.assertAlmostEqual(previous, design['score_after'])
        self.assertAlmostEqual(design['score_after'], design['task_scores_after']['task_0'] - design['task_scores_after']['task_1'])
        self.assertIn(design['designed_sequence'], result['results']['fasta_text'])
        self.assertEqual(len(design['designed_sequence']), 200)
        self.assertEqual(design['designed_sequence'][:80], design['seed_sequence'][:80])
        self.assertEqual(design['designed_sequence'][120:], design['seed_sequence'][120:])

    def test_batch_upload_matches_pair_comparisons_and_ranks(self):
        response = self.client.post('/batch-variants', data={
            'model': 'demo', 'file': (io.BytesIO(variant_example(200).encode()), 'variants.csv'), 'shuffles': '5'})
        self.assertEqual(response.status_code, 200)
        result = record(response)
        ranked = result['results']['ranked_variants']
        self.assertEqual([row['rank'] for row in ranked], [1, 2, 3])
        self.assertEqual([abs(row['top_effect']) for row in ranked], sorted((abs(row['top_effect']) for row in ranked), reverse=True))
        for source, variant in zip(result['inputs'], result['results']['variants'], strict=True):
            reference = source['sequence']
            offset = source['pos'] - 1
            alternate = reference[:offset] + source['alt'] + reference[offset + 1:]
            pair = self.client.post('/api/compare', json={'model': 'demo', 'reference': reference, 'alternate': alternate}).json
            ref, alt = pair['results']['predictions']
            for track, delta in variant['effects'].items():
                self.assertAlmostEqual(delta, alt['scores'][track] - ref['scores'][track])
                self.assertGreaterEqual(variant['pvalues'][track], 1 / 6)
                self.assertLessEqual(variant['pvalues'][track], 1)

    def test_scan_deltas_match_pair_and_skip_unknown_bases(self):
        sequence = example_sequence(200)
        sequence = sequence[:101] + 'N' + sequence[102:]
        response = self.post('mutagenesis', text=sequence, start='101', end='103', target='0')
        self.assertEqual(response.status_code, 200)
        result = record(response)
        self.assertEqual(result['results']['substitutions'], 6)
        self.assertEqual(result['results']['skipped_positions'], 1)
        self.assertEqual(result['results']['positions'][1]['deltas'], {})
        position = result['results']['positions'][0]
        base, delta = next(iter(position['deltas'].items()))
        alternate = sequence[:100] + base + sequence[101:]
        pair = self.client.post('/api/compare', json={'model': 'demo', 'reference': sequence, 'alternate': alternate}).json
        ref, alt = pair['results']['predictions']
        self.assertAlmostEqual(delta, alt['scores']['task_0'] - ref['scores']['task_0'])

    def test_attribution_matches_local_finite_difference(self):
        import numpy as np
        import torch
        from core.models import load_model
        from core.sequences import one_hot
        from studio_web import MODELS

        sequence = example_sequence(200)
        sequence = sequence[:100] + 'N' + sequence[101:]
        response = self.post('attribution', text=sequence, start='51', end='150', target='0', randomized_check='on')
        self.assertEqual(response.status_code, 200)
        result = record(response)
        matrix = np.array(result['results']['matrix'])
        self.assertEqual(matrix.shape, (4, 200))
        self.assertTrue(np.isfinite(matrix).all())
        self.assertEqual(float(matrix[:, 100].sum()), 0)
        self.assertEqual(len(result['results']['display_positions']), 100)
        top = result['results']['top_positions'][0]
        position = top['position'] - 1
        base = 'ACGT'.index(top['reference'])
        self.assertAlmostEqual(matrix[base, position], top['contribution'])
        adapter, _ = load_model(MODELS['demo'], 'cpu', batch_size=1)
        adapter.module.double()
        x = torch.from_numpy(one_hot([sequence], 200)).double()
        direction = torch.zeros_like(x)
        direction[0, :, position] = -.25
        direction[0, base, position] += 1
        epsilon = 1e-5
        with torch.no_grad():
            above = adapter.module(x + epsilon * direction)[0, 0].mean().item()
            below = adapter.module(x - epsilon * direction)[0, 0].mean().item()
        self.assertAlmostEqual(matrix[base, position], (above - below) / (2 * epsilon), places=6)
        correlation = result['results']['randomized_correlation']
        self.assertTrue(correlation is None or -1 <= correlation <= 1)

    def test_trained_length_mismatch_has_actionable_guidance(self):
        from studio_web import MODELS, PROVENANCE
        with patch.dict(MODELS, {'human': {'source': 'demo', 'seq_len': 2114, 'n_tasks': 1}}), patch.dict(PROVENANCE, {'human': {'model_id': 'Test trained-length model'}}):
            response = self.client.post('/predict', data={'model': 'human', 'fasta': example_sequence(200)})
            self.assertEqual(response.status_code, 400)
            self.assertIn('2,114 bases', response.text)
            self.assertIn('Load example', response.text)
            self.assertIn('choose a model that accepts your sequence length', response.text)

    def test_bad_inputs_do_not_reach_workflow_execution(self):
        valid = variant_example(200)
        cases = [
            ('design', {'text': example_sequence(200), 'target': '99'}),
            ('design', {'text': example_sequence(200), 'target': '0', 'off_target': '0'}),
            ('design', {'text': example_sequence(200), 'iterations': '1.5'}),
            ('design', {'text': example_sequence(200), 'edit_start': '100', 'edit_end': '99'}),
            ('design', {'text': 'N' * 200}),
            ('design', {'text': '>a\n' + 'A' * 200 + '\n>b\n' + 'C' * 200}),
            ('batch-variants', {'text': valid.replace('variant_2', 'variant_1')}),
            ('batch-variants', {'text': 'variant_id,sequence,pos,ref,alt\na,' + 'A' * 200 + ',201,A,C'}),
            ('batch-variants', {'text': 'variant_id,sequence,pos,ref,alt\na,' + 'A' * 200 + ',1,C,G'}),
            ('batch-variants', {'text': 'variant_id,sequence,pos,ref,alt\na,' + 'A' * 200 + ',1,A,A'}),
            ('batch-variants', {'text': 'variant_id,sequence,pos,ref,alt\na,ACGT,1,A,C'}),
            ('batch-variants', {'text': valid, 'shuffles': '100'}),
            ('attribution', {'text': example_sequence(200), 'target': '99'}),
            ('mutagenesis', {'text': example_sequence(200), 'start': '1', 'end': '25'}),
            ('mutagenesis', {'text': example_sequence(200), 'start': '100', 'end': '99'}),
        ]
        with patch('engine.execute_platform_workflow') as execute:
            for mode, values in cases:
                with self.subTest(mode=mode, values=values):
                    self.assertEqual(self.post(mode, **values).status_code, 400)
            execute.assert_not_called()

    def test_dinucleotide_shuffles_preserve_composition(self):
        import random
        for sequence in (example_sequence(200), 'AAAAACCCCCGGGGGTTTTT', 'ACGTNNNACGT', 'A', 'AC'):
            expected = Counter(zip(sequence, sequence[1:]))
            for seed in range(10):
                shuffled = dinuc_shuffle(sequence, random.Random(seed))
                self.assertEqual(Counter(shuffled), Counter(sequence))
                self.assertEqual(Counter(zip(shuffled, shuffled[1:])), expected)


if __name__ == '__main__':
    unittest.main()
