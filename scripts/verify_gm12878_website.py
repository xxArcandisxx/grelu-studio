"""Check real benchmark inputs through isolated Flask requests (no browser)."""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_gm12878 import DATA, REVISION, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'test-results/gm12878-biological')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    import torch
    torch.set_num_threads(2)
    os.environ.update(GRELU_HUMAN_CHECKPOINT=hf_hub_download(
        'Genentech/GM12878_dnase-model', 'model.ckpt', revision=REVISION, local_files_only=True),
        GRELU_HUMAN_INPUT_LENGTH='2114', GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from studio_web import app
    from tests.run_gm12878_sample import record
    state = args.output / 'website-state'
    state.mkdir(parents=True, exist_ok=True)
    app.config.update(TESTING=True, ACCOUNT_DATABASE=str(state/'accounts.db'),
                      HISTORY_DATABASE=str(state/'history.db'), LAB_ROOT=str(state/'lab'))
    app.secret_key = 'isolated-biological-benchmark-verification'
    client = app.test_client()
    client.get('/experiments')
    with client.session_transaction() as session:
        csrf = session['csrf_token']
    checks = {}
    regions = list(csv.DictReader((DATA/'dnase.csv').open()))[:2]
    fasta = ''.join(f">{r['id']}\n{r['sequence']}\n" for r in regions)
    response = client.post('/predict', data={'csrf_token': csrf, 'model': 'human', 'fasta': fasta})
    result = record(response)
    expected = list(csv.DictReader((args.output/'dnase_predictions.csv').open()))[:2]
    for row, target in zip(result['results']['predictions'], expected, strict=True):
        value = next(iter(row['scores'].values()))
        assert math.isclose(value, float(target['predicted']), rel_tol=1e-5, abs_tol=1e-5), (value, target)
    checks['prediction'] = {'status': 'passed', 'records': 2, 'matches_benchmark_predictions': True}
    (args.output/'website_prediction.html').write_text(response.text)
    variant = next(csv.DictReader((DATA/'dsqtl.csv').open()))
    response = client.post('/api/compare', json={'model': 'human',
        'reference': variant['reference_sequence'], 'alternate': variant['alternate_sequence']})
    assert response.status_code == 200, response.text
    assert response.json['ok'] and not response.json['demo']
    assert response.json['inputs']['position'] == 1057
    write_json(args.output/'website_variant.json', response.json)
    checks['variant_comparison'] = {'status': 'passed', 'published_variant': variant['id']}
    variants = list(csv.DictReader((DATA/'website-inputs/batch_variants_24.csv').open()))[:2]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=['variant_id', 'sequence', 'pos', 'ref', 'alt'])
    writer.writeheader(); writer.writerows(variants)
    response = client.post('/batch-variants', data={'csrf_token': csrf, 'model': 'human', 'shuffles': '0',
        'file': (io.BytesIO(text.getvalue().encode()), 'published_variants.csv')}, content_type='multipart/form-data')
    result = record(response)
    assert len(result['results']['ranked_variants']) == 2
    checks['batch_variants'] = {'status': 'passed', 'published_variants': 2}
    (args.output/'website_batch_variants.html').write_text(response.text)
    metrics = json.loads((args.output/'metrics.json').read_text())
    for name, kind in [('dnase', 'regression'), ('dsqtl', 'binary')]:
        response = client.post('/research/benchmark', data={'csrf_token': csrf, 'kind': kind,
            'file': (io.BytesIO((args.output/f'{name}_website_benchmark.csv').read_bytes()), f'{name}.csv')},
            content_type='multipart/form-data')
        result = record(response)
        actual = result['metrics']
        pairs = [('Pearson r', 'Pearson r'), ('Spearman rho', 'Spearman rho'), ('RMSE', 'RMSE')] if name == 'dnase' else [('AUROC', 'AUROC'), ('Average precision', 'average_precision')]
        for website_key, metric_key in pairs:
            assert math.isclose(actual[website_key], metrics[name][metric_key], rel_tol=1e-8, abs_tol=1e-8)
        checks[f'{name}_benchmark'] = {'status': 'passed', 'metrics_match_local_evaluation': True}
        (args.output/f'website_{name}_benchmark.html').write_text(response.text)
    write_json(args.output/'website_checks.json', {'mode': 'isolated Flask requests, no browser automation', 'checks': checks})
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
