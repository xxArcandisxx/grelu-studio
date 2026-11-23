"""Read-only numerical evidence, CSV ID, preservation and artifact audit.

Run after verify_gate4.py. No model execution or prediction modification.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run or BASE / json.loads((BASE / 'latest.json').read_text())['run']
    import numpy as np
    results = json.loads((run / 'results.json').read_text())
    fixture = json.loads((run / 'fixtures.json').read_text())
    website = json.loads((run / 'website/manifest.json').read_text())
    direct = json.loads((run / 'direct/manifest.json').read_text())
    reference = np.load(run / 'direct/all-batch1.npy', allow_pickle=False)
    query_index = {r['sequence']: i for i, r in enumerate(fixture['reference_queries'])}
    problems = []
    raw_arrays = 0
    mappings = []
    for case in fixture['cases']:
        observed = next(c for c in website['cases'] if c['id'] == case['id'])
        if case.get('expected_error'):
            if observed.get('status') != case['expected_error']:
                problems.append(case['id'] + ': expected model failure status missing')
            continue
        for call, intended in zip(observed['calls'], case['expected_calls'], strict=True):
            offset = 0
            full_expected = reference[[query_index[r['sequence']] for r in intended]]
            for forward in call['forwards']:
                actual = np.load(run / 'website' / forward['file'], allow_pickle=False)
                expected = full_expected[offset:offset + len(actual)]
                if not np.array_equal(actual, expected):
                    problems.append(case['id'] + ': raw tensor is not exactly equal to reference')
                raw_arrays += 1
                offset += len(actual)
            if offset != len(intended):
                problems.append(case['id'] + ': count mismatch')
        if case['id'] in ['http_variant_csv', 'http_csv_fault_recovery']:
            actual_ids = [v['variant_id'] for v in observed['response_record']['results']['variants']]
            expected_ids = [r['id'].removesuffix(':reference') for r in case['expected_calls'][0][:3]]
            mappings.append(dict(case=case['id'], actual_variant_ids=actual_ids, expected_variant_ids=expected_ids,
                                 raw_comparison_evidence='comparisons/' + case['id'] + '/case.json'))
            if actual_ids != expected_ids:
                problems.append(case['id'] + ': CSV identifier mapping mismatch')
        if case.get('fault'):
            returned = observed.get('response_json') or observed.get('response_record') or observed.get('engine_outcome') or {}
            if not any('fell back' in warning for warning in returned.get('warnings', [])):
                problems.append(case['id'] + ': returned fallback warning missing')
    metrics = [json.loads(p.read_text()) for p in (run / 'comparisons').rglob('*.json')
               if 'shape_actual' in json.loads(p.read_text())]
    before = json.loads((run / 'protected-before.json').read_text())
    protected_changes = [name for name, value in before.items() if sha(ROOT / name) != value]
    if protected_changes:
        problems += ['Protected file changed: ' + name for name in protected_changes]
    missing = [p for check in results['checks'] for p in check['evidence'] if not (run / p).is_file()]
    if missing:
        problems += ['Missing check evidence: ' + name for name in missing]
    audit = dict(status='PASS' if not problems else 'FAIL', issues=problems,
        actual_raw_forward_arrays_rechecked=raw_arrays,
        all_raw_arrays_exactly_equal_to_independent_reference=not any('raw tensor' in p for p in problems),
        numerical_comparison_count=len(metrics), all_comparisons_exact=all(r['exact_equal'] for r in metrics),
        max_absolute_error=max(r['max_absolute_error'] for r in metrics),
        max_relative_error=max(r['max_relative_error'] for r in metrics),
        max_mean_absolute_error=max(r['mean_absolute_error'] for r in metrics),
        total_mismatched_elements=sum(r['mismatched_elements'] for r in metrics),
        csv_id_mappings=mappings, protected_file_count=len(before),
        protected_changes=protected_changes, missing_check_evidence=missing,
        scope='Audit of already generated actual pretrained arrays and metadata; no new inference or biological analysis.')
    save(run / 'evidence-audit.json', audit)
    # Save source evidence for native pooling and exact raw-output boundary.
    library = ROOT / '.venv/lib/python3.14/site-packages/grelu'
    sources = {}
    for filename, names in {'model/models.py': ['DilatedConvModel', 'BaseModel'],
                            'model/heads.py': ['ConvHead'], 'model/layers.py': ['AdaptivePool'],
                            'lightning/__init__.py': ['predict_on_seqs', 'format_input', 'forward'],
                            'sequence/format.py': ['strings_to_indices', 'indices_to_one_hot', 'strings_to_one_hot']}.items():
        path = library / filename
        text = path.read_text()
        tree = ast.parse(text)
        definitions = [dict(name=n.name, line=n.lineno, source=ast.get_source_segment(text, n))
                       for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        sources[filename] = dict(path=str(path), sha256=sha(path), definitions=definitions)
    save(run / 'installed-source-evidence.json', sources)
    save(run / 'model-settings-manifest.json', dict(identity=fixture['identity'],
        checkpoint_path=fixture['checkpoint_path'], checkpoint_resolved=str(Path(fixture['checkpoint_path']).resolve()),
        direct_checkpoint_sha256=direct['checkpoint_sha256'], direct_state_sha256=direct['state_sha256_after'],
        website_selected_spec=website['selected_spec'], website_provenance=website['provenance'],
        native_output_shape=['batch', 1, 1], output_axes=direct['output_axes'],
        pooling='Native DilatedConvModel → ConvHead → AdaptivePool(avg) → AdaptiveAvgPool1d(1)',
        activation=direct['activation'], transform=direct['transform'],
        input_shape=['batch', 4, 2114], channel_order=['A', 'C', 'G', 'T'],
        dtype='float32', device='cpu', versions=direct['versions'],
        tolerance=fixture['tolerance'], optimization_steps=0, training_executed=False,
        website_augmentation_calls=website['augmentation_calls'],
        settings_per_call='website/manifest.json', settings_direct='direct/manifest.json'))
    print(json.dumps(audit, indent=2))
    return 0 if not problems else 1


if __name__ == '__main__':
    raise SystemExit(main())
