"""Classify immutable Gate 3 evidence and verify investigation preservation.

Does not rerun forward tensor tests, relabel failures, or start another gate.
"""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def counts(checks):
    return {status: sum(r['status'] == status for r in checks)
            for status in ['PASS', 'FAIL', 'SKIP', 'BLOCKED']}


def main():
    original_path = BASE.parent / 'runs/20261122T142625611276Z/results.json'
    original = json.loads(original_path.read_text())
    optional = [r for r in original['checks'] if r['workflow'] == 'native optional inverse']
    required = [r for r in original['checks'] if r['workflow'] != 'native optional inverse']
    compatibility = json.loads((BASE / 'standalone-direct/results.json').read_text())
    audit = json.loads((BASE / 'usage-audit.json').read_text())
    before = json.loads((BASE / 'protected-before.json').read_text())
    after = {name: sha(ROOT / name) for name in before}
    changed = [name for name in before if before[name] != after[name]]
    source_changed = [name for name, value in audit['source_sha256'].items() if sha(ROOT / name) != value]
    save(BASE / 'protected-after.json', after)
    save(BASE / 'preservation.json', dict(protected_file_count=len(before),
         changed_files=changed, audited_source_files_changed=source_changed,
         status='PASS' if not changed and not source_changed else 'FAIL'))
    original_by_id = {r['id']: r for r in optional}
    repeated = {r['id']: r for r in compatibility['checks']}
    same = set(original_by_id) == set(repeated) and all(
        repeated[name]['status'] == old['status'] for name, old in original_by_id.items())
    valid_inputs = all(r['accepted_by_library_check'] and r['equals_native_encoder']
                       and r['dtype'] == 'torch.float32' and r['device'] == 'cpu'
                       for r in compatibility['checks'])
    safe = (not changed and not source_changed and same and valid_inputs
            and compatibility['isolation_ok'] and not audit['website_hit_count']
            and not audit['python_parse_errors'] and not counts(required)['FAIL']
            and not counts(required)['BLOCKED'])
    result = dict(
        recommendation='SAFE TO PROCEED WITH DOCUMENTED LIMITATIONS' if safe else 'BLOCK GATE 4',
        review_required_before_gate4=True, gate4_started=False,
        original_gate3=dict(path=str(original_path.relative_to(ROOT)),
                           sha256=sha(original_path), status=original['status'], counts=original['counts']),
        required_website_preprocessing=dict(counts=counts(required),
            basis='Explicit classification of original saved checks; not a new tensor-suite run. All statuses preserved.',
            checks=required),
        original_optional_compatibility=dict(counts=counts(optional), checks=optional),
        new_optional_dependency_test=dict(path='standalone-direct/results.json',
            sha256=sha(BASE / 'standalone-direct/results.json'), status=compatibility['status'],
            counts=compatibility['counts'], original_statuses_reproduced=same,
            all_inputs_supported_and_equal_native_encoder=valid_inputs),
        known_external_dependency_limitation=True,
        impact_audit='USAGE_AUDIT.md', audit_evidence='usage-audit.json',
        preservation_evidence='preservation.json',
        original_failures_remain_failures=True, tests_or_production_code_relabelled=False,
        no_model_loaded_or_run_in_investigation=True)
    save(BASE / 'results.json', result)
    print(json.dumps({k: result[k] for k in ['recommendation', 'original_gate3',
          'new_optional_dependency_test', 'gate4_started']}, indent=2))
    return 0 if safe else 1


if __name__ == '__main__':
    raise SystemExit(main())
