"""Gate 4 only: supervised real website and independent gReLU inference."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
RTOL, ATOL = 1e-5, 1e-6


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def protected():
    folders = ['Testing/testing-phase-1', 'test-results/gm12878-biological', 'test-results/gm12878-verified',
               'Testing/testing-phase-2/input-output/runs', 'Testing/testing-phase-2/input-output/gate-2',
               'Testing/testing-phase-2/input-output/gate-3']
    return {str(p.relative_to(ROOT)): sha(p) for name in folders for p in sorted((ROOT / name).rglob('*')) if p.is_file()}


def rc(sequence):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement[b] for b in sequence[::-1])


def fixtures():
    identity = json.loads((BASE.parent / 'fixtures/model_identity.json').read_text())
    source = BASE.parent / 'gate-3/runs/20261122T142625611276Z/fixtures.json'
    rows = json.loads(source.read_text())['records']
    dna = {r['id']: r['sequence'] for r in rows}
    first = dna['first']
    records = [dict(id=name, sequence=sequence) for name, sequence in
               [('Z', first), ('A', dna['variant']), ('N_context', dna['unknown']),
                ('copy_Z', first), ('last', dna['cropped'])]]
    cases = []

    def groups(records, reverse=False):
        result = [records]
        if reverse:
            result.append([dict(id=r['id'] + ':RC', sequence=rc(r['sequence'])) for r in records])
        return result

    def fasta(records):
        return ''.join(f">{r['id']}\n{r['sequence']}\n" for r in records)

    def add(name, workflow, expected, **kwargs):
        cases.append(dict(id=name, workflow=workflow, expected_calls=expected, **kwargs))

    for name, subset, size, backend in [
        ('adapter_single', records[:1], 1, 'grelu'),
        ('adapter_batch1', records, 1, 'grelu'),
        ('adapter_batch2', records, 2, 'grelu'),
        ('adapter_batch4', records, 4, 'grelu'),
        ('adapter_permuted', [records[i] for i in [4, 2, 0, 3, 1]], 2, 'grelu'),
        ('adapter_builtin', records, 2, 'builtin')]:
        add(name, 'shared adapter', groups(subset), kind='adapter', records=subset,
            batch_size=size, backend=backend)
    for name in ['http_fasta', 'http_repeat']:
        add(name, '/predict', groups(records, True), kind='form', url='/predict',
            fields=dict(model='human', fasta=fasta(records)))
    upload = [records[i] for i in [0, 2, 4]]
    add('http_upload', '/predict', groups(upload, True), kind='form', url='/predict',
        fields=dict(model='human'), upload=fasta(upload))
    mutants = []
    for position in [1, 1057, 2114]:
        base = first[position - 1]
        alt = {'A': 'C', 'C': 'G', 'G': 'T', 'T': 'A'}[base]
        mutant = first[:position - 1] + alt + first[position:]
        mutants.append(dict(id=f'variant_{position}', sequence=mutant, position=position, ref=base, alt=alt))
        pair = [dict(id='reference', sequence=first), dict(id='alternate', sequence=mutant)]
        add('http_variant_' + str(position), '/api/compare', groups(pair, True), kind='json',
            url='/api/compare', body=dict(model='human', reference=first, alternate=mutant))
    for mode, variant_rows in [('allele-specific', mutants[:1]), ('prioritize', mutants[:2]), ('multi-allelic', mutants[:2])]:
        pair = [dict(id='reference', sequence=first)] + [dict(id=r['id'], sequence=r['sequence']) for r in variant_rows]
        add('http_' + mode, '/' + mode, groups(pair, True), kind='form', url='/' + mode,
            fields=dict(model='human', fasta=fasta(pair)))
    csv = 'variant_id,sequence,pos,ref,alt\n' + ''.join(
        f"{r['id']},{first},{r['position']},{r['ref']},{r['alt']}\n" for r in mutants)
    refs = [dict(id=r['id'] + ':reference', sequence=first) for r in mutants]
    alts = [dict(id=r['id'] + ':alternate', sequence=r['sequence']) for r in mutants]
    add('http_variant_csv', '/batch-variants', groups(refs + alts, True), kind='form',
        url='/batch-variants', fields=dict(model='human', text=csv, shuffles='0'))
    position = 1057
    small_mutants = [dict(id=f'mutant:{position}:{base}', sequence=first[:position - 1] + base + first[position:])
                     for base in 'ACGT' if base != first[position - 1]]
    ref = [dict(id='reference', sequence=first)]
    add('http_csv_fault_recovery', '/batch-variants', groups(refs + alts, True), kind='form',
        url='/batch-variants', fields=dict(model='human', text=csv, shuffles='0'), fault='once')
    add('engine_design_fault_recovery', '/design', [ref, ref], kind='engine_design',
        records=ref, fault='once', partial='Real zero-iteration/no-background engine design regression; no optimization. Browser design candidate/output behavior remains unverified.')
    add('http_mutagenesis', '/mutagenesis', [ref, small_mutants], kind='form', url='/mutagenesis',
        fields=dict(model='human', text=first, start=str(position), end=str(position), target='0'))
    for mode in ['design', 'attribution']:
        add('http_' + mode + '_baseline', '/' + mode, [ref], kind='form', url='/' + mode,
            fields=dict(model='human', text=first, start='1057', end='1057', target='0'),
            stop_after_calls=1, partial='Only real baseline prediction; no design optimization or attribution gradients are run.')
    motif = 'TATAAA'
    motif_mutant = dict(id='motif:1:+', sequence=motif + first[len(motif):])
    add('http_motif', '/motif-simulation', [ref + [motif_mutant]], kind='form', url='/motif-simulation',
        fields=dict(model='human', text=first, motif=motif, positions='1', orientation='+', gaps='0', target='0'))
    compare = records[:2]
    add('http_model_comparison', '/model-comparison', [compare, compare], kind='form', url='/model-comparison',
        fields=dict(model_a='human', model_b='human', text=fasta(compare), policy='strict', target_a='0', target_b='0'),
        partial='Both real GM12878 selections are exercised. A distinct second verified checkpoint is unavailable; no demo is used.')
    locus_key, locus = next(iter(json.loads((ROOT / 'curated_loci.json').read_text()).items()))
    sequence = locus['dna']
    position0 = 1053
    curated_mutants = [dict(id=f'{locus_key}:{position0 + 1}:{base}', sequence=sequence[:position0] + base + sequence[position0 + 1:])
                       for base in 'ACGT' if base != sequence[position0]]
    add('http_curated_subset', '/saturation', [[dict(id=locus_key, sequence=sequence)], curated_mutants],
        kind='form', url='/saturation', fields=dict(locus=locus_key), stop_after_calls=2,
        partial='Stop after real reference and first three substitution outputs; remaining curated positions and HTTP completion are not exercised.')
    pair = [dict(id='reference', sequence=first), dict(id='alternate', sequence=mutants[0]['sequence'])]
    add('http_fault_recovery', 'exception recovery', groups(pair, True), kind='json', url='/api/compare',
        body=dict(model='human', reference=first, alternate=mutants[0]['sequence']), fault='once')
    add('http_persistent_fault', 'exception recovery', [], kind='json', url='/api/compare',
        body=dict(model='human', reference=first, alternate=mutants[0]['sequence']), fault='persistent', expected_error=500)
    queries = {}
    for case in cases:
        for group in case['expected_calls']:
            for record in group:
                if record['sequence'] not in queries:
                    queries[record['sequence']] = dict(id='q' + str(len(queries)).zfill(3), sequence=record['sequence'])
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download(identity['repo_id'], identity['filename'], revision=identity['revision'], local_files_only=True)
    if sha(checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('Pinned checkpoint content mismatch')
    return dict(identity=identity, checkpoint_path=checkpoint, source_gate3=str(source.relative_to(ROOT)),
        source_gate3_fixtures_sha256=sha(source), cases=cases, reference_queries=list(queries.values()),
        tolerance=dict(rtol=RTOL, atol=ATOL, justification='User-specified initial tolerance, fixed before execution. Exact equality additionally recorded; batch-size effects are measured without loosening thresholds.'))


def supervise(script, run, role):
    import psutil
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'HF_HUB_OFFLINE': '1', 'MPLBACKEND': 'Agg', 'MPLCONFIGDIR': str(run / (role + '-mpl'))}
    command = [sys.executable, '-I', str(BASE / script), '--run', str(run)]
    peak = 0
    start = time.monotonic()
    reason = None
    with (run / (role + '.log')).open('w') as log:
        process = subprocess.Popen(command, cwd='/tmp', env=env, stdout=log, stderr=subprocess.STDOUT)
        tracked = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                peak = max(peak, tracked.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            if peak > 1_800_000_000:
                reason = 'Worker exceeded 1.8 GB RSS limit.'
            if psutil.virtual_memory().available < 650_000_000:
                reason = 'Less than 650 MB available memory.'
            if time.monotonic() - start > 600:
                reason = 'Worker exceeded 600 seconds.'
            if reason:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(.25)
    return dict(command=command, exit_code=process.returncode, peak_rss_bytes=peak,
                seconds=time.monotonic() - start, blocked_reason=reason)


def compare_arrays(actual, expected, ids, path, label):
    import numpy as np
    info = dict(label=label, sequence_ids=ids, shape_actual=list(actual.shape), shape_expected=list(expected.shape),
        dtype_actual=str(actual.dtype), dtype_expected=str(expected.dtype), rtol=RTOL, atol=ATOL,
        finite_actual=bool(np.isfinite(actual).all()), finite_expected=bool(np.isfinite(expected).all()))
    ok = actual.shape == expected.shape and actual.dtype == expected.dtype == np.float32 and info['finite_actual'] and info['finite_expected']
    if actual.shape == expected.shape and info['finite_actual'] and info['finite_expected']:
        delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        denominator = np.maximum(np.abs(expected.astype(np.float64)), np.finfo(np.float32).tiny)
        relative = delta / denominator
        close = np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
        maximum = np.unravel_index(np.argmax(delta), delta.shape)
        maximum_relative = np.unravel_index(np.argmax(relative), relative.shape)
        mismatches = np.argwhere(~close)
        info.update(exact_equal=bool(np.array_equal(actual, expected)), max_absolute_error=float(delta.max()),
            max_relative_error=float(relative.max()), mean_absolute_error=float(delta.mean()),
            mismatched_elements=int((~close).sum()), exact_unequal_elements=int((actual != expected).sum()),
            relative_error_definition='abs(actual-expected) / max(abs(expected), float32.tiny); tolerance uses atol+rtol*abs(expected)',
            max_relative_error_location=dict(index=[int(i) for i in maximum_relative], sequence_id=ids[maximum_relative[0]],
                task=int(maximum_relative[1]), output_position_0based=int(maximum_relative[2]),
                actual=float(actual[maximum_relative]), expected=float(expected[maximum_relative])),
            max_error_location=dict(index=[int(i) for i in maximum], sequence_id=ids[maximum[0]],
                task=int(maximum[1]), output_position_0based=int(maximum[2]),
                actual=float(actual[maximum]), expected=float(expected[maximum])),
            mismatch_locations=[dict(index=[int(i) for i in idx], sequence_id=ids[idx[0]],
                actual=float(actual[tuple(idx)]), expected=float(expected[tuple(idx)])) for idx in mismatches])
        try:
            np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
        except AssertionError as exc:
            ok = False
            info['assertion'] = str(exc)
    info['status'] = 'PASS' if ok else 'FAIL'
    save(path, info)
    return ok, info


def evaluate(run, fixture, resources):
    import numpy as np
    checks = []
    coverage = []
    website = json.loads((run / 'website/manifest.json').read_text())
    direct = json.loads((run / 'direct/manifest.json').read_text())
    expected_full = np.load(run / 'direct/all-batch1.npy', allow_pickle=False)
    by_sequence = {r['sequence']: i for i, r in enumerate(fixture['reference_queries'])}
    common_settings = direct['checkpoint_sha256'] == fixture['identity']['checkpoint_sha256'] and direct['state_sha256_before'] == direct['state_sha256_after']
    common_settings = common_settings and not direct['studio_modules_loaded'] and direct['evaluation_mode'] and not website['augmentation_calls']
    common_settings = common_settings and direct['tasks'] == fixture['identity']['tasks'] and direct['train_window'] == 2114

    def check(name, ok, detail, evidence, workflow):
        row = dict(id=name, status='PASS' if ok else 'FAIL', explanation=detail, evidence=evidence, workflow=workflow)
        checks.append(row)
        print(row['status'], name, flush=True)

    matched_cases = {}
    for case, observed in zip(fixture['cases'], website['cases'], strict=True):
        if case.get('expected_error'):
            continue
        folder = run / 'comparisons' / case['id']
        folder.mkdir(parents=True)
        case_ok = common_settings and len(observed['calls']) == len(case['expected_calls']) and not observed.get('error')
        if case.get('stop_after_calls'):
            case_ok = case_ok and observed.get('deliberately_stopped_after_real_outputs', False)
        elif case['kind'] not in ('adapter', 'engine_design'):
            case_ok = case_ok and observed.get('status') == 200
        per_call = []
        for index, (call, intended) in enumerate(zip(observed['calls'], case['expected_calls'])):
            ids = [r['id'] for r in intended]
            desired_sequences = [r['sequence'] for r in intended]
            evidence = dict(index=index, ids=ids, sequences_preserved=call['input_sequences'] == desired_sequences,
                tasks_correct=call['tasks'] == fixture['identity']['tasks'],
                selected_checkpoint=(Path(call['model_meta'].get('path', '')).resolve() == Path(fixture['checkpoint_path']).resolve()
                    and sha(call['model_meta']['path']) == fixture['identity']['checkpoint_sha256']),
                selected_window=call['sequence_length'], device=call['device'],
                batch_size=call['batch_size'])
            settings_ok = evidence['sequences_preserved'] and evidence['tasks_correct'] and evidence['selected_checkpoint']
            settings_ok = settings_ok and call['device'] == 'cpu' and call['sequence_length'] == 2114
            settings_ok = settings_ok and call['state_sha256_before'] == call.get('state_sha256_after') == direct['state_sha256_before']
            expected = expected_full[[by_sequence[r['sequence']] for r in intended]]
            np.save(folder / f'call-{index:02d}-reference.npy', expected, allow_pickle=False)
            if case['kind'] == 'adapter':
                settings_ok = settings_ok and call['batch_size'] == case['batch_size'] and call['backend'] == case['backend']
            if not case.get('fault') and call['backend'] == 'grelu':
                settings_ok = settings_ok and call['native_enabled_before'] and call.get('native_enabled_after') and not call.get('notes')
            call_ok = bool(settings_ok) and 'result' in call
            offset = 0
            raw_chunks = []
            for batch_index, raw in enumerate(call['forwards']):
                array = np.load(run / 'website' / raw['file'], allow_pickle=False)
                batch = len(array)
                file = folder / f'call-{index:02d}-raw-{batch_index:02d}.json'
                ok, metrics = compare_arrays(array, expected[offset:offset + batch], ids[offset:offset + batch], file, 'native forward return before Studio _norm3')
                raw_ok = (raw['device'] == 'cpu' and raw['dtype'] == 'torch.float32'
                          and raw['input_device'] == 'cpu' and raw['input_dtype'] == 'torch.float32'
                          and not raw['training_modules'] and not any(raw['dropout_modules'].values())
                          and raw['input_shape'] == [batch, 4, 2114]
                          and raw['activation'] == direct['activation'] and raw['transform'] == direct['transform'])
                call_ok = call_ok and ok and raw_ok
                offset += batch
                raw_chunks.append(array)
            expected_batches = [min(call['batch_size'], len(intended) - start) for start in range(0, len(intended), call['batch_size'])]
            observed_batches = [r['shape'][0] for r in call['forwards']]
            call_ok = call_ok and offset == len(intended) and len(call['normalizations']) == len(call['forwards']) and observed_batches == expected_batches
            if call.get('result'):
                actual = np.load(run / 'website' / call['result'], allow_pickle=False)
                ok, metrics = compare_arrays(actual, expected, ids, folder / f'call-{index:02d}-returned.json', 'adapter return before workflow aggregation')
                # The full return must preserve native axes, concatenation and values.
                call_ok = call_ok and ok and bool(raw_chunks) and np.array_equal(actual, np.concatenate(raw_chunks))
            for normal, raw in zip(call['normalizations'], call['forwards']):
                pre = np.load(run / 'website' / normal['raw'], allow_pickle=False)
                normalized = np.load(run / 'website' / normal['normalized'], allow_pickle=False)
                native = np.load(run / 'website' / raw['file'], allow_pickle=False)
                call_ok = call_ok and np.array_equal(pre, native) and np.array_equal(normalized, native)
            evidence.update(numerical_and_settings_pass=bool(call_ok), observed_batch_sizes=[r['shape'][0] for r in call['forwards']],
                            total_predictions=offset)
            save(folder / f'call-{index:02d}-mapping.json', evidence)
            per_call.append(evidence)
            case_ok = case_ok and call_ok
        # Actual reader IDs are checked separately from intended tensor labels.
        if case['workflow'] in ['/predict', '/api/compare', '/allele-specific', '/prioritize', '/multi-allelic']:
            actual_records = observed['reader_records'][0] if observed['reader_records'] else []
            intended = case['expected_calls'][0]
            case_ok = case_ok and actual_records == [[r['id'], r['sequence']] for r in intended]
        if case.get('fault'):
            disclosed = observed.get('response_json') or observed.get('engine_outcome') or {}
            if not disclosed and observed.get('response_record'):
                disclosed = observed['response_record']
            warning_visible = any('fell back' in note for note in disclosed.get('warnings', []))
            save(folder / 'fallback-warning.json', dict(adapter_notes=[c.get('notes') for c in observed['calls']],
                actual_returned_warnings=disclosed.get('warnings', []), warning_visible=warning_visible))
            case_ok = case_ok and bool(observed['faults']) and warning_visible and any('fell back' in note for c in observed['calls'] for note in c.get('notes', []))
        save(folder / 'case.json', dict(id=case['id'], numerical_comparison_and_mapping_pass=bool(case_ok), calls=per_call,
            expected_calls=len(case['expected_calls']), observed_calls=len(observed['calls']),
            website_evidence=f'../../website/{case["id"]}.json', scope=case.get('partial', 'Complete tested request/adapter call; final score/display correctness deferred.')))
        check(case['id'], bool(case_ok), 'Full real forward tensors and adapter returns compared at every task/position; checkpoint, settings, counts and intended DNA/ID order checked.',
              [str((folder / 'case.json').relative_to(run)), 'website/' + case['id'] + '.json'], case['workflow'])
        matched_cases[case['id']] = bool(case_ok)

    # All these invariants are supported by full pretrained output comparisons.
    direct2 = np.load(run / 'direct/all-batch2.npy', allow_pickle=False)
    ids = [r['id'] for r in fixture['reference_queries']]
    ok, info = compare_arrays(direct2, expected_full, ids, run / 'comparisons/direct-batching.json', 'independent gReLU batch2 versus batch1')
    check('reference_batch_invariance', ok and common_settings, 'Native full output arrays across batch sizes; fixed tolerances, exact differences retained.', ['comparisons/direct-batching.json'], 'batch invariants')
    for name, left, right in [('repeat_identical_requests', 'http_fasta', 'http_repeat')]:
        observations = {r['id']: r for r in website['cases']}
        a = np.load(run / 'website' / observations[left]['calls'][0]['result'], allow_pickle=False)
        b = np.load(run / 'website' / observations[right]['calls'][0]['result'], allow_pickle=False)
        logical_ids = [r['id'] for r in next(c for c in fixture['cases'] if c['id'] == left)['expected_calls'][0]]
        ok, info = compare_arrays(a, b, logical_ids, run / 'comparisons/repeat.json', 'repeat identical request outputs')
        check(name, ok and np.array_equal(a, b) and matched_cases[left] and matched_cases[right],
              'Repeated real request outputs must also be exactly equal, not merely within tolerance.', ['comparisons/repeat.json'], 'determinism')
    fault = next(r for r in website['cases'] if r['id'] == 'http_persistent_fault')
    ok = fault.get('status') == 500 and fault['faults'] and all('result' not in c for c in fault['calls'])
    check('model_exception_handling', bool(ok and matched_cases['http_fault_recovery']),
          'Explicitly injected exceptions: persistent error gives HTTP 500/no fabricated outputs; transient native error has disclosed builtin recovery whose real full outputs match direct gReLU.',
          ['website/http_persistent_fault.json', 'website/http_fault_recovery.json', 'comparisons/http_fault_recovery/case.json'], 'exception recovery')
    scopes = [
        ('distinct_second_checkpoint', '/model-comparison', 'Only pinned GM12878 has previously verified identity/window. Both same-model selections run; a different model/window comparison is unverified. No demo or new download is substituted.'),
        ('remaining_curated_positions', '/saturation', 'First reference and three substitutions numerically verified, then explicitly stopped. Full curated scan and its completed response are unverified.'),
        ('attribution_gradient_outputs', '/attribution', 'Real baseline and shared builtin forward are numerically covered; gradient attribution outputs are not numerical model-output equivalence and are not run.'),
        ('design_optimization', '/design', 'Real seed prediction numerically verified; optimization and subsequent candidates are not run.'),
        ('gpu_and_other_models', 'device/models', 'CPU pinned GM12878 only; no CUDA/MPS or other architecture certification.'),
        ('browser_and_processed_outputs', 'Gate 5/6', 'No browser, score aggregation, rounding, visualization or export correctness checks. Responses may naturally be generated but are not certified.'),
        ('training_fine_tuning', 'training', 'No training, fine-tuning, optimizer or gradient computation is executed.')]
    for name, workflow, reason in scopes:
        save(run / 'scope' / (name + '.json'), dict(reason=reason))
        checks.append(dict(id=name, status='SKIP', workflow=workflow, explanation=reason, evidence=['scope/' + name + '.json']))
    for workflow in sorted({c['workflow'] for c in fixture['cases']}):
        cases = [c for c in fixture['cases'] if c['workflow'] == workflow and not c.get('expected_error')]
        status = 'FAILED' if any(not matched_cases.get(c['id']) for c in cases) else 'PARTIAL' if any(c.get('partial') for c in cases) else 'VERIFIED'
        coverage.append(dict(workflow=workflow, classification=status, cases=[c['id'] for c in cases],
            limitations=[c['partial'] for c in cases if c.get('partial')]))
    for workflow in ['/qc', '/research/dataset', '/research/bed', '/motifs', '/pwm-scan', '/training', '/fine-tuning', '/variants', '/models', 'frontend']:
        coverage.append(dict(workflow=workflow, classification='NOT TESTED', cases=[],
            limitations=['Non-inference descriptive/retired path, training, or browser behavior outside this gate.']))
    return dict(checks=checks, coverage=coverage, reference_manifest='direct/manifest.json',
                website_manifest='website/manifest.json', resources=resources, gate=4, gate5_started=False,
                biological_benchmark=False, tolerances=fixture['tolerance'])


def main():
    run = BASE / 'runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run.mkdir(parents=True)
    before = protected()
    save(run / 'protected-before.json', before)
    fixture = fixtures()
    save(run / 'fixtures.json', fixture)
    (run / 'inputs.fasta').write_text(''.join(f">{r['id']}\n{r['sequence']}\n" for r in fixture['reference_queries']))
    checkpoint_before = sha(fixture['checkpoint_path'])
    resources = {}
    result = dict(checks=[], coverage=[], gate=4, gate5_started=False)
    for role, script in [('website', 'website_worker.py'), ('direct', 'direct_reference.py')]:
        resources[role] = supervise(script, run, role)
        save(run / 'resources.json', resources)
        print(role, json.dumps(resources[role]), flush=True)
        if resources[role]['exit_code'] != 0:
            result['checks'].append(dict(id=role + '_execution', status='BLOCKED' if resources[role]['blocked_reason'] else 'FAIL',
                workflow='execution', explanation=resources[role]['blocked_reason'] or 'Worker failed; see saved log. No numerical PASS is claimed.', evidence=[role + '.log']))
            break
    else:
        try:
            result = evaluate(run, fixture, resources)
        except Exception:
            import traceback
            (run / 'evaluation-error.log').write_text(traceback.format_exc())
            result['checks'].append(dict(id='evaluation', status='FAIL', workflow='execution',
                explanation='Evaluation failed; raw evidence retained.', evidence=['evaluation-error.log']))
    after = protected()
    save(run / 'protected-after.json', after)
    unchanged = all(after.get(name) == value for name, value in before.items())
    checkpoint_unchanged = sha(fixture['checkpoint_path']) == checkpoint_before
    result['preservation'] = dict(earlier_file_count=len(before), earlier_evidence_unchanged=unchanged,
        checkpoint_sha256_before=checkpoint_before, checkpoint_sha256_after=sha(fixture['checkpoint_path']),
        gate3_optional_findings_preserved=True if unchanged else False)
    if not unchanged or not checkpoint_unchanged:
        result['checks'].append(dict(id='preservation', status='FAIL', workflow='preservation', explanation='Earlier evidence or checkpoint changed.', evidence=['protected-before.json', 'protected-after.json']))
    result['resources'] = resources
    result['counts'] = {s: sum(c['status'] == s for c in result['checks']) for s in ['PASS', 'FAIL', 'SKIP', 'BLOCKED']}
    result['status'] = 'FAIL' if result['counts']['FAIL'] else 'BLOCKED' if result['counts']['BLOCKED'] else 'PASS'
    result['source_hashes'] = {str(p.relative_to(ROOT)): sha(p) for folder in ['core', 'workflows'] for p in (ROOT / folder).glob('*.py')}
    for script in ['verify_gate4.py', 'website_worker.py', 'direct_reference.py']:
        result['source_hashes'][str((BASE / script).relative_to(ROOT))] = sha(BASE / script)
        (run / (script + '.txt')).write_text((BASE / script).read_text())
    save(run / 'results.json', result)
    save(run / 'workflow-coverage.json', result['coverage'])
    save(BASE / 'latest.json', dict(run=str(run.relative_to(BASE)), status=result['status'], counts=result['counts']))
    lines = ['# Gate 4 — numerical inference equivalence', '', f"**{result['status']} — {result['counts']}**", '',
        'Actual pretrained CPU outputs, not biological accuracy. No Gate 5/6 checks are started.', '',
        '| Check | Status | Explanation | Evidence |', '|---|---|---|---|']
    for row in result['checks']:
        refs = ', '.join(f'[{p}]({p})' for p in row['evidence'])
        lines.append(f"| {row['id']} | {row['status']} | {row['explanation']} | {refs} |")
    lines += ['', '## Workflow coverage', '', '| Workflow | Classification | Limitations |', '|---|---|---|']
    for row in result['coverage']:
        lines.append(f"| {row['workflow']} | {row['classification']} | {' '.join(row['limitations'])} |")
    lines += ['', '## Evidence and reproduction', '',
        'See [pipeline and independence](../../PIPELINE.md), [bugs and fixes](../../BUGS_AND_FIXES.md), [fixtures](fixtures.json), [website captures](website/manifest.json), [direct manifest](direct/manifest.json), [resources](resources.json), and [machine-readable results](results.json).', '',
        f'Fixed before execution: rtol={RTOL}, atol={ATOL}. Exact equality, zero-safe relative error, every position/task, mismatches and maximum locations are saved for each comparison. Original .npy arrays are retained on both paths.', '',
        f'Preservation evidence: {result["preservation"]}', '',
        '```bash', '.venv/bin/python Testing/testing-phase-2/input-output/gate-4/verify_gate4.py', '```', '',
        'Each run is retained. Exit nonzero on FAIL/BLOCKED. Read all SKIP and PARTIAL limitations. Stopped after Gate 4 for review; no inference-equivalence conclusion is a biological-accuracy claim.']
    (run / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(report=str(run / 'REPORT.md'), status=result['status'], counts=result['counts'])), flush=True)
    return 1 if result['status'] == 'FAIL' else 2 if result['status'] == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
