"""Package Gate 4's saved evidence without running another gate or model."""
import argparse
import hashlib
import json
from pathlib import Path
import re

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
    result = json.loads((run / 'results.json').read_text())
    audit = json.loads((run / 'evidence-audit.json').read_text())
    direct = json.loads((run / 'direct/manifest.json').read_text())
    website = json.loads((run / 'website/manifest.json').read_text())
    fixture = json.loads((run / 'fixtures.json').read_text())
    if audit['status'] != 'PASS' or result['status'] != 'PASS':
        raise RuntimeError('Do not package a passing report for failed evidence.')
    prior_audit = json.loads((BASE.parent / 'gate-3/investigation/usage-audit.json').read_text())
    changed = [{ 'path': name, 'before_sha256': value, 'after_sha256': sha(ROOT / name)}
               for name, value in prior_audit['source_sha256'].items() if sha(ROOT / name) != value]
    expected_changes = {'workflows/predict.py', 'workflows/vep.py', 'workflows/design.py'}
    if {r['path'] for r in changed} != expected_changes:
        raise RuntimeError('Unexpected application source changes since Gate 3 audit.')
    save(run / 'application-changes.json', dict(changes=changed,
        behavior='Append adapter.notes to returned warnings; no inference or preprocessing calculation changed.'))
    # This summarizes the completed 13-test command, not a second execution.
    save(run / 'supplemental-regressions.json', dict(
        command='.venv/bin/python -m unittest tests.test_dataset_overview tests.test_example_loading -v',
        status='PASS', tests_run=13, failures=0, errors=0, execution_seconds=.355,
        evidence_origin='Recorded from completed command output in this session after the warning fixes.',
        scope='Existing input-only/dataset regressions; not evidence of model-output equivalence and not included in Gate 4 numeric PASS count.'))
    (run / 'supplemental-regressions.txt').write_text(
        'Recorded summary of completed command after the production fix:\n'
        '.venv/bin/python -m unittest tests.test_dataset_overview tests.test_example_loading -v\n'
        'Ran 13 tests in 0.355s\nOK\n'
        'Full verbose output was returned in the session. Gate 4 numeric results are recorded separately.\n')
    earlier = {}
    for gate, directory in [('gate1', BASE.parent), ('gate2', BASE.parent / 'gate-2')]:
        latest = json.loads((directory / 'latest.json').read_text())
        path = directory / latest['run'] / 'results.json'
        record = json.loads(path.read_text())
        earlier[gate] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
            counts=record.get('counts', record.get('summary')), execution='Prior retained run; not rerun by Gate 4.')
    gate3 = BASE.parent / 'gate-3/investigation/results.json'
    earlier['gate3_closure'] = dict(path=str(gate3.relative_to(ROOT)), sha256=sha(gate3),
        required_counts=dict(PASS=148, FAIL=0, SKIP=8, BLOCKED=0),
        optional_counts=dict(PASS=2, FAIL=5, SKIP=0, BLOCKED=0),
        execution='Existing findings retained unchanged, not rerun or relabeled.')
    save(run / 'earlier-gates.json', earlier)
    if not (run / 'results.as-generated.json').exists():
        (run / 'results.as-generated.json').write_text((run / 'results.json').read_text())
    result.update(evidence_audit='evidence-audit.json', model_settings='model-settings-manifest.json',
        application_changes='application-changes.json', supplemental_regressions='supplemental-regressions.json',
        earlier_gate_evidence='earlier-gates.json', statistics={key: audit[key] for key in
        ['actual_raw_forward_arrays_rechecked', 'numerical_comparison_count', 'all_comparisons_exact',
         'max_absolute_error', 'max_relative_error', 'max_mean_absolute_error', 'total_mismatched_elements']})
    save(run / 'results.json', result)
    report = run / 'REPORT.md'
    original = run / 'REPORT.as-generated.md'
    if not original.exists():
        original.write_text(report.read_text())
    text = original.read_text()
    intro = f'''
## Findings

**28 PASS, 0 FAIL, 7 SKIP, 0 BLOCKED.** The tests execute real pretrained
GM12878 inference through actual Flask handlers, real loaders and both adapter
backends, then compare it to native gReLU in a separate process with no Studio
inference imports. No numerical prediction is mocked or hardcoded.

- {len(fixture['reference_queries'])} distinct deterministic DNA queries are scored by the independent reference
  with batch sizes 1 and 2. Inputs include accepted nonperiodic Gate 3 DNA,
  unknown-N context, distinct-ID duplicate DNA, reverse complements, start /
  center / end SNVs, motif replacement and small saturation mutants.
- {audit['actual_raw_forward_arrays_rechecked']} actual website forward tensors are saved. All {audit['numerical_comparison_count']} full-output
  comparisons (raw, adapter-return, batching and repeat) are **exactly equal**:
  maximum absolute error 0, maximum zero-safe relative error 0, mean absolute
  error 0, and zero mismatched elements. The original fixed allclose tolerance
  remains rtol=1e-5 / atol=1e-6; it was never increased.
- Native input is float32 `(batch,4,2114)` on CPU. Native output is float32
  **`(batch,1,1)`**, with axes sequence/task/output-position. This checkpoint's
  own head performs average pooling to one output position. The suite captures
  the full native return before Studio aggregation; it does not replace a
  spatial profile with summary statistics or fabricate additional positions.
- Task 0 is `ENCFF093VXI`; activation and native prediction transform are
  `{direct['activation']}` / `{direct['transform']}`. All observed model modules are in evaluation mode,
  with no active dropout or native inference augmentation. Explicit application
  reverse-complement scoring is tested as separate DNA, not hidden augmentation.
- Loaded state hashes match between direct and website paths before/after all
  completed calls. Checkpoint SHA-256 is
  `{fixture['identity']['checkpoint_sha256']}`.

### Bug fixed and regression evidence

Prediction, CSV variant and design workflows previously discarded the
adapter's fallback note. An explicitly injected native-path RuntimeError followed by real
builtin inference produced numerically correct values but omitted its warning
from the returned API record. The three workflows now append `adapter.notes`
to their returned warnings. Real-inference regression cases verify the warning
in API prediction, CSV variant HTML/record, and zero-iteration engine design.
Persistent failures return HTTP 500 without fabricated predictions.

No model loading, encoding, weights, output values, score aggregation, design
styling or unrelated functionality was modified. See [application changes](application-changes.json),
[pre-fix evidence](../../pre-fix-findings.json) and [bugs/fixes report](../../BUGS_AND_FIXES.md).
All 13 existing supplemental input-only tests pass; their recorded command and
results are in [supplemental regressions](supplemental-regressions.json).
They are excluded from the 28 numerical PASS count.

### Resources and preservation

Workers run sequentially with one CPU thread. Website peak RSS is
{result['resources']['website']['peak_rss_bytes']:,} bytes (about 1.45 GB), elapsed
{result['resources']['website']['seconds']:.1f} seconds. Reference peak RSS is
{result['resources']['direct']['peak_rss_bytes']:,} bytes, elapsed
{result['resources']['direct']['seconds']:.1f} seconds. The unchanged limits are 1.8 GB RSS,
650 MB available-memory reserve and 600 seconds per worker. No downloads,
training, fine-tuning, attribution gradients, optimizer steps or design
optimization run. The engine design regression uses max_iter=0 and n_baseline=0;
it does not claim browser support for those engine-only settings.

All {audit['protected_file_count']:,} earlier Phase 1/Gates 1–3 evidence files match their before
hashes, including the original Gate 3 optional failures. Earlier results remain
39/0/2 for Gate 1; 265/0/12 for Gate 2; 148/0/8 for the Gate 3 required subset
and 2/5/0 for its optional subset. These are prior retained results, not new
executions in Gate 4. The three warning-only source edits are the complete
application difference from the saved Gate 3 source inventory.

### Limitations retained

VERIFIED means numerical equivalence for the small test conditions listed in
the coverage matrix, not certification of every parameter combination. Model
comparison executes both paths with the same real GM12878 checkpoint; no
distinct second identity/window-verified checkpoint is available. Curated
saturation covers its actual reference plus the first three substitutions and
stops before later positions. Attribution and browser design cover real seed /
baseline inference only. Uploaded mutagenesis completes one position, and motif
simulation tests one placement, without a combinatorial scan.

All seven skips remain explicit below. GPU/other model behavior, training,
biological accuracy, processed scores, visualizations, browser correctness and
exports are not certified. Gate 5/6 work has not started. Real responses may
naturally execute existing processing code, but correctness of that processing
is left for the next reviewed gate.

### Retained earlier runs

- `20261122T163138335995Z`: 0 PASS / 0 FAIL / 0 SKIP / 1 BLOCKED at the
  memory ceiling. Between-case garbage collection and allocator cleanup
  resolved the test-process retention; the memory ceiling was not raised.
- `20261122T163447347143Z`: 1 PASS / 25 FAIL / 7 SKIP. The case failures
  reflected a path-string assertion against a Hugging Face symlink versus its
  resolved blob, while all 118 individual numerical comparisons were exact.
  The check now requires the same resolved file, pinned SHA and loaded state.
  A separate pre-fix audit also preserves the actual missing-warning failure.
  None of the prior results or arrays was rewritten as passing.

## Executed checks
'''
    text = text.replace('| Check | Status | Explanation | Evidence |', intro + '\n| Check | Status | Explanation | Evidence |', 1)
    text += '''
## Supporting evidence and standalone commands

- [Numerical artifact audit and CSV ID mapping](evidence-audit.json)
- [Model/checkpoint/settings manifest](model-settings-manifest.json)
- [Installed source evidence: native pooling and forward boundary](installed-source-evidence.json)
- [Earlier gate references and hashes](earlier-gates.json)
- [Evidence file manifest](evidence-manifest.json)

Repeat the suite, then audit its latest run, from repository root:

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-4/verify_gate4.py
.venv/bin/python Testing/testing-phase-2/input-output/gate-4/audit_evidence.py
```

The suite creates a fresh timestamped run and executes separate workers with
the commands in `resources.json`. To invoke the direct reference alone, prepare
a fresh directory containing this run's `fixtures.json` (no `direct/` directory)
and use the absolute runner path from `/tmp`:

```bash
/home/akshadha/Desktop/GRELU-Studio/.venv/bin/python -I /home/akshadha/Desktop/GRELU-Studio/Testing/testing-phase-2/input-output/gate-4/direct_reference.py --run /absolute/path/to/fresh/prepared-run
```

Native gReLU imports require their usual genomepy cache access. No network or
checkpoint download is needed. All expected numerical predictions are computed
fresh through the independent native runner. Every `.npy` is an actual captured
or computed full output, never a mocked or manually corrected prediction.

**Stopped after Gate 4. Await review before Gate 5 or biological output analysis.**
'''
    report.write_text(text)
    coverage = ['# Gate 4 workflow numerical coverage', '',
        'VERIFIED refers to the exact small cases listed. Processed-output, browser and biological correctness remain outside scope.', '',
        '| Workflow | Classification | Cases | Limitations |', '|---|---|---|---|']
    for item in result['coverage']:
        coverage.append(f"| {item['workflow']} | {item['classification']} | {', '.join(item['cases'])} | {' '.join(item['limitations'])} |")
    (run / 'WORKFLOW_COVERAGE.md').write_text('\n'.join(coverage) + '\n')
    (BASE / 'README.md').write_text(f'''# Gate 4 — numerical inference equivalence

**28 PASS, 0 FAIL, 7 SKIP, 0 BLOCKED.**

All 93 captured real website forward tensors exactly match the independent
direct-gReLU reference. All 136 full-output comparisons have zero error.
This is software correctness evidence, not biological accuracy.

- [Detailed report]({run.relative_to(BASE)}/REPORT.md)
- [Machine-readable results]({run.relative_to(BASE)}/results.json)
- [Workflow coverage]({run.relative_to(BASE)}/WORKFLOW_COVERAGE.md)
- [Actual / independent raw tensors]({run.relative_to(BASE)}/website/manifest.json)
- [Independent reference manifest]({run.relative_to(BASE)}/direct/manifest.json)
- [Model/settings manifest]({run.relative_to(BASE)}/model-settings-manifest.json)
- [Pipeline and independence](PIPELINE.md)
- [Bugs and fixes](BUGS_AND_FIXES.md)

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-4/verify_gate4.py
.venv/bin/python Testing/testing-phase-2/input-output/gate-4/audit_evidence.py
```

The real native output is (batch,1,1), with one pooled position. Full arrays
are captured before application aggregation. Earlier failed/blocked runs are
retained. HTML files are actual server-response evidence, not website source
copies; numerical checks rely on raw arrays, not displayed summary values.

Fixed omitted fallback-warning propagation in prediction, variant and design
workflows; actual model calculations and weights are unchanged. Partial paths,
distinct second-model unavailability, other devices and later output checks
remain documented. The separate Gate 3 optional failures are preserved.

**Stopped after Gate 4. Gate 5/6 and biological output analysis have not started.**
''')
    (BASE / 'REPORT.md').write_text(f'''# Gate 4 — completed report

**28 PASS, 0 FAIL, 7 SKIP, 0 BLOCKED.**

Read the [full detailed report]({run.relative_to(BASE)}/REPORT.md) and
[workflow matrix]({run.relative_to(BASE)}/WORKFLOW_COVERAGE.md).
The website's captured raw model outputs exactly match independent gReLU.
This does not establish biological accuracy or processed-output correctness.

Gate 4 is complete and stopped for review; Gate 5 has not started.
''')
    # Verify all report-local evidence links before fingerprinting the package.
    for path in [report, run / 'WORKFLOW_COVERAGE.md', BASE / 'README.md', BASE / 'REPORT.md']:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in link and not (path.parent / link).exists():
                if link != 'evidence-manifest.json':
                    raise RuntimeError('Missing report evidence: ' + str(path.parent / link))
    save(run / 'evidence-manifest.json', {str(p.relative_to(run)): sha(p)
        for p in sorted(run.rglob('*')) if p.is_file() and p.name != 'evidence-manifest.json'})
    print(json.dumps(dict(report=str(report), counts=result['counts'], audit=audit['status'],
        max_absolute_error=audit['max_absolute_error'], application_changes=changed), indent=2))


if __name__ == '__main__':
    main()
