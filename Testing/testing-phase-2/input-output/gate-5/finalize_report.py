"""Gate 5 packaging/HTML data audit; no inference, downloads or persistence tests."""
import ast
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n')


class DataAttributes(HTMLParser):
    def __init__(self):super().__init__();self.history=[];self.positions=[]
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='svg' and 'data-history' in attrs:self.history.append(json.loads(attrs['data-history']))
        if tag=='svg' and 'data-positions' in attrs:self.positions.append(json.loads(attrs['data-positions']))


def main():
    run=BASE/json.loads((BASE/'latest.json').read_text())['run']
    result=json.loads((run/'results.json').read_text())
    if result['status']!='PASS':raise RuntimeError('Cannot publish passing closure for failed checks.')
    fresh=run/'fresh';design=json.loads((fresh/'design.json').read_text());attr=json.loads((fresh/'attribution.json').read_text())
    design_html=DataAttributes();design_html.feed((fresh/'design.html').read_text())
    attr_html=DataAttributes();attr_html.feed((fresh/'attribution.html').read_text())
    problems=[]
    if design_html.history!=[d['history'] for d in design['results']['designs']]:problems.append('Actual HTML design chart dataset differs from backend history.')
    if attr_html.positions!=[attr['results']['display_positions']]:problems.append('Actual HTML attribution chart dataset differs from backend region.')
    expected=np.load(fresh/'independent-corrected-input.npy',allow_pickle=False)
    actual=np.load(fresh/'actual-attribution.npy',allow_pickle=False)
    if not np.allclose(actual,expected,rtol=1e-5,atol=1e-7):problems.append('Attribution oracle mismatch.')
    source=BASE.parent/'gate-4/runs/20261122T163955672475Z'
    fixture=json.loads((source/'fixtures.json').read_text());native=json.loads((source/'direct/manifest.json').read_text())
    safety=json.loads((fresh/'safety.json').read_text())
    if sha(Path(fixture['checkpoint_path']))!=fixture['identity']['checkpoint_sha256']:problems.append('Pinned checkpoint SHA changed.')
    if safety['state_before']!=native['state_sha256_after'] or safety['state_after']!=safety['state_before']:problems.append('Loaded native weight bytes changed.')
    preserved=json.loads((run/'protected-before.json').read_text())
    changed=[name for name,value in preserved.items() if sha(ROOT/name)!=value]
    if changed:problems+=changed
    missing=[r['evidence'] for r in result['checks'] if not (run/r['evidence']).is_file()]
    if missing:problems+=missing
    raw_files=[]
    for path in sorted(fresh.glob('*-raw-*.npy')):
        values=np.load(path,allow_pickle=False)
        raw_files.append(dict(file=path.name,shape=list(values.shape),dtype=str(values.dtype),
            logical_ids=[f'{path.stem}:record-{i}' for i in range(len(values))],native_task=fixture['identity']['tasks'],
            sha256=sha(path),mode='FRESH_REAL_INFERENCE'))
    if not raw_files:problems.append('Phase-specific raw numerical evidence is missing.')
    audit=dict(status='PASS' if not problems else 'FAIL',issues=problems,
        attribution_chart_dataset_matches_backend=attr_html.positions==[attr['results']['display_positions']],
        design_chart_dataset_matches_backend=design_html.history==[d['history'] for d in design['results']['designs']],
        attribution_matrix_shape=list(actual.shape),attribution_max_absolute_error=float(np.abs(actual-expected).max()),
        attribution_mismatches=int((~np.isclose(actual,expected,rtol=1e-5,atol=1e-7)).sum()),
        attribution_rtol=1e-5,attribution_atol=1e-7,
        checkpoint_sha256=sha(Path(fixture['checkpoint_path'])),loaded_state_sha256=safety['state_after'],
        protected_file_count=len(preserved),changed_protected_files=changed,missing_test_evidence=missing,
        browser_executed=False,gate6_started=False)
    save(run/'artifact-audit.json',audit);save(run/'fresh-raw-lineage.json',raw_files)
    if problems:raise RuntimeError(str(problems))
    # Pin actual production source locations and explicitly separate Gate 4's changes.
    earlier=json.loads((BASE.parent/'gate-3/investigation/usage-audit.json').read_text())['source_sha256']
    earlier.update(json.loads((source/'results.json').read_text())['source_hashes'])
    inventory={};changes=[]
    for folder in ['core','workflows','templates','static']:
        for p in (ROOT/folder).glob('*'):
            if not p.is_file() or p.suffix not in ['.py','.html','.js']:continue
            name=str(p.relative_to(ROOT));entry=dict(sha256=sha(p))
            if p.suffix=='.py':entry['symbols']=[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
            inventory[name]=entry
            if earlier.get(name)!=entry['sha256']:changes.append(dict(path=name,before_sha256=earlier.get(name),after_sha256=entry['sha256']))
    for name in ['curated_scan.py','exploration_views.py','report_builder.py','studio_web.py','workbench.py','advanced_views.py','studio_services.py','variant_views.py']:
        p=ROOT/name;inventory[name]=dict(sha256=sha(p),symbols=[dict(name=n.name,line=n.lineno) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,(ast.FunctionDef,ast.ClassDef))])
        if earlier.get(name)!=sha(p):changes.append(dict(path=name,before_sha256=earlier.get(name),after_sha256=sha(p)))
    save(run/'source-inventory.json',inventory);save(run/'application-changes.json',changes)
    coverage=[]
    def row(workflow,numerical,html,why):coverage.append(dict(workflow=workflow,numerical=numerical,server_HTML_and_chart_data=html,interactive_browser='NOT TESTED',limits=why))
    row('Prediction and batch prediction (/predict)','PARTIAL','VERIFIED','Single/multi/upload/duplicate IDs/RC scores and precision verified. Optional shuffled-prediction control statistics not tested.')
    row('Variant pair API (/api/compare)','VERIFIED','PARTIAL','Full API values, positions, alleles, warning metadata and singleton data verified. Dynamic browser tables/plots not executed.')
    for mode in ['Allele-specific','Variant prioritization','Multi-allelic']:
        row(mode,'VERIFIED','VERIFIED','Tested supplied variants: signed delta/RC delta, stable ranking, IDs and actual formatted HTML effect rows. No biological-effect claim.')
    row('Batch variants','PARTIAL','VERIFIED','Raw reference/alternate, task-0 effects, stable absolute rank and RC sign agreement verified. Synthetic multitask arithmetic only; positive-n empirical p-value path unverified.')
    row('Uploaded mutagenesis','PARTIAL','VERIFIED','One known position and all three substitutions tested, with actual base-axis/cell formatting. Larger scans and N-skipped-position branch not fully certified.')
    row('Curated saturation','PARTIAL','PARTIAL','Explicit one-position harness subset from Gate 4; genomic/base/delta mapping checked. Full curated scan/interactive updates not certified.')
    row('Model comparison','PARTIAL','PARTIAL','Same verified checkpoint/task calculations tested. Synthetic unverified-scale policy omits difference and separates chart axes. No distinct checkpoint equivalence or biological scale validity.')
    row('Attribution','VERIFIED','PARTIAL','Actual 4×2114 matrix independently checked; signed channel sum, N zero, IDs, top-20 and display slice tested. Actual HTML chart dataset matches backend; interactive bars not executed.')
    row('Sequence design','PARTIAL','PARTIAL','One real iteration/two candidates/five backgrounds verified. Synthetic on-minus-off objective uses zero iterations, no invented outcome. Actual history chart dataset verified; larger searches/browser execution unverified.')
    row('Motif simulation','PARTIAL','PARTIAL','One saved-real replacement/reference/delta verified. No-control None checked; combinatorial orientations/partners and positive controls not fully tested.')
    row('Non-inference motif/PWM/dataset/training workflows','NOT TESTED','NOT TESTED','Not this gate’s inference-output scope; no training or Gate 6 checks executed.')
    save(run/'workflow-coverage.json',coverage)
    lines=['# Gate 5 workflow coverage','',
        'VERIFIED applies only to listed numerical/HTML conditions. All interactive browser behavior is NOT TESTED. Source inspection is not browser validation.','',
        '| Workflow | Numerical | HTML / chart data | Browser | Limits |','|---|---|---|---|---|']
    for c in coverage:lines.append(f"| {c['workflow']} | {c['numerical']} | {c['server_HTML_and_chart_data']} | {c['interactive_browser']} | {c['limits']} |")
    (run/'WORKFLOW_COVERAGE.md').write_text('\n'.join(lines)+'\n')
    shutil.copyfile(BASE/'legacy-input-regressions.log',run/'legacy-input-regressions.log')
    save(run/'regression-summary.json',dict(legacy_input_tests=10,legacy_failures=0,evidence='legacy-input-regressions.log',
        scope='Selected existing input/statistics/metadata checks; no model output exports, history, training, or persistence certification.',
        earlier_Gates='Prior Gates 1–4 evidence hashes unchanged, not re-executed.',raw_regression='fresh/checks/fresh_raw_gate4_regression.json'))
    modes=result['modes'];fresh_resource=result['resources']['fresh'];reused=fresh_resource.get('reused_saved_evidence',False)
    report=run/'REPORT.md'
    if not (run/'REPORT.as-generated.md').exists():(run/'REPORT.as-generated.md').write_text(report.read_text())
    result.update(artifact_audit='artifact-audit.json',workflow_coverage='workflow-coverage.json',
        source_inventory='source-inventory.json',application_changes='application-changes.json',
        raw_lineage='fresh-raw-lineage.json',regressions='regression-summary.json')
    save(run/'results.json',result)
    intro=f'''
## Findings and verification scope

**{result['counts']['PASS']} PASS, {result['counts']['FAIL']} FAIL, {result['counts']['SKIP']} SKIP,
{result['counts']['BLOCKED']} BLOCKED.** The processing checks pass for the exact
tested cases. [Workflow coverage](WORKFLOW_COVERAGE.md) separately classifies
numerical calculations, actual server HTML/chart data and browser behavior.
Several complete workflows remain PARTIAL; this is not a blanket certification.

- The pinned checkpoint still returns one task / one pooled output position.
  Studio passes that native score through, with no probability/log conversion,
  clipping, invented units or additional 1,000-position aggregation.
- Real Gate 4 tensors feed actual Flask processing and independent formulas
  verify scores, reference/alternate mapping, signed alternate−reference effects,
  stable absolute rankings, displayed precision, chart datasets and warnings.
- Controlled synthetic processing fixtures verify negative/zero/tiny/large
  values, ties, multitask arithmetic, missing metadata, invalid shapes and
  NaN/±Infinity rejection. Their test modes, adapter metadata and warnings mark
  them as synthetic, never as pretrained predictions.
- Real input attribution was independently calculated for all 4×2,114 elements,
  not inferred from a baseline score. Maximum absolute difference is
  {audit['attribution_max_absolute_error']:.12g}, with zero elements outside the
  predeclared rtol=1e-5 / atol=1e-7 tolerance. Full matrices/gradient inputs are
  saved under `fresh/` and chart data matches the actual backend display slice.
- One real design iteration evaluated two actual candidates and five real
  shuffled baselines. An explicit independent maximum/strict-improvement oracle
  verifies the chosen DNA, unchanged outer bases, actual history and descriptive
  background percentage. No model training or weight update occurred.
- Unverified model/task scales no longer yield a numeric subtraction or a
  forced common plot axis. Individual scores are preserved; the difference is
  uncomputed with an explicit policy/warning. Only same-configuration/same-task
  differences are justified here.

### Fixes and original failure evidence

Six original failures are retained in [baseline observations](../../baseline-failures.json):
wrong batch accepted, nonfinite aggregate accepted, finite float32 mean
overflow, silently discarded chart series, tiny score rounded to zero in chart
data, and misleading pooled-score labels. The additional
[unverified-scale subtraction failure](../../scale-policy-baseline.json) uses
explicit synthetic processing values. No original failure was relabeled as a
pass. [Bugs and fixes](../../BUGS_AND_FIXES.md) explains each production change
and the three retained signed-HTML expectation failures from an earlier harness
revision. No model prediction was edited to match its oracle.

The fixes add generic output/task validation, single-output pass-through,
stable arithmetic, full-precision chart data and neutral native-score labels.
No installed package, checkpoint weight, stylesheet or persistence/download
implementation was modified. Source hashes and actual changed files are in
[source inventory](source-inventory.json) and [application changes](application-changes.json).

### Evidence modes and resources

These counts are assertion counts, not numbers of independent model samples.
Modes distinguish saved-real-output replay, fresh real inference/gradients,
synthetic arithmetic, HTML and JS static inspection; none is silently treated
as a fresh inference benchmark. Counts by mode are in `results.json`.

Final processing worker peak RSS: {result['resources']['processing']['peak_rss_bytes']:,} bytes.
Fresh-worker evidence peak RSS: {fresh_resource['peak_rss_bytes']:,} bytes ({fresh_resource['peak_rss_bytes']/1e9:.2f} GB).
Workers use one CPU thread, offline cached weights, an unchanged 1.8 GB ceiling,
a 650 MB available-memory reserve and a 420-second timeout.
This final report uses {'source-hash-checked saved fresh evidence from '+fresh_resource.get('reused_from','') if reused else 'newly executed fresh evidence'}.
For a renderer-only rerun, completed real evidence may be copied after verifying
the production and fresh-oracle hashes; this is recorded as
`reused_saved_evidence`, with its original command/duration. Default reproduction
runs both workers fresh. The raw lineage retains distinct design/attribution
filenames and logical record IDs.

### Preserved evidence and remaining uncertainty

All {len(preserved):,} protected Phase 1/Gates 1–4 files still match their initial
hashes; the pinned checkpoint SHA and loaded state match Gate 4. The original
Gate 3 external inverse findings remain 2 PASS/5 FAIL, unchanged. Ten selected
existing input-only regression tests also pass; their log is retained and
excluded from the Gate 5 main count. Earlier full gate suites were not rerun.

No installed browser/Node/Playwright/Selenium is available, and no runtime was
downloaded. **Interactive browser rendering is NOT TESTED.** Actual Flask HTML,
server SVG calculations and graph data attributes are tested; JS source checks
are static. Full saturation, larger design searches, optional positive-n variant
background tests and combinatorial motif cases remain limited as specified in
the matrix. Distinct pretrained model comparison/scale calibration is unverified.

Raw inference correctness comes from Gate 4; these are Gate 5 processing and
presentation checks. Persistence, downloaded CSV/JSON files, experiment history
and Gate 6 remain unverified. Existing routes may naturally save test responses
in isolated databases, but those storage operations are not certified. No
biological accuracy, clinical utility or causal effect is established.

## Executed checks
'''
    text=(run/'REPORT.as-generated.md').read_text().replace('| Check | Mode | Workflow | Status | Evidence |',intro+'\n| Check | Mode | Workflow | Status | Evidence |',1)
    text+='''
## Additional evidence and commands

- [Pipeline/classification/formulas](../../PIPELINE.md)
- [Coverage matrix](WORKFLOW_COVERAGE.md)
- [Rendered chart dataset / matrix / identity audit](artifact-audit.json)
- [Fresh raw lineage](fresh-raw-lineage.json)
- [Regression summary](regression-summary.json) and [actual regression log](legacy-input-regressions.log)
- [Mode counts and machine-readable assertions](results.json)
- [File fingerprints](evidence-manifest.json)

From repository root:

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-5/verify_gate5.py
.venv/bin/python Testing/testing-phase-2/input-output/gate-5/finalize_report.py
```

The default command creates a new run, replays unchanged Gate 4 outputs into
current processing, and executes the bounded real design/gradient checks.
It exits nonzero on FAIL/BLOCKED. Native gReLU imports need their usual genomepy
cache access; no network or model download is required. All earlier runs,
including failed expectations and original baseline failures, are retained.

**Stopped after Gate 5. Await review before Gate 6 or biological diagnostics.**
'''
    report.write_text(text)
    relative=str(run.relative_to(BASE))
    (BASE/'README.md').write_text(f'''# Gate 5 — output processing, calculations and visualization

**{result['counts']['PASS']} PASS, 0 FAIL, {result['counts']['SKIP']} SKIP, 0 BLOCKED.**

- [Detailed report]({relative}/REPORT.md)
- [Machine-readable results]({relative}/results.json)
- [Numerical / HTML / browser coverage]({relative}/WORKFLOW_COVERAGE.md)
- [Pipeline and independent formulas](PIPELINE.md)
- [Bugs, baseline failures and fixes](BUGS_AND_FIXES.md)
- [Artifact audit]({relative}/artifact-audit.json)

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-5/verify_gate5.py
.venv/bin/python Testing/testing-phase-2/input-output/gate-5/finalize_report.py
```

Saved Gate 4 raw outputs test actual postprocessing/Flask responses. Synthetic
boundary fixtures are labeled separately. A small real design run and full
independent attribution matrix calculation support advanced results. Fresh
evidence reuse is explicit in resources and requires unchanged source hashes.

Browser execution is unavailable; interactive behavior is NOT TESTED. Partial
workflow/optional-control coverage remains documented. No export, database,
history or biological-accuracy certification is made. Prior evidence and
checkpoint bytes are preserved. Gate 6 has not started.
''')
    (BASE/'REPORT.md').write_text(f'# Gate 5\n\n**{result["counts"]}**\n\n[Read the detailed report]({relative}/REPORT.md).\n\nStopped for review; Gate 6 has not started.\n')
    for path in [report,BASE/'README.md',BASE/'REPORT.md']:
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in link and not (path.parent/link).exists() and link!='evidence-manifest.json':raise RuntimeError('Missing evidence link: '+link)
    save(run/'evidence-manifest.json',{str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='evidence-manifest.json'})
    print(json.dumps(dict(report=str(report),counts=result['counts'],audit=audit,changed_files=len(changes)),indent=2))


if __name__=='__main__':main()
