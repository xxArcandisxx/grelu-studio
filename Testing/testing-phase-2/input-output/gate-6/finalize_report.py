"""Package completed Gate 6 evidence without modifying Gates 1–5."""
import collections
import hashlib
from importlib.metadata import version,PackageNotFoundError
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]
PRODUCTION=['core/export_values.py','experiment_store.py','experiment_views.py','lab_jobs.py','lab_views.py',
    'workflows/predict.py','workflows/vep.py','static/advanced.js','static/exploration.js',
    'templates/experiment.html','templates/experiments.html','templates/advanced.html','templates/exploration.html']


def save(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def summary(prefix,counts,relative):
    return f'''# Phase 2, Step 1 — final verification summary

**Gate 6 is complete for its tested backend scope, with documented limitations;
Step 1 awaits user review/closure. Step 2 has not started.**

These are assertion counts, not a count of biologically validated samples. The
previous gates are retained results, not newly rerun full suites in Gate 6.

| Gate | PASS | FAIL | SKIP | BLOCKED | Evidence / conclusion |
|---|---:|---:|---:|---:|---|
| 1 — model identity | 39 | 0 | 2 | 0 | [Report]({prefix}../runs/20261122T141545575344Z/REPORT.md): pinned checkpoint/state identity; native 2114 workflow window rejects conflicting 200 config; variable architecture distinguished from workflow constraint |
| 2 — DNA input integrity | 265 | 0 | 12 | 0 | [Report]({prefix}../gate-2/runs/20261122T141643941401Z/REPORT.md): tested backend input validation/record integrity; genome/browser paths and full curated workflows limited |
| 3 — required website preprocessing | 148 | 0 | 8 | 0 | [Original report]({prefix}../gate-3/runs/20261122T142625611276Z/REPORT.md), [closure]({prefix}../gate-3/investigation/REPORT.md): actual forward-boundary tensors match independent encoding oracle |
| 3 — optional dependency compatibility | 2 | **5** | 0 | 0 | Same closure: reproducible gReLU single-base inverse defect; audited affected path unused by website; original five FAIL results retained |
| 4 — numerical inference equivalence | 28 | 0 | 7 | 0 | [Report]({prefix}../gate-4/runs/20261122T163955672475Z/REPORT.md): independent native gReLU vs actual website raw output, exact equality in tested CPU fixtures |
| 5 — output processing and presentation | 273 | 0 | 5 | 0 | [Report]({prefix}../gate-5/runs/20261122T180450337008Z/REPORT.md): actual processing/Flask HTML and limited fresh attribution/design; browser unverified |
| 6 — persistence, exports and history | {counts['PASS']} | {counts['FAIL']} | {counts['SKIP']} | {counts['BLOCKED']} | [Report]({prefix}{relative}/REPORT.md): isolated real storage/download/restart tests; approved-output replay, fresh inference and synthetic fixtures separately labeled |

## What the evidence supports

The pinned GM12878 workflow's tested input parsing, encoding, raw numerical
inference, processing and stored/downloaded values agree with the independent
oracles described in each gate. It produces one task (`ENCFF093VXI`) and one
pooled output position: `(batch,1,1)`, not a fabricated spatial profile or
calibrated probability. Gate 6 keeps exact finite JSON values and original IDs,
preserves attribution matrices/design histories, and exercises actual owned
JSON/HTML downloads, engine CSV files and fresh-process history retrieval.

## Remaining limitations and closure decision

- **Browser execution remains unverified** across all six gates. Actual Flask
  HTML/static JS checks do not certify interactive charts, client number parsing,
  submission or Blob CSV/JSON/FASTA downloads. [Browser follow-up]({prefix}BROWSER_SMOKE_PLAN.md).
- Full curated saturation, large design searches, motif/control combinations,
  positive-n variant-background tests and a distinct second pretrained model
  remain limited as recorded in earlier workflow matrices. Saving those partial
  results does not close their numerical/scientific gaps.
- Genome coordinate work without required local resources, GPU/other models,
  live-server identity and deployment behavior remain unverified where skipped.
- Gate 3's optional inverse defect remains an external compatibility limitation;
  no failure has been silently converted into PASS or SKIP.
- Gate 6 covers specific real archive workflows; PWM/research track/benchmark
  lifecycles are audited but not executed. Training/install tasks were not run.
  Generic job failure protocol uses synthetic tasks, not model-training evidence.
- JSON snapshots are processed results/replay settings, not native tensor or
  checkpoint archives. Genome/seed may be unknown, original older snapshots lack
  new metadata, and save-time versions do not establish original replay versions.
- Filesystem status and SQLite are not one atomic store. Pending archive status
  and explicit cleanup flags expose partial failures; durable task recovery and
  arbitrary repeated-disk-failure recovery are not certified.
- CSV escaping mitigates tested formula prefixes but is not a universal guarantee
  for every spreadsheet/import/re-save behavior. Canonical JSON IDs stay exact.

**Recommendation:** accept Step 1 only for the documented tested scope after
review; retain these limitations and the optional five external FAIL findings.
Do not describe the entire site as end-to-end browser verified or biologically
accurate. No Gate 6 failure currently blocks that scoped review if the run is
PASS. Biological usefulness, generalization, calibration and disease/variant
validity require separate authorized analysis; no Step 2 work was performed.
'''


def main():
    latest=json.loads((BASE/'latest.json').read_text());run=BASE/latest['run']
    result=json.loads((run/'results.json').read_text());counts=result['counts']
    cases=json.loads((run/'storage/results.json').read_text())['workflow_cases']
    case_count=len(cases);workflow_count=len({case['workflow'] for case in cases})
    if len(result['checks'])!=len({r['id'] for r in result['checks']}):raise RuntimeError('Duplicate check IDs; evidence may have been overwritten.')
    missing=[r['evidence'] for r in result['checks'] if not (run/r['evidence']).is_file()]
    if missing:raise RuntimeError('Missing supporting evidence: '+repr(missing))
    packages={}
    for name in ['numpy','torch','grelu','Flask','pandas','psutil','pytorch-lightning']:
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    environment=dict(python=platform.python_version(),executable=sys.executable,packages=packages,
        executables={name:shutil.which(name) for name in ['chromium','chromium-browser','google-chrome','firefox','node','nodejs','bun','deno']},
        frameworks={name:importlib.util.find_spec(name) is not None for name in ['playwright','selenium','pyppeteer','quickjs','js2py']},
        network_downloads_required=False,browser_executed=False,
        inference_settings=dict(device='cpu',batch_size=1,threads=1,training=False,optimizer_steps=0,offline=True),
        comparison='Storage/download exact JSON types/value/order and float.hex; fresh scores exactly match retained direct Gate 4 raw arrays.')
    save(run/'environment.json',environment)
    source=run/'production-source';source.mkdir(exist_ok=True)
    inventory={}
    for name in PRODUCTION:
        path=ROOT/name;target=source/(name.replace('/','__')+'.txt');shutil.copyfile(path,target)
        inventory[name]=dict(sha256=sha(path),snapshot=str(target.relative_to(run)))
    save(run/'source-inventory.json',inventory)
    for name in ['ARCHITECTURE.md','WORKFLOW_COVERAGE.md','BUGS_AND_FIXES.md','BROWSER_SMOKE_PLAN.md']:
        shutil.copyfile(BASE/name,run/name)
    bugs=(run/'BUGS_AND_FIXES.md').read_text()
    for target in ['baseline-findings.json','baseline-overflow/observations.json','probe_overflow.py']:
        bugs=bugs.replace(']('+target+')','](../../'+target+')')
    (run/'BUGS_AND_FIXES.md').write_text(bugs)
    for path in BASE.glob('*.py'):shutil.copyfile(path,run/(path.name+'.txt'))
    # Retain the complete generated assertion table separately, then write the
    # detailed overview without changing any test result or historical run.
    if not (run/'CHECKS.md').exists():shutil.copyfile(run/'REPORT.md',run/'CHECKS.md')
    modes='\n'.join(f"| {mode} | {v['PASS']} | {v['FAIL']} | {v['SKIP']} | {v['BLOCKED']} |" for mode,v in result['mode_counts'].items())
    history=[]
    for folder in sorted((BASE/'runs').iterdir()):
        if folder==run or not (folder/'results.json').exists():continue
        old=json.loads((folder/'results.json').read_text());history.append(f"- `{folder.name}`: {old['counts']} ({old['status']}); retained unchanged.")
    resource_lines='\n'.join(f"- {name}: {v['seconds']:.2f} seconds; peak process-tree RSS {v['peak_rss_bytes']:,} bytes; exit {v['exit_code']}." for name,v in result['resources'].items())
    report=f'''# Gate 6 — persistence, exports and experiment history

**{counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP, {counts['BLOCKED']} BLOCKED.**

Completed for the tested backend scope. **Stopped after Gate 6 for review;
Step 2 biological diagnostics have not started.** This is software verification,
not evidence of biological accuracy. These counts measure assertions, not
independent samples or scientific effect sizes.

## Findings

- Actual Flask handlers archive completed results into isolated SQLite databases;
  owned detail, JSON attachment and HTML-report routes reopen those snapshots.
  {case_count} fixtures across {workflow_count} workflows (including separate FASTA/upload cases) use approved
  Gate 4/5 results; the coverage matrix states exactly which capabilities ran.
- Prediction IDs/order, distinct-ID duplicate DNA, native scores, RC scores,
  reference/alternate association, signed effects, stable rankings, known warnings,
  attribution matrices and design history/sequence survive actual storage and
  download. Original producer snapshots are captured **before storage** and
  compared in full; independent numeric expectations come from previously
  independently verified raw/processed evidence, not the stored record.
- A separate fresh unpatched GM12878 `/api/compare` request performs real
  pretrained inference, then saves/retrieves/downloads exact reference/alternate
  scores and checkpoint/task identity. Other storage-only model paths are
  **explicitly labeled replay** and are not fresh-inference evidence.
- Engine prediction and VEP CSV files are generated, read with independent
  `csv.reader`/`DictReader` and checked for headers, order, IDs, effects, blanks,
  Unicode/commas/quotes/newlines, text escaping and full score/GC precision.
  There is no server history CSV route. Browser CSV buttons remain unverified.
- A second OS process opens the same isolated DB with the test session and
  retrieves every archived workflow through real JSON/detail endpoints. Payloads
  equal the original before-save objects. History title/type/link/status,
  newest-first tie order, 30+4 pagination, search/filter, UTC timestamps and
  timezone rollover are tested.
- Strict finite JSON preserves nested NumPy/Python values, zero, negative zero,
  negative/tiny/large finite numbers, Unicode, explicit null and false vs missing.
  Corrupt/malformed/overflowed JSON produces visible errors and original bytes
  stay intact. Finite-number handling is not a claim about JS integer precision.
- Real SQLite trigger/commit failures, duplicate IDs, missing/invalid downloads,
  separate guest/account access, escaping, safe filenames, job/file ownership,
  atomic status write failure, dead worker detection and two-store partial
  failure are exercised. Synthetic job tasks never train or fabricate models.

## Evidence modes

| Mode | PASS | FAIL | SKIP | BLOCKED |
|---|---:|---:|---:|---:|
{modes}

Real persistence/retrieval/export calls are never mocked. In replay mode only
the model adapter is replaced with previously verified real tensors/matrices;
the result carries a replay warning/metadata. Synthetic serialization and job
fixtures have explicit modes, independent literal expected values, and no
pretrained prediction claims. Existing input/account regressions are a separate
mode, counted here. Earlier full Gate 1–5 suites were **not rerun**.

## Identity, oracles and precision

Checkpoint: `Genentech/GM12878_dnase-model`, revision
`9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e`, SHA-256
`f266b296f49b4e97e3ac52e94594da0aaae1d467778ea58b421e1ee7c4482bea`.
Task `ENCFF093VXI`; CPU, batch size 1, eval inference, offline cache, no training.
Native tested input `(batch,4,2114)` and output `(batch,1,1)` remain as established
in Gates 3/4. Gate 6 does not reinterpret the pooled native score as probability,
new units, a calibrated effect, or a spatial profile.

`storage/expected-manifest.json` is saved **before the requests**. It records
identity, original DNA/IDs/query groups, raw values from direct Gate 4 `.npy`,
workflow/request settings/status, source hashes and approved Gate 5 processed
attribution/design expectations. Arithmetic for signed effects and absolute
rankings is independent of SQLite and application storage helpers.
`fresh/expected-before-request.json` likewise precedes fresh execution.
`storage/synthetic-expected.json` contains independent serialization literals.
`storage/producer-snapshots-before-storage.json` supplements these manifests with
the full original producer object (never reconstructed from a downloaded record).

JSON/storage comparisons require exact Python JSON types, list ordering and
values; floats compare `float.hex()` including negative zero. CSV numeric text
must parse back to the identical float. No loose storage tolerance is used.
Fresh scores exactly match saved independent direct-gReLU values; Gate 4's
inference tolerances are not relaxed. Some server display tables deliberately
round values; that earlier Gate 5 contract is distinct from stored/exported
full precision. Missing optional CSV fields are blank, not invented zero.

## Bugs and regression evidence

See [bugs/fixes](BUGS_AND_FIXES.md), original
[baseline findings](../../baseline-findings.json) and
[legacy/job exponent-overflow reproduction](../../baseline-overflow/observations.json).
Fixes address nonfinite downloads (including legal JSON exponent overflow),
upload-path retention, read-side job directory creation, formula-like CSV text,
GC truncation, safe download filenames, metadata and explicit partial-job state.
NumPy snapshots now have an explicit supported conversion contract.

The first full run's two failures were observer-binding errors, **retained**;
the corrected observer counts actual calls and committed rows in both bindings.
No stored/inference value was changed to satisfy expected predictions.

{chr(10).join(history)}

## Remaining unverified behavior / five SKIP results

1. **Browser execution SKIP:** no installed runtime/framework; server-rendered
   HTML and static client source are not browser evidence.
2. **Client CSV/Blob downloads SKIP:** real engine CSV and owned server JSON/HTML
   are tested; client CSV/JSON/FASTA click/download bytes are NOT TESTED.
3. **Raw tensor archiving SKIP/UNSUPPORTED:** processed attribution matrices are
   retained, but native tensors and weights are not part of snapshots/exports.
4. **Training/install execution SKIP:** no training, optimizer, installation or
   model downloads; generic job protocol tests are synthetic only.
5. **Full scientific workflow coverage SKIP:** all earlier saturation/design/
   motif/distinct-model limitations remain; persistence does not close them.

Additionally, audited PWM and research track/benchmark lifecycles were not
executed. See the [per-capability workflow matrix](WORKFLOW_COVERAGE.md).
The [browser smoke plan](BROWSER_SMOKE_PLAN.md) is follow-up work, not executed
evidence. No live deployment, arbitrary concurrent multi-process writer,
distributed transaction, task resumption or power-loss durability certification
is made. Atomic rename plus SQLite commit is not a cross-store transaction.
Known genome/seed are preserved where supplied; unknown values remain null.
Older records keep unknown provenance/status rather than new invented metadata.

CSV uses apostrophe-prefix mitigation for defined risky text prefixes and
quoting. JSON preserves exact original IDs. This is not universal spreadsheet
safety; import/re-save behavior varies, as described in
[OWASP CSV Injection](https://community.owasp.org/attacks/CSV_Injection).
Full DNA/replay inputs remain locally retained for existing repeat functionality
until explicit record deletion; separate job files/checkpoints have no automatic
GC. Metadata hashes add no duplicate DNA strings. See the architecture map.

## Artifacts and reproducible commands

- [Architecture/schema/retention map](ARCHITECTURE.md)
- [Every automated assertion and supporting evidence](CHECKS.md)
- [Machine-readable results](results.json), [environment](environment.json),
  [resource log](resources.json), [production sources/hashes](source-inventory.json)
- [Independent expected manifest](storage/expected-manifest.json),
  [original producers](storage/producer-snapshots-before-storage.json),
  [experiment IDs](storage/experiment-ids.json)
- [Actual CSV/CLI artifacts](storage/engine-files/predictions.csv),
  [VEP CSV](storage/engine-vep/vep_results.csv); actual per-workflow JSON/HTML
  attachments in `storage/exports/` and Flask response bodies in `storage/responses/`
- Isolated databases: `storage/state/experiments.db`, `accounts.db`, `legacy.db`;
  synthetic job state under `storage/state/lab/jobs/`. Intentional corruption
  fixtures are retained; these are not genuine user records.
- [Restart statuses](storage/restart/statuses.json),
  [new-process log](storage/restart-process.log), original/new-process downloads
  under `storage/restart/`; [actual regression log](regressions/unittest.log)
- [Fresh expected manifest](fresh/expected-before-request.json),
  [actual fresh API](fresh/actual-api.json), [actual fresh download](fresh/actual-download.json)
- [Final Step 1 summary](STEP_1_SUMMARY.md), [artifact hashes](evidence-manifest.json)

From repository root:

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-6/verify_gate6.py
.venv/bin/python Testing/testing-phase-2/input-output/gate-6/finalize_report.py
```

The first command makes a new timestamped run with entirely isolated state and
nonzero exit on FAIL/BLOCKED. The second packages that completed run. It does not
change test results. Installed gReLU needs its ordinary genomepy cache lock;
cached checkpoint access is offline and no model download is required. Optional
minimal overflow reproduction (also isolated, no model inference):

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-6/probe_overflow.py /tmp/grelu-gate6-overflow-review
```

After the fix both reproduction download statuses must be 500; original 200
responses remain in the baseline evidence. Run the reproduction in a new empty
directory, not a user database or an existing evidence directory.

## Resources, preservation and recommendation

Workers run sequentially with one CPU thread, a 1.8 GB process-tree RSS ceiling,
650 MB available-memory reserve and 420-second limit. Final observed workers:

{resource_lines}

Protected files: **{result['protected_file_count']:,}**; changed: **{len(result['changed_protected_files'])}**.
[Before](protected-before.json) and [after](protected-after.json) include prior
Phase 1/Gates 1–5 evidence and the checkpoint. No genuine experiment DB, installed
gReLU file or pretrained weight was modified. All historical Gate 3 optional
compatibility FAIL findings remain preserved. No website CSS/design redesign.

**Recommendation: review Gate 6 / close Step 1 only for the documented tested
scope.** Browser and partial-workflow limitations remain. Storage correctness,
export correctness, history correctness and earlier numerical checks do not
establish biological accuracy. **Stopped; awaiting review. Step 2 not started.**
'''
    (run/'REPORT.md').write_text(report)
    (BASE/'STEP_1_SUMMARY.md').write_text(summary('',counts,latest['run']))
    (run/'STEP_1_SUMMARY.md').write_text(summary('../../',counts,latest['run']))
    (BASE/'REPORT.md').write_text(f"# Gate 6 — review report\n\n**{counts['PASS']} PASS, {counts['FAIL']} FAIL, {counts['SKIP']} SKIP, {counts['BLOCKED']} BLOCKED.**\n\n[Full detailed report]({latest['run']}/REPORT.md), [machine-readable results]({latest['run']}/results.json), [workflow matrix](WORKFLOW_COVERAGE.md), [final Step 1 summary](STEP_1_SUMMARY.md).\n\nReal isolated persistence/download/restart tests, labeled approved-output replay, synthetic failure fixtures and a fresh GM12878 lifecycle. Browser-triggered exports remain unverified. Prior evidence and weights unchanged.\n\nStopped after Gate 6; awaiting review. No Step 2 biological analysis.\n")
    fingerprints={str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file() and p.name!='evidence-manifest.json'}
    save(run/'evidence-manifest.json',fingerprints)
    print(json.dumps(dict(report=str(run/'REPORT.md'),counts=counts,artifact_count=len(fingerprints),browser=environment['executables']),indent=2))


if __name__=='__main__':main()
