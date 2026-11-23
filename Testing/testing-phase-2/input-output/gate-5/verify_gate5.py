"""Gate 5 only. Sequential postprocessing replay and bounded fresh CPU checks."""
from datetime import datetime, timezone
import argparse
import shutil
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]


def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=str)+'\n')


def run_worker(role,script,out):
    import psutil
    folder=out/role;folder.mkdir()
    command=[sys.executable,'-I',str(BASE/script),str(folder)]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1',
         'MPLBACKEND':'Agg','MPLCONFIGDIR':str(folder/'mpl')}
    start=time.monotonic();peak=0;reason=None
    with (out/(role+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
        tracked=psutil.Process(process.pid)
        while process.poll() is None:
            try:peak=max(peak,tracked.memory_info().rss)
            except psutil.NoSuchProcess:pass
            if peak>1800000000:reason='Worker exceeded unchanged 1.8 GB RSS ceiling.'
            if psutil.virtual_memory().available<650000000:reason='Available RAM fell below 650 MB reserve.'
            if time.monotonic()-start>420:reason='Worker exceeded 420 second limit.'
            if reason:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(.2)
    return dict(command=command,exit_code=process.returncode,blocked_reason=reason,peak_rss_bytes=peak,seconds=time.monotonic()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--reuse-fresh',type=Path);args=parser.parse_args()
    out=BASE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    protected=json.loads((BASE/'protected-before.json').read_text())
    save(out/'protected-before.json',protected)
    source_files=[p for folder in ['core','workflows','templates','static'] for p in (ROOT/folder).glob('*') if p.is_file() and p.suffix in ['.py','.html','.js']]+[ROOT/n for n in ['studio_web.py','workbench.py','advanced_views.py','curated_scan.py','report_builder.py']]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files};save(out/'execution-source-hashes.json',source_hashes)
    checks=[];resources={}
    for role,script in [('processing','verify_processing.py'),('fresh','verify_fresh.py')]:
        if role=='fresh' and args.reuse_fresh:
            previous=args.reuse_fresh.resolve()
            if json.loads((previous/'execution-source-hashes.json').read_text())!=source_hashes or sha(previous/'verify_fresh.py.txt')!=sha(BASE/'verify_fresh.py'):
                raise ValueError('Fresh evidence cannot be reused after source or oracle changes.')
            previous_result=json.loads((previous/'fresh/results.json').read_text())
            if previous_result['counts']['FAIL'] or previous_result['counts']['BLOCKED']:raise ValueError('Cannot reuse failed fresh evidence.')
            shutil.copytree(previous/'fresh',out/'fresh')
            resources[role]=json.loads((previous/'resources.json').read_text())[role]
            resources[role].update(reused_saved_evidence=True,reused_from=str(previous),original_seconds=resources[role]['seconds'],seconds=0)
        else:resources[role]=run_worker(role,script,out)
        save(out/'resources.json',resources)
        if (out/role/'results.json').exists():
            record=json.loads((out/role/'results.json').read_text())
            for row in record['checks']:
                row['evidence']=role+'/'+row['evidence'];checks.append(row)
        else:
            checks.append(dict(id=role+'_execution',status='BLOCKED' if resources[role]['blocked_reason'] else 'FAIL',
                workflow=role,mode='EXECUTION',evidence=role+'.log',explanation=resources[role]['blocked_reason'] or 'Worker did not finish; inspect retained traceback.'))
    after={name:sha(ROOT/name) for name in protected}
    save(out/'protected-after.json',after)
    changed=[name for name in protected if protected[name]!=after[name]]
    if changed:checks.append(dict(id='preservation',status='FAIL',workflow='preservation',mode='PRESERVATION',evidence='protected-after.json',explanation=str(changed)))
    modes={mode:{s:sum(c['mode']==mode and c['status']==s for c in checks) for s in ['PASS','FAIL','SKIP','BLOCKED']} for mode in sorted({c['mode'] for c in checks})}
    counts={s:sum(c['status']==s for c in checks) for s in ['PASS','FAIL','SKIP','BLOCKED']}
    status='FAIL' if counts['FAIL'] else 'BLOCKED' if counts['BLOCKED'] else 'PASS'
    result=dict(gate=5,status=status,counts=counts,modes=modes,checks=checks,resources=resources,
        preserved_files=len(protected),changed_protected_files=changed,gate6_started=False,
        biological_accuracy_established=False,checkpoint_modified=False,
        earlier_evidence='Phase 1/Gates 1–4 including Gate 3 optional FAILs retained unchanged; these earlier suites were not rerun by this command.')
    save(out/'results.json',result);save(BASE/'latest.json',dict(run=str(out.relative_to(BASE)),status=status,counts=counts))
    environment=dict(python=sys.version,versions={name:version(name) for name in ['numpy','torch','grelu','Flask','pandas']},
        browser_available=False,browser_reason='No installed Node/browser executable or Playwright/Selenium distribution; no new download attempted.',
        CPU_threads=1,HF_HUB_OFFLINE=True,RSS_limit=1800000000,available_RAM_reserve=650000000)
    save(out/'environment.json',environment)
    lines=['# Gate 5 — output processing, calculations and presentation','',f'**{status} — {counts}**','',
        'Replay of original real Gate 4 raw tensors verifies actual application calculations/Flask HTML. Synthetic processing fixtures are separate. Fresh real design and independent attribution calculations run in a bounded worker. None of these establishes biological accuracy.','',
        '| Check | Mode | Workflow | Status | Evidence |','|---|---|---|---|---|']
    for c in checks:lines.append(f"| {c['id']} | {c['mode']} | {c['workflow']} | {c['status']} | [evidence]({c['evidence']}) |")
    lines+=['','## Independence and boundaries','',
        'Expected calculations use literal formulas and Gate 4 direct raw arrays, never Studio scoring or aggregation helpers. Every evidence row distinguishes raw source, expected/actual, formula, errors and numerical tolerance. Native GM12878 output remains (batch,1,1); single scores pass through, with Identity native activation/transform and unknown units.','',
        'Saved-output replay replaces the inference boundary only, returning recorded real tensors into actual Studio handlers. It is postprocessing evidence, not a new inference-equivalence claim. Synthetic fixtures test arithmetic/guards only. Fresh design uses actual pretrained scores; attribution uses actual gradients and independently encoded input and explicit channel correction.','',
        'Browser execution is SKIP/NOT TESTED: HTML and JS static verification is not interactive browser verification. No persistence, history, CSV/JSON download-file correctness or Gate 6 certification is performed. Existing routes may naturally save their results in isolated temporary test databases; their storage is not certified.','',
        f'Protected prior evidence files: {len(protected)}; changed: {len(changed)}. See [before](protected-before.json) and [after](protected-after.json). Resources and versions: [resources](resources.json), [environment](environment.json).','',
        '```bash','.venv/bin/python Testing/testing-phase-2/input-output/gate-5/verify_gate5.py','```','',
        'Read the pipeline map, bugs/fixes, modes and partial coverage before review. Stopped after Gate 5; Gate 6 and biological diagnostics have not started.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for p in BASE.glob('*.py'):(out/(p.name+'.txt')).write_text(p.read_text())
    print(json.dumps(dict(report=str(out/'REPORT.md'),status=status,counts=counts,resources=resources),indent=2),flush=True)
    return 1 if status=='FAIL' else 2 if status=='BLOCKED' else 0


if __name__=='__main__':raise SystemExit(main())
