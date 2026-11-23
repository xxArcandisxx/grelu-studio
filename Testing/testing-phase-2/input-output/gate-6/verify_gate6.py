"""Final Step 1 gate, supervised isolated workers. No biological diagnostics."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n')


def worker(role,script,out):
    import psutil
    folder=out/role;folder.mkdir()
    command=[sys.executable,'-I',str(BASE/script),str(folder)]
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1','MPLBACKEND':'Agg','MPLCONFIGDIR':str(folder/'mpl')}
    start=time.monotonic();peak=0;reason=None
    with (out/(role+'.log')).open('w') as log:
        process=subprocess.Popen(command,cwd='/tmp',env=env,stdout=log,stderr=subprocess.STDOUT)
        tracked=psutil.Process(process.pid)
        while process.poll() is None:
            try:
                resident=tracked.memory_info().rss
                for child in tracked.children(recursive=True):
                    try:resident+=child.memory_info().rss
                    except psutil.NoSuchProcess:pass
                peak=max(peak,resident)
            except psutil.NoSuchProcess:pass
            if peak>1800000000:reason='Exceeded 1.8 GB RSS ceiling.'
            if psutil.virtual_memory().available<650000000:reason='Less than 650 MB available RAM.'
            if time.monotonic()-start>420:reason='Exceeded 420-second limit.'
            if reason:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(.2)
    return dict(command=command,exit_code=process.returncode,peak_rss_bytes=peak,seconds=time.monotonic()-start,blocked_reason=reason)


def main():
    out=BASE/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out.mkdir(parents=True)
    before=json.loads((BASE/'protected-before.json').read_text());save(out/'protected-before.json',before)
    checks=[];resources={}
    for role,script in [('storage','verify_storage.py'),('fresh','verify_fresh.py'),('regressions','verify_regressions.py')]:
        resources[role]=worker(role,script,out);save(out/'resources.json',resources)
        path=out/role/'results.json'
        if path.exists():
            result=json.loads(path.read_text())
            for row in result['checks']:row['evidence']=role+'/'+row['evidence'];checks.append(row)
        else:checks.append(dict(id=role+'_execution',status='BLOCKED' if resources[role]['blocked_reason'] else 'FAIL',mode='EXECUTION',workflow=role,evidence=role+'.log',reason=resources[role]['blocked_reason'] or 'Worker did not complete; retained traceback/log. No pass inferred.'))
    after={name:sha(ROOT/name) for name in before};changed=[name for name in before if before[name]!=after[name]]
    save(out/'protected-after.json',after)
    if not changed:checks.append(dict(id='preservation',status='PASS',mode='PRESERVATION',workflow='prior evidence and checkpoint',evidence='protected-after.json',reason=f'All {len(before)} protected file SHA-256 values unchanged.'))
    if changed:checks.append(dict(id='preservation',status='FAIL',mode='PRESERVATION',workflow='prior evidence',evidence='protected-after.json',changed=changed))
    counts={s:sum(c['status']==s for c in checks) for s in ['PASS','FAIL','SKIP','BLOCKED']}
    status='FAIL' if counts['FAIL'] else 'BLOCKED' if counts['BLOCKED'] else 'PASS'
    result=dict(gate=6,status=status,counts=counts,checks=checks,resources=resources,
        mode_counts={m:{s:sum(c['mode']==m and c['status']==s for c in checks) for s in counts} for m in sorted({r['mode'] for r in checks})},
        protected_file_count=len(before),changed_protected_files=changed,step2_started=False,
        biological_accuracy_established=False,browser_executed=False)
    save(out/'results.json',result);save(BASE/'latest.json',dict(run=str(out.relative_to(BASE)),status=status,counts=counts))
    for path in BASE.glob('*.py'):(out/(path.name+'.txt')).write_text(path.read_text())
    report=['# Gate 6 — persistence, exports and experiment history','',f'**{status} — {counts}**','',
        'Isolated SQLite/job files only. Approved real-output replay and synthetic serialization/job fixtures are distinct from the small fresh real-model lifecycle. No biological diagnostics.','',
        '| Check | Workflow | Mode | Status | Evidence |','|---|---|---|---|---|']
    for row in checks:report.append(f"| {row['id']} | {row['workflow']} | {row['mode']} | {row['status']} | [evidence]({row['evidence']}) |")
    report+=['','## Reproduction and limits','','```bash','.venv/bin/python Testing/testing-phase-2/input-output/gate-6/verify_gate6.py','```','',
        'Actual download endpoints and engine files are tested. Client Blob CSV/JSON/FASTA downloads and browser interaction remain unverified if no runtime exists. All expected-value manifests precede storage and use Gate 4 raw/Gate 5 approved results or independent synthetic literals.','',
        f'Protected earlier files: {len(before)}; changed: {len(changed)}. [Resources](resources.json), [before](protected-before.json), [after](protected-after.json).','',
        'Stopped after Gate 6. Step 2 has not started. Read the architecture, coverage, bugs, and final Step 1 summary before closure.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(dict(report=str(out/'REPORT.md'),status=status,counts=counts,resources=resources),indent=2),flush=True)
    return 1 if status=='FAIL' else 2 if status=='BLOCKED' else 0


if __name__=='__main__':raise SystemExit(main())
