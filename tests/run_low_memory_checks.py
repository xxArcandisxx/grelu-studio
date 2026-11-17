"""Run one predefined test phase with RAM, time, and child-process cleanup guards."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def available_mb():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])/1024
    return 0


def processes():
    result={}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            info={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (directory/'status').read_text().splitlines() if ':' in line}
            result[int(directory.name)]=(int(info['PPid']),int(info.get('VmRSS','0 kB').split()[0])/1024)
        except (OSError,ValueError,KeyError):pass
    return result


def tree_rss(pid):
    table=processes();ids={pid}
    while True:
        children={child for child,(parent,_) in table.items() if parent in ids}
        if children<=ids:break
        ids|=children
    return sum(table.get(child,(0,0))[1] for child in ids)


def stop(process):
    if process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=5)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['browser','browser-setup','fine-tuning','model-zoo','persist-create','persist-verify','regression'],required=True)
    parser.add_argument('--url',default='http://127.0.0.1:5008')
    parser.add_argument('--output',default='test-results/low-memory')
    args=parser.parse_args()
    output=(ROOT/args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    browser=args.phase=='browser'
    budget=600 if args.phase=='browser-setup' else 800 if browser else 1800 if args.phase in ('fine-tuning','model-zoo') else 800
    reserve=1100 if args.phase in ('fine-tuning','model-zoo') else 900
    current=available_mb()
    required=reserve+min(budget,1000)
    if current<required:
        print(json.dumps({'phase':args.phase,'status':'resource_limited','available_mb':round(current),'required_mb':required}),flush=True)
        return 3
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',GRELU_TRAIN_BATCH_SIZE='1')
    if browser:
        command=[str(ROOT/'.test-tools/venv/bin/python'),str(ROOT/'tests/run_website_suite.py'),'--url',args.url,
            '--browser-cache',str(ROOT/'.test-tools/browsers'),'--library-path',str(ROOT/'.test-tools/libraries/usr/lib/x86_64-linux-gnu'),
            '--only','pwm-meme,accounts,history-reports,mobile-navigation,negative-inputs',
            '--account-fixture',str(ROOT/'test-inputs/website-suite/account-shared-preview.json'),
            '--output',str(output/'browser')]
    elif args.phase=='browser-setup':command=[str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/setup_browser_qa.py')]
    elif args.phase=='regression':command=[str(ROOT/'.venv/bin/python'),'-m','unittest','discover','-s','tests','-q']
    else:command=[str(ROOT/'.venv/bin/python'),str(ROOT/'tests/finish_file_checks.py'),'--case',args.phase,'--output',str(output)]
    process=None;peak=0;lowest=current;reason=None;started=time.monotonic()
    try:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,start_new_session=True)
        while process.poll() is None:
            used=tree_rss(process.pid);available=available_mb()
            peak=max(peak,used);lowest=min(lowest,available)
            if used>budget:reason=f'Test process tree exceeded {budget} MB RSS';break
            if available<reserve:reason=f'Available WSL memory fell below {reserve} MB';break
            if time.monotonic()-started>300:reason='Phase exceeded five-minute time limit';break
            time.sleep(.2)
        if reason:stop(process)
        code=process.returncode
    except BaseException:
        if process:stop(process)
        raise
    finally:
        if process:stop(process)
    summary={'phase':args.phase,'status':'resource_limited' if reason else 'passed' if code==0 else 'failed',
        'reason':reason,'exit_code':code,'peak_tree_rss_mb':round(peak,1),'minimum_available_mb':round(lowest,1),
        'tree_rss_budget_mb':budget,'available_memory_reserve_mb':reserve,'seconds':round(time.monotonic()-started,2)}
    (output/(args.phase+'-resources.json')).write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    return 3 if reason else code


if __name__=='__main__':sys.exit(main())
