"""Explicit, sequential GM12878 tests on UCSC DNA; never part of unittest discovery.

Run .venv/bin/python tests/run_gm12878_sample.py. Each feature gets a fresh
process and isolated databases. Memory guards stop only the owned test process.
"""
import argparse
import csv
import gc
from html.parser import HTMLParser
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'datasets/gm12878'
FIXTURES = DATA / 'derived-tests'
CASES = ['quality', 'prediction', 'variant-comparison', 'batch-variants', 'alleles',
         'prioritization', 'mutagenesis', 'attribution', 'motifs', 'pwm', 'simulation',
         'design', 'model-zoo', 'training', 'fine-tuning', 'model-comparison', 'evaluation']


class RunRecord(HTMLParser):
    def __init__(self):
        super().__init__(); self.capture = False; self.chunks = []
    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'run-record': self.capture = True
    def handle_endtag(self, tag):
        if tag == 'script': self.capture = False
    def handle_data(self, text):
        if self.capture: self.chunks.append(text)


def record(response):
    assert response.status_code == 200, response.text[:1500]
    parser = RunRecord(); parser.feed(response.text)
    assert parser.chunks, 'No result record in page'
    return json.loads(''.join(parser.chunks))


def finite(value):
    if isinstance(value, float): assert math.isfinite(value), 'Non-finite result'
    elif isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, list):
        for child in value: finite(child)


def worker(case, output):
    from huggingface_hub import hf_hub_download
    checkpoint = hf_hub_download('Genentech/GM12878_dnase-model', 'model.ckpt',
        revision='9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e', local_files_only=True)
    os.environ.update(GRELU_HUMAN_CHECKPOINT=checkpoint, GRELU_HUMAN_INPUT_LENGTH='2114',
                      GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from studio_web import app
    state = output / 'state'; state.mkdir(exist_ok=True)
    app.config.update(TESTING=True, TRAIN_BATCH_SIZE=1, LAB_ROOT=str(state/'lab'),
        ACCOUNT_DATABASE=str(state/'accounts.db'), HISTORY_DATABASE=str(state/'history.db'))
    app.secret_key = 'isolated-gm12878-sample-checks'
    client = app.test_client()
    with client.session_transaction() as session:
        session['lab_owner'] = 'isolated-gm12878-sample-owner'
    client.get('/experiments')
    with client.session_transaction() as session: csrf = session['csrf_token']
    all_text = (DATA/'test_10.fasta').read_text()
    records = []
    for block in all_text.split('>')[1:]:
        lines = block.splitlines(); records.append((lines[0], ''.join(lines[1:])))
    reference = records[0][1]
    seed = (FIXTURES/'seed.fasta').read_text()
    results = []
    def post(path, fields=None, upload=None):
        data = {'csrf_token':csrf, **(fields or {})}
        if upload:
            for name, file in upload.items(): data[name] = (io.BytesIO(file.read_bytes()), file.name)
        return client.post(path, data=data, content_type='multipart/form-data')
    def run(path, fields=None, upload=None):
        response = post(path, fields, upload)
        data = record(response); finite(data); results.append(data)
        assert data.get('ok', True), data
        return data
    def job(path, fields):
        response = post(path, fields)
        assert response.status_code == 302, response.text[:1500]
        job_id = response.location.rsplit('/',1)[-1]
        observed = []
        deadline = time.monotonic()+150
        while time.monotonic()<deadline:
            state = client.get('/api/jobs/'+job_id).json
            if state['state'] not in observed: observed.append(state['state'])
            if state['state'] in ('complete','failed'): break
            time.sleep(.1)
        assert state['state']=='complete', state
        finite(state); results.append(state)
        details['observed_job_states'] = observed
        return state['result'], job_id
    details = {'case':case, 'mode':'isolated Flask request integration; no browser',
               'reference_data':'UCSC hg38 first 10 supplied BED intervals',
               'source_fasta':str(DATA/'test_10.fasta'), 'status':'passed', 'limitations':[]}
    if case=='quality':
        data = run('/qc', {'model':'human','fasta':all_text})
        data = run('/research/dataset', {'kind':'fasta','text':all_text})
        assert data['summary']['records']==10 and data['summary']['length_min']==2114 and data['summary']['length_max']==2114
        details['summary'] = data['summary']
        assert client.get('/research/bed').status_code==404
        details['bed_validation'] = 'Removed from website; source coordinates/lengths were checked by the conversion helper.'
    elif case=='prediction':
        predictions=[]
        # Two records per HTTP request keeps actual website inference batches small.
        for start in range(0,10,2):
            fasta=''.join(f'>{name}\n{seq}\n' for name,seq in records[start:start+2])
            data=run('/predict',{'model':'human','fasta':fasta})
            assert not data.get('demo',False)
            predictions.extend(data['results']['predictions'])
            gc.collect()
        assert len(predictions)==10
        details['predictions']=predictions
    elif case=='variant-comparison':
        alternate=''.join((FIXTURES/'alternate.fasta').read_text().splitlines()[1:])
        response=client.post('/api/compare',json={'model':'human','reference':reference,'alternate':alternate})
        assert response.status_code==200,response.text
        data=response.json; finite(data);results.append(data)
        assert not data['demo'] and data['inputs']['position']==1058
        details['limitations']=['Constructed single-base substitution, not an observed variant.']
    elif case=='batch-variants':
        data=run('/batch-variants',{'model':'human','shuffles':'0'}, {'file':FIXTURES/'variants.csv'})
        assert len(data['results']['ranked_variants'])==3
        details['limitations']=['Three constructed SNVs; shuffled significance testing disabled.']
    elif case in ('alleles','prioritization'):
        path='/allele-specific' if case=='alleles' else '/prioritize'
        file=FIXTURES/('alleles.fasta' if case=='alleles' else 'prioritize.fasta')
        data=run(path,{'model':'human'},{'file':file})
        comparisons=data['results']['comparisons'];assert comparisons
        if case=='alleles': assert comparisons[0]['differences']==2
        else: assert {row['rank_within_track'] for row in comparisons}=={1,2}
        details['limitations']=['Constructed alternate sequences; no observed allele/phasing dataset supplied.']
    elif case in ('mutagenesis','attribution'):
        start,end=('1058','1059') if case=='mutagenesis' else ('1038','1077')
        data=run('/'+case,{'model':'human','text':seed,'start':start,'end':end,'target':'0'})
        if case=='mutagenesis': assert data['results']['substitutions']==6
        else: assert len(data['results']['positions'])==2114
    elif case=='motifs':
        data=run('/motifs',{'text':all_text,'patterns':(FIXTURES/'motifs.txt').read_text(),'k':'6'})
        assert data['records']==10 and data['hits']
        assert 'discovery' in data
        details['limitations']=['Exact pattern derived from the reference sequence; k-mer enrichment is exploratory.']
    elif case=='pwm':
        data=run('/pwm-scan',{'text':all_text,'threshold':'0.8','background':'0.25,0.25,0.25,0.25','smoothing':'0.1'},
            {'matrix-file':ROOT/'test-inputs/new-workflows/pwm-motifs.jaspar'})
        assert data['motifs']
        details['limitations']=['Explicitly synthetic JASPAR-format matrix; no annotated biological motif matrix supplied.']
    elif case=='simulation':
        data=run('/motif-simulation',{'model':'human','text':seed,'motif':reference[1050:1056],
            'positions':'1058','orientation':'+','target':'0','controls':'on'})
        assert data['results']
    elif case=='design':
        data=run('/design',{'model':'human','text':seed,'target':'0','iterations':'1','candidates':'2','edit_start':'1050','edit_end':'1065'})
        assert data['results']['designs']
        details['limitations']=['One iteration, two candidate edits; no claim of biological improvement.']
    elif case=='model-zoo':
        data,job_id=job('/model-zoo',{'action':'register-cached','repo':'Genentech/GM12878_dnase-model',
            'revision':'9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e'})
        key=data['model_key'];assert data['source']=='local_cache'
        inspected=client.get('/api/model-info?model='+key)
        assert inspected.status_code==200 and inspected.json['seq_len']==2114
        predicted=run('/predict',{'model':key,'fasta':seed})
        assert predicted['results']['predictions']
        assert post('/model-zoo',{'action':'remove','key':key}).status_code==200
        details['limitations']=['Pinned cache installation tested; remote downloads and other catalogue models not tested.']
    elif case in ('training','fine-tuning'):
        fields={'text':(FIXTURES/(case+'.csv')).read_text(), 'epochs':'1','learning_rate':'0.0001','target':'0'}
        fields['length']='2000'
        if case=='fine-tuning': fields['model']='human'
        data,job_id=job('/'+case,fields)
        assert data['optimizer_steps']>0
        response=client.get('/jobs/'+job_id+'/checkpoint');assert response.status_code==200
        checkpoint=output/(case+'-checkpoint.pt');checkpoint.write_bytes(response.data);response.close()
        seq=reference if case=='fine-tuning' else reference[57:2057]
        run('/predict',{'model':data['model_key'],'fasta':seq})
        details['model_key']=data['model_key']
        details['optimizer_steps']=data['optimizer_steps']
        details['limitations']=['Labels are synthetic GC fractions, not measured DNase activity. Optimizer/checkpoint execution only.']
        if case=='training': details['limitations'].append('Compact CNN trained from scratch on center-cropped 2,000-base windows; not training GM12878 from scratch.')
    elif case=='model-comparison':
        previous=output/'fine-tuning-result.json'
        if not previous.exists() or 'model_key' not in json.loads(previous.read_text()):
            details.update(status='not_tested',limitations=['A second independently trained compatible checkpoint was not available after fine-tuning.'])
        else:
            key=json.loads(previous.read_text())['model_key']
            data=run('/model-comparison',{'model_a':'human','model_b':key,'target_a':'0','target_b':'0','policy':'strict',
                'text':(FIXTURES/'comparison.fasta').read_text()})
            assert len(data['results']['comparisons'])==3
            details['limitations']=['Original DNase model versus synthetic-GC fine-tuned checkpoint; different target scales, descriptive comparison only.']
    elif case=='evaluation':
        for kind,name in [('regression','09a_Benchmark_Evaluation_regression.csv'),('binary','09b_Benchmark_Evaluation_binary_classification.csv')]:
            data=run('/research/benchmark',{'kind':kind},{'file':ROOT/'test-inputs'/name})
            assert data['metrics']
        data=run('/research/tracks',upload={'file':ROOT/'test-inputs/10_Track_Analysis_three_numeric_tracks.csv'})
        assert data['pearson']
        details['limitations']=['Evaluation/track tools checked with bundled synthetic numeric fixtures. BED contains no measured DNase targets; GM12878 accuracy, AUROC, and biological track concordance were not evaluated.']
    # Check archive privacy and real exports for completed runs in every phase.
    for data in results:
        run_id=data.get('saved_run_id') or (data.get('result') or {}).get('saved_run_id')
        if not run_id: continue
        base='/experiments/'+run_id
        assert client.get(base).status_code==200
        snapshot=client.get(base+'/record.json');assert snapshot.status_code==200
        report=client.get(base+'/report.html');assert report.status_code==200
        (output/(case+'-'+run_id[:8]+'-report.html')).write_bytes(report.data)
        assert app.test_client().get(base).status_code==404
    details['results']=results
    return details


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=CASES)
    parser.add_argument('--output',type=Path,default=ROOT/'test-results/gm12878-real-data')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                      GRELU_TRAIN_BATCH_SIZE='1',HF_HUB_OFFLINE='1')
    if args.case:
        started=time.monotonic()
        try: result=worker(args.case,output)
        except Exception as error:
            result={'case':args.case,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        result['seconds']=round(time.monotonic()-started,2)
        (output/(args.case+'-result.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({key:value for key,value in result.items() if key not in {'results','predictions','traceback'}}),flush=True)
        return 0 if result['status'] in {'passed','not_tested'} else 1
    from run_low_memory_checks import available_mb, tree_rss, stop
    summaries=[]
    for case in CASES:
        if available_mb()<1900:
            item={'case':case,'status':'resource_limited','reason':'Less than 1,900 MB currently available; no worker started.'}
        else:
            command=[sys.executable,str(Path(__file__).resolve()),'--case',case,'--output',str(output)]
            started=time.monotonic();peak=0;lowest=available_mb();reason=None
            with (output/(case+'.log')).open('w') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    while process.poll() is None:
                        rss=tree_rss(process.pid);available=available_mb();peak=max(peak,rss);lowest=min(lowest,available)
                        if rss>1700: reason='Test process tree exceeded 1,700 MB RSS';break
                        if available<850: reason='Available system memory fell below the 850 MB reserve';break
                        if time.monotonic()-started>180: reason='Feature exceeded the three-minute time limit';break
                        time.sleep(.2)
                    if reason: stop(process)
                finally: stop(process)
            path=output/(case+'-result.json')
            data=json.loads(path.read_text()) if path.exists() else {}
            item={key:value for key,value in data.items() if key not in {'results','predictions','traceback'}}
            item.update(case=case,status='resource_limited' if reason else data.get('status','failed'),
                        peak_rss_mb=round(peak,1),minimum_available_mb=round(lowest,1),seconds=round(time.monotonic()-started,2))
            if reason:item['reason']=reason
        summaries.append(item)
        (output/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    lines=['# GM12878 checks on UCSC hg38 sample','',
           'Real model execution through isolated Flask requests; no browser automation. Each feature used a fresh process.',
           'Derived variants and labels are constructed/synthetic as noted; these are execution checks, not biological accuracy results.','',
           '| Feature | Status | Seconds | Peak RSS MB |','|---|---|---|---|']
    for item in summaries: lines.append(f"| {item['case']} | {item['status']} | {item.get('seconds','—')} | {item.get('peak_rss_mb','—')} |")
    lines += ['', '## Limits', '', 'BED validation is removed from the website. All 435,055 source intervals span 2,114 bases, but only the requested first ten were fetched.',
              'No measured DNase labels, observed variant effects, biological JASPAR motif, or independent second published checkpoint were supplied.']
    for item in summaries:
        for limit in item.get('limitations',[]):lines.append(f"- {item['case']}: {limit}")
        if item.get('reason'):lines.append(f"- {item['case']}: {item['reason']}")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return 0 if all(item['status'] in {'passed','not_tested'} for item in summaries) else 1


if __name__=='__main__':sys.exit(main())
