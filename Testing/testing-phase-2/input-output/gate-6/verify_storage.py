"""Actual isolated storage/export tests using approved real-result replay.

No replay/synthetic value is claimed as fresh pretrained inference.
"""
from contextlib import ExitStack
from datetime import datetime,timezone,timedelta
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3];sys.path.insert(0,str(ROOT))
G4=BASE.parent/'gate-4/runs/20261122T163955672475Z'
G5=BASE.parent/'gate-5/runs/20261122T180450337008Z'


def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')


def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def strict_json(text):
    def bad(value):raise ValueError('Invalid JSON constant '+value)
    return json.loads(text,parse_constant=bad)


def exact(a,b):
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,list):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return a.hex()==b.hex()
    return a==b


class Recorder:
    def __init__(self,out):self.out=out;self.rows=[];self.mode='SAVED_REAL_RESULT_REPLAY'
    def check(self,name,actual,expected,reason,workflow='storage'):
        ok=exact(actual,expected)
        file=self.out/'checks'/(name+'.json')
        save(file,dict(id=name,status='PASS' if ok else 'FAIL',actual=actual,expected=expected,
            reason=reason,mode=self.mode,workflow=workflow,comparison='Exact JSON type/value/order; floats compared by float.hex, no storage tolerance.'))
        self.rows.append(dict(id=name,status='PASS' if ok else 'FAIL',mode=self.mode,workflow=workflow,evidence=str(file.relative_to(self.out))))
        print(self.rows[-1]['status'],name,flush=True)
    def skip(self,name,reason,workflow):
        file=self.out/'checks'/(name+'.json');save(file,dict(status='SKIP',reason=reason))
        self.rows.append(dict(id=name,status='SKIP',mode='LIMITATION',workflow=workflow,evidence=str(file.relative_to(self.out))))


def configure(out):
    fixture=json.loads((G4/'fixtures.json').read_text())
    os.environ.update(GRELU_SECRET_KEY='gate6-isolated-testing',GRELU_HUMAN_CHECKPOINT=fixture['checkpoint_path'],
                      GRELU_HUMAN_INPUT_LENGTH='2114',GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from studio_web import app
    state=out/'state';state.mkdir(parents=True,exist_ok=True)
    app.config.update(TESTING=True,PROPAGATE_EXCEPTIONS=False,DISPLAY_TIMEZONE='UTC',LAB_ROOT=str(state/'lab'),
        ACCOUNT_DATABASE=str(state/'accounts.db'),HISTORY_DATABASE=str(state/'experiments.db'))
    return app,fixture


def restart(out):
    setup=json.loads((out/'restart-request.json').read_text())
    app,_=configure(out);client=app.test_client();client.set_cookie('session',setup['cookie'])
    statuses={}
    for identifier in setup['ids']:
        response=client.get('/experiments/'+identifier+'/record.json')
        (out/'restart'/f'{identifier}.json').parent.mkdir(exist_ok=True)
        (out/'restart'/f'{identifier}.json').write_bytes(response.data)
        statuses[identifier]=dict(json=response.status_code,detail=client.get('/experiments/'+identifier).status_code)
    (out/'restart/history.html').write_text(client.get('/experiments').text)
    save(out/'restart/statuses.json',statuses)


def main(out):
    app,fixture=configure(out)
    import experiment_store as store
    import core.models,workflows.predict,workflows.vep,workflows.scan,workflows.attribution,workflows.design,workflows.explore
    import curated_scan,lab_jobs,exploration_views
    from studio_web import MODELS
    client=app.test_client();client.get('/experiments')
    with client.session_transaction() as session:csrf=session['csrf_token'];owner='guest:'+session['lab_owner']
    rec=Recorder(out)
    raw=np.load(G4/'direct/all-batch1.npy',allow_pickle=False)
    lookup={q['sequence']:raw[i].copy() for i,q in enumerate(fixture['reference_queries'])}
    task=fixture['identity']['tasks'][0]
    design_source=json.loads((G5/'fresh/design.json').read_text())
    attr_source=json.loads((G5/'fresh/attribution.json').read_text())
    current={'case':None,'design_call':0};captured=[]
    fallback=json.loads((G4/'website/http_fault_recovery.json').read_text())['calls'][0]['notes']
    class Replay:
        def __init__(self,spec):
            self.seq_len=2114;self.task_names=[task];self.meta={**spec,'verification_fixture_mode':'APPROVED_GATE4_5_REPLAY'}
            self.backend='saved_result_replay';self.device='cpu';self.notes=list(fallback)
            self.module=SimpleNamespace(data_params={},performance=None)
        def predict(self,sequences):
            if current['case']=='design':
                index=current['design_call'];current['design_call']+=1
                values=np.load(G5/f'fresh/design-raw-{index:02d}.npy',allow_pickle=False)
                if len(values)!=len(sequences):raise ValueError('Design replay order/count differs from approved Gate 5 calls.')
                return values.copy()
            return np.stack([lookup[s] for s in sequences])
        def grad_attribution(self,sequences,tasks,center):
            expected=attr_source['inputs'][0]['sequence']
            if sequences!=[expected] or tasks!=[0]:raise ValueError('Unapproved attribution replay input.')
            return np.load(G5/'fresh/actual-attribution.npy',allow_pickle=False)[None].copy()
    def loader(spec,*a,**kw):return Replay(spec),['Gate 6 storage test: approved saved Gate 4/5 outputs, not fresh inference.']
    original_save=store.save_experiment
    def observe_save(title,workflow,payload,path,form,*a,**kw):
        # Capture producer object BEFORE storage. Independent numeric manifests
        # are constructed separately below, never from this captured snapshot.
        captured.append(dict(title=title,workflow=workflow,payload=strict_json(json.dumps(payload,allow_nan=False)),path=path,form=dict(form)))
        return original_save(title,workflow,payload,path,form,*a,**kw)
    def count():
        with sqlite3.connect(out/'state/experiments.db') as db:return db.execute('SELECT COUNT(*) FROM experiments').fetchone()[0]
    def post(case):
        if case['kind']=='json':return client.post(case['url'],json=case['body'])
        fields={'csrf_token':csrf,**case.get('fields',{})}
        if case.get('upload'):fields['file']=(io.BytesIO(case['upload'].encode()),'approved.fasta')
        return client.post(case['url'],data=fields,content_type='multipart/form-data' if case.get('upload') else None)
    def payload_from(response):
        if response.is_json:return response.get_json()
        found=re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>',response.text,re.S)
        return strict_json(found[1]) if found else {}
    def score(record):return float(lookup[record['sequence']][0,0])
    cases=[];expected_manifest={}
    wanted={'http_fasta','http_upload','http_variant_1','http_allele-specific','http_prioritize','http_multi-allelic','http_variant_csv','http_mutagenesis','http_motif','http_model_comparison','http_curated_subset'}
    for case in fixture['cases']:
        if case['id'] not in wanted:continue
        cases.append(case.copy())
        groups=case['expected_calls'];expected={'source':'Independent Gate 4 raw arrays and Gate 5 formulas','workflow':case['workflow'],
            'inputs':groups[0], 'raw_groups':[[dict(id=r['id'],score=score(r)) for r in group] for group in groups],
            'model_id':fixture['identity']['website_label'],'checkpoint_sha256':fixture['identity']['checkpoint_sha256'],'task':task,'warnings_include':fallback}
        expected_manifest[case['id']]=expected
    first=fixture['cases'][0]['records'][0]['sequence']
    cases += [dict(id='design',kind='form',url='/design',workflow='/design',fields=dict(model='human',text=first,iterations='1',candidates='2',edit_start='1057',edit_end='1058',target='0')),
              dict(id='attribution',kind='form',url='/attribution',workflow='/attribution',fields=dict(model='human',text=attr_source['inputs'][0]['sequence'],start='1054',end='1060',target='0')),
              dict(id='qc',kind='form',url='/qc',workflow='/qc',fields=dict(model='human',fasta=f'>α\n{first}\n>copy\n{first}')),
              dict(id='dataset',kind='form',url='/research/dataset',workflow='/research/dataset',fields=dict(text='>α\nACGT\n>copy\nACGT',kind='fasta')),
              dict(id='motifs',kind='form',url='/motifs',workflow='/motifs',fields=dict(text='>α\nTATAAAGATA',patterns='named:TATAAA',k='3'))]
    expected_manifest['design']=dict(source='Approved Gate 5 processed result independently checked against real raw candidate scores',results=design_source['results'],input_sequence=first)
    expected_manifest['attribution']=dict(source='Approved Gate 5 processed matrix checked against independent native gradients',results=attr_source['results'],input_sequence=attr_source['inputs'][0]['sequence'])
    expected_manifest['qc']=dict(record_ids=['α','copy'],input_sequences=[first,first])
    expected_manifest['dataset']=dict(record_ids=['α','copy'],record_count=2,duplicate_count=1,input_text='>α\nACGT\n>copy\nACGT')
    expected_manifest['motifs']=dict(input_id='α',motif='named',sequence='TATAAA',start=1,end=6,strand='+')
    for case in cases:
        expected_manifest[case['id']].update(expected_status='complete',
            request_kind=case['kind'],workflow=case['workflow'],
            requested_settings=case.get('fields',case.get('body',{})),
            model_identity_scope='Pinned fixture identity at manifest root for model workflows; no model for descriptive workflows.')
    save(out/'expected-manifest.json',dict(identity=fixture['identity'],mode='APPROVED_REAL_RESULTS_REPLAY',cases=expected_manifest,
        source_files={str(p):digest(p) for p in [G4/'direct/all-batch1.npy',G4/'fixtures.json',G5/'fresh/design.json',G5/'fresh/actual-attribution.npy']}))
    ids={};producer_snapshots={};independent_matches=[]
    with ExitStack() as stack:
        for target in [core.models,workflows.predict,workflows.vep,workflows.scan,workflows.attribution,workflows.design,workflows.explore]:stack.enter_context(patch.object(target,'load_model',loader))
        stack.enter_context(patch.object(store,'save_experiment',observe_save))
        # This module imports the saver by name at import time. Observe that
        # actual binding too; otherwise completed real inserts are missed by
        # the observer (the original run's two false failure expectations).
        stack.enter_context(patch.object(exploration_views,'save_experiment',observe_save))
        stack.enter_context(patch.object(curated_scan,'POSITIONS',range(1053,1054)))
        for case in cases:
            current.update(case=case['id'],design_call=0);captured.clear();before=count();start=datetime.now(timezone.utc).replace(microsecond=0)
            response=post(case);name=case['id'];payload=payload_from(response)
            (out/'responses').mkdir(exist_ok=True);(out/'responses'/(name+'.body')).write_bytes(response.data)
            rec.check(name+'_response',response.status_code,200,'Actual completed processing request with approved replay.',case['workflow'])
            rec.check(name+'_single_save',len(captured),1,'Exactly one archive insertion per completed request.',case['workflow'])
            if response.status_code!=200 or not captured:continue
            identifier=payload['saved_run_id'];ids[name]=identifier;producer=captured[0];producer_snapshots[name]=producer
            rec.check(name+'_id',bool(re.fullmatch('[0-9a-f]{32}',identifier)),True,'Generated experiment identifier contract.',case['workflow'])
            rec.check(name+'_rowcount',count(),before+1,'One committed row, no hidden duplicate.',case['workflow'])
            record_response=client.get('/experiments/'+identifier+'/record.json');record=strict_json(record_response.text)
            (out/'exports').mkdir(exist_ok=True);(out/'exports'/(name+'.json')).write_bytes(record_response.data)
            rec.check(name+'_json_status',record_response.status_code,200,'Actual JSON download endpoint.',case['workflow'])
            rec.check(name+'_snapshot',record['payload'],producer['payload'],'Entire original producer object captured before save round-trips exactly, supplementing independent numeric manifest.',case['workflow'])
            rec.check(name+'_workflow',record['workflow'],producer['workflow'],'Correct workflow association.',case['workflow'])
            rec.check(name+'_title',record['title'],producer['title'],'Unmodified experiment title.',case['workflow'])
            created=datetime.fromisoformat(record['created']);rec.check(name+'_timestamp',start<=created<=datetime.now(timezone.utc),True,'UTC-aware timestamp generated inside request interval.',case['workflow'])
            rec.check(name+'_metadata_time',record['metadata']['saved_at_utc'],record['created'],'Metadata and row timestamp agree.',case['workflow'])
            rec.check(name+'_status',record['metadata']['status'],'complete','Only completed request was archived.',case['workflow'])
            if name not in ('qc','dataset','motifs'):
                rec.check(name+'_model_count',len(record['metadata']['models']),2 if name=='http_model_comparison' else 1,'Every model has a separate reproducibility identity entry.',case['workflow'])
                for i,identity in enumerate(record['metadata']['models']):
                    rec.check(name+f'_model_identity_{i}',identity,dict(model_id=fixture['identity']['website_label'],checkpoint_sha256=fixture['identity']['checkpoint_sha256'],tasks=[task]),'Independent pinned model/checkpoint/task identity, including each comparison model.',case['workflow'])
            inputs=record['payload'].get('inputs')
            if inputs is not None:
                expected_hash=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
                rec.check(name+'_input_hash',record['metadata']['input_sha256'],expected_hash,'Independent canonical input fingerprint; actual full input snapshot compared separately.',case['workflow'])
            rec.check(name+'_versions',all(record['metadata']['versions'].get(p) for p in ['numpy','torch','grelu','Flask','pandas']),True,'Installed save-time library versions present; original inference metadata remains in payload.',case['workflow'])
            rec.check(name+'_cache',record_response.headers['Cache-Control'],'no-store','Private result downloads are not cached.',case['workflow'])
            rec.check(name+'_paths',str(ROOT) not in record_response.text and '/home/akshadha/.cache/' not in record_response.text,True,'No internal workspace/checkpoint/temp path leaked by normal inference snapshot.',case['workflow'])
            detail=client.get('/experiments/'+identifier)
            (out/'responses'/(name+'.detail.html')).write_bytes(detail.data)
            rec.check(name+'_detail',detail.status_code,200,'Actual history detail reopens.',case['workflow'])
            history=client.get('/experiments',query_string={'q':producer['title'],'workflow':producer['workflow']})
            (out/'responses'/(name+'.history.html')).write_bytes(history.data)
            rec.check(name+'_history_link',('/experiments/'+identifier+'"') in history.text,True,'Actual title/workflow filtered history card links to this exact saved snapshot.',case['workflow'])
            html_report=client.get('/experiments/'+identifier+'/report.html');(out/'exports'/(name+'.report.html')).write_bytes(html_report.data)
            rec.check(name+'_report',html_report.status_code,200,'Actual standalone HTML report download.',case['workflow'])
            rec.check(name+'_report_policy',"default-src 'none'" in html_report.headers.get('Content-Security-Policy',''),True,'Downloaded report has restrictive content policy.',case['workflow'])
            expected=expected_manifest.get(name,{})
            actual=record['payload'].get('results',{})
            if 'predictions' in actual:
                groups=case['expected_calls'];pred=actual['predictions']
                rec.check(name+'_ordered_ids',[p['id'] for p in pred],[r['id'] for r in groups[0]],'Independent original input IDs, including distinct-ID duplicate DNA.',case['workflow'])
                for i,p in enumerate(pred):
                    rec.check(f'{name}_score_{i}',p['scores'][task],score(groups[0][i]),'Independent Gate 4 raw score, not reconstructed from SQLite.',case['workflow'])
                    rec.check(f'{name}_rc_{i}',p['reverse_complement_scores'][task],score(groups[1][i]),'Correct RC sequence/raw association.',case['workflow'])
                if 'comparisons' in actual:
                    expected_effects=[(r['id'],score(r)-score(groups[0][0])) for r in groups[0][1:]]
                    if case['workflow']=='/prioritize':expected_effects=sorted(expected_effects,key=lambda x:-abs(x[1]))
                    rec.check(name+'_effect_ids',[r['alternate'] for r in actual['comparisons']],[r[0] for r in expected_effects],'Stable ranking/reference mapping survives storage.',case['workflow'])
                    rec.check(name+'_effects',[r['delta'] for r in actual['comparisons']],[r[1] for r in expected_effects],'Exact signed alternate-reference effects.',case['workflow'])
            elif name=='http_variant_csv':
                groups=case['expected_calls'];n=len(actual['variants'])
                expected_effects=[score(groups[0][n+i])-score(groups[0][i]) for i in range(n)]
                rec.check(name+'_effects',[r['effects'][task] for r in actual['variants']],expected_effects,'Independent reference block / alternate block alignment.',case['workflow'])
                expected_ids=[groups[0][i]['id'].removesuffix(':reference') for i in sorted(range(n),key=lambda i:-abs(expected_effects[i]))]
                rec.check(name+'_ranks',[r['variant_id'] for r in actual['ranked_variants']],expected_ids,'Actual original IDs remain associated with independently sorted effects.',case['workflow'])
            elif name in ['design','attribution']:
                if name=='design':
                    for key in ['score_before','score_after','history','designed_sequence','task_scores_before','task_scores_after','n_mutations']:
                        rec.check(name+'_'+key,actual['designs'][0][key],expected['results']['designs'][0][key],'Authoritative independently verified Gate 5 result, never read from this test database.',case['workflow'])
                else:
                    for key in ['matrix','positions','top_positions','display_positions','reference_score','matrix_bases']:
                        rec.check(name+'_'+key,actual[key],expected['results'][key],'Authoritative Gate 5 complete processed gradient result exact round-trip.',case['workflow'])
            elif name=='http_mutagenesis':
                groups=case['expected_calls'];ref=score(groups[0][0]);position=int(case['fields']['start']);bases=[b for b in 'ACGT' if b!=groups[0][0]['sequence'][position-1]]
                rec.check(name+'_deltas',actual['positions'][0]['deltas'],{b:score(r)-ref for b,r in zip(bases,groups[1])},'Saved mutation/base mapping from independent raw outputs.',case['workflow'])
            elif name=='http_curated_subset':
                ref=score(case['expected_calls'][0][0]);bases=[b for b in 'ACGT' if b!=curated_scan.LOCI[case['fields']['locus']]['dna'][1053]]
                rec.check(name+'_deltas',record['payload']['positions'][0]['deltas'],{b:score(r)-ref for b,r in zip(bases,case['expected_calls'][1])},'Preserve only the verified one-position subset; not full saturation science.',case['workflow'])
            elif name=='http_motif':
                rec.check(name+'_delta',actual['conditions'][0]['delta'],score(case['expected_calls'][0][1])-score(case['expected_calls'][0][0]),'Independent replacement-reference effect.',case['workflow'])
            elif name=='http_model_comparison':
                rec.check(name+'_differences',[r['raw_difference'] for r in actual['comparisons']],[0.0,0.0],'Same-model/task difference remains exactly zero.',case['workflow'])
            elif name=='qc':rec.check(name+'_ids',[r['id'] for r in record['payload']['records']],expected['record_ids'],'Original descriptive QC IDs.',case['workflow'])
            elif name=='dataset':
                rec.check(name+'_record_count',record['payload']['summary']['records'],2,'Independent two-record dataset fixture.',case['workflow'])
                rec.check(name+'_duplicate_count',record['payload']['summary']['duplicate_records'],1,'Identical DNA with distinct IDs is retained and reported, not dropped.',case['workflow'])
            elif name=='motifs':
                rec.check(name+'_hit',record['payload']['hits'][0],dict(record='α',motif='named',sequence='TATAAA',start=1,end=6,strand='+'),'Independent supplied motif coordinates/identity survive snapshot.',case['workflow'])
            for warning in fallback:
                if name not in ('qc','dataset','motifs'):rec.check(name+'_warning',warning in record['payload'].get('warnings',[]),True,'Actual historical fallback note survives snapshot and JSON download.',case['workflow'])
            independent_matches.append(name)
        # Generated engine CSV/JSON files are real existing capabilities, not
        # invented server CSV routes. Arbitrary allowed IDs are tested here.
        current.update(case='engine',design_call=0)
        unusual=['=SUM(1,2)','  +formula','α,"quoted"\nline','-identifier','plain','＝SUM(1,2)','\tidentifier']
        records=[dict(id=identifier,sequence=first) for identifier in unusual]
        from engine import execute_platform_workflow
        output=out/'engine-files';outcome=execute_platform_workflow('predict',dict(model=MODELS['human'],device='cpu',sequences=records,n_shuffles=0,output_dir=str(output)))
        rec.check('engine_predict_ok',outcome['ok'],True,'Actual file-based processing path with approved tensors.','generated CSV')
        with (output/'predictions.csv').open(newline='') as stream:rows=list(csv.reader(stream))
        rec.check('csv_predict_headers',rows[0],['id','length','gc_content','score_'+task],'Actual generated header order.','generated CSV')
        def safe_text(value):
            return "'"+value if value not in ('+','-') and (value.startswith(('\t','\r','\n')) or value.lstrip().startswith(('=','+','-','@','＝','＋','－','＠'))) else value
        rec.check('csv_quoted_ids',[r[0] for r in rows[1:]],[safe_text(x) for x in unusual],'Independent spreadsheet-safe policy; JSON preserves the canonical original.','generated CSV')
        rec.check('engine_json_original_ids',[r['id'] for r in strict_json((output/'predict_result.json').read_text())['results']['predictions']],unusual,'Unicode/comma/quote/newline IDs remain exact in JSON.','generated JSON')
        gc=(first.count('G')+first.count('C'))/sum(first.count(b) for b in 'ACGT')
        rec.check('csv_gc_precision',[float(r[2]) for r in rows[1:]],[gc]*len(unusual),'Full precision, no four-decimal GC rounding.','generated CSV')
        rec.check('csv_score_precision',[float(r[3]) for r in rows[1:]],[float(lookup[first][0,0])]*len(unusual),'CSV model scores round-trip exactly.','generated CSV')
        variant_case=next(c for c in fixture['cases'] if c['id']=='http_variant_csv');csv_input=out/'variants-input.csv';csv_input.write_text(variant_case['fields']['text'])
        outcome=execute_platform_workflow('vep',dict(model=MODELS['human'],device='cpu',variants_csv=str(csv_input),n_shuffles=0,output_dir=str(out/'engine-vep')))
        with (out/'engine-vep/vep_results.csv').open(newline='') as stream:
            reader=csv.DictReader(stream);rows=list(reader);headers=reader.fieldnames
        rec.check('csv_variant_headers',headers,['variant_id','ref','alt','status','top_task','top_effect','top_task_pvalue','rc_sign_agreement','effect_'+task],'Actual generated variant column contract.','generated CSV')
        groups=variant_case['expected_calls'][0];n=3
        rec.check('csv_variant_order',[r['variant_id'] for r in rows],['variant_1','variant_1057','variant_2114'],'Generated CLI file source ordering (UI ranking is a separate representation).','generated CSV')
        rec.check('csv_variant_effects',[float(r['effect_'+task]) for r in rows],[score(groups[n+i])-score(groups[i]) for i in range(n)],'Independent signed effects, full numeric precision.','generated CSV')
        rec.check('csv_optional_blank',[r['top_task_pvalue'] for r in rows],['']*3,'Unperformed empirical test is an empty CSV optional field, never zero.','generated CSV')
        # Simulated actual database failures cannot generate completed records.
        with sqlite3.connect(out/'state/experiments.db') as db:db.execute("CREATE TRIGGER gate6_deny BEFORE INSERT ON experiments BEGIN SELECT RAISE(ABORT,'Gate6 write fault'); END")
        before=count();case=next(c for c in cases if c['id']=='http_variant_1');response=post(case)
        rec.check('write_failure_visible',response.status_code,500,'Injected real SQLite INSERT failure, not a fabricated HTTP success.','errors')
        rec.check('write_failure_rollback',count(),before,'No partial/completed row from failed write.','errors')
        with sqlite3.connect(out/'state/experiments.db') as db:db.execute('DROP TRIGGER gate6_deny')
        # Repeat is intentionally a new experiment; original remains unchanged.
        identifier=ids['http_fasta'];original=client.get('/experiments/'+identifier+'/record.json').data
        saved=strict_json(original);current['case']='repeat';before=count()
        response=client.post('/predict',data={'csrf_token':csrf,**saved['form']});new_payload=payload_from(response)
        rec.check('repeat_new_id',new_payload['saved_run_id']!=identifier,True,'Intentional rerun generates distinct identity.','history')
        rec.check('repeat_one_new_row',count(),before+1,'One row for one repeat request.','history')
        rec.check('repeat_original_unchanged',client.get('/experiments/'+identifier+'/record.json').data.decode(),original.decode(),'Original snapshot immutable after repeat.','history')
    save(out/'producer-snapshots-before-storage.json',producer_snapshots)
    save(out/'experiment-ids.json',ids)
    # Actual second interpreter / app instance, same isolated DB and guest cookie.
    save(out/'restart-request.json',dict(ids=list(ids.values()),cookie=client.get_cookie('session').value))
    process=subprocess.run([sys.executable,'-I',str(Path(__file__).resolve()),'--restart',str(out)],cwd='/tmp',capture_output=True,text=True,timeout=90)
    (out/'restart-process.log').write_text(process.stdout+process.stderr)
    rec.check('restart_process',process.returncode,0,'Fresh OS process imports a new Flask application and opens existing SQLite.','restart')
    statuses=json.loads((out/'restart/statuses.json').read_text())
    for name,identifier in ids.items():
        record=strict_json((out/'restart'/f'{identifier}.json').read_text())
        rec.check(name+'_restart_status',statuses[identifier],dict(json=200,detail=200),'Independent process real endpoints reopen record.','restart')
        rec.check(name+'_restart_payload',record['payload'],producer_snapshots[name]['payload'],'Exact original pre-save producer object after real restart.','restart')
    # Guest and authenticated-account isolation uses only test-owned fixtures.
    other=app.test_client();other.get('/experiments')
    for suffix in ['', '/record.json','/report.html']:
        rec.check('other_guest_'+(suffix.replace('/','').replace('.','_') or 'detail'),other.get('/experiments/'+ids['http_fasta']+suffix).status_code,404,'Different session cannot access known ID.','security')
    for identifier in ['missing','../private','%2e%2e%2fprivate','F'*32,'0'*32]:
        rec.check('invalid_id_'+str(len(rec.rows)),client.get('/experiments/'+identifier+'/record.json').status_code,404,'Bad/missing/download traversal IDs reveal no result.','security')
    rec.check('unsupported_csv_route',client.get('/experiments/'+ids['http_fasta']+'/record.csv').status_code,404,'No server CSV endpoint exists; capability is not invented.','capabilities')
    with client:
        client.get('/experiments')
        rec.mode='SYNTHETIC_SERIALIZATION_ONLY'
        payload=dict(mode=rec.mode,zero=0,negative=-2.25,small=1e-30,large=1e30,negative_zero=-0.0,flag=False,
            null=None,empty='',ids=['α,quote"\nline','=unsafe'],array=np.array([[1.,-2.],[0.,.25]],np.float32),
            scalar32=np.float32(.5),scalar64=np.float64(math.pi),integer=np.int64(9007199254740993),settings=None)
        expected=dict(mode=rec.mode,zero=0,negative=-2.25,small=1e-30,large=1e30,negative_zero=-0.0,flag=False,
            null=None,empty='',ids=['α,quote"\nline','=unsafe'],array=[[1.,-2.],[0.,.25]],scalar32=.5,scalar64=math.pi,integer=9007199254740993,settings=None)
        save(out/'synthetic-expected.json',expected)
        identifier=store.save_experiment('Synthetic "<script>" α','synthetic-serialization',payload,'/qc',
            {'csrf_token':'secret','PASSWORD':'secret','file':'/private/secret.fa','output_dir':'/private','empty':'','zero':0,'flag':False,'null':None})
        rec.check('numpy_json_roundtrip',client.get('/experiments/'+identifier+'/record.json').get_json()['payload'],expected,'Independent literal manifest; exact types/float hex/order/Unicode/null distinction.','serialization')
        stored=store.experiment(identifier)
        rec.check('form_safe_fields',stored['form'],dict(empty='',zero='0'),'HTML replay form intentionally omits unchecked/null controls and secret/file keys; payload does not.','security')
        rec.check('unknown_model_metadata',stored['metadata']['models'],[],'No fabricated model identity for a synthetic snapshot.','metadata')
        before=count()
        for index,value in enumerate([float('nan'),float('inf'),float('-inf'),np.float32(np.nan),np.array([1,np.inf])]):
            error=None
            try:store.save_experiment('invalid','synthetic',dict(value=value),'/qc',{})
            except (ValueError,TypeError) as e:error=str(e)
            rec.check('reject_nonfinite_'+str(index),error is not None,True,'Invalid serialization must reject before commit, never replace with null or zero.','serialization')
        rec.check('invalid_serialization_no_rows',count(),before,'No partial rows from five rejected values.','serialization')
        for index,bad in enumerate([{'ok':False},{'status':'failed'},{'status':'running'},{'status':'queued'}]):
            rejected=False
            try:store.save_experiment('must not complete','synthetic',bad,'/qc',{})
            except ValueError:rejected=True
            rec.check('reject_incomplete_'+str(index),rejected,True,'Failed/incomplete producer cannot become a completed archive.','errors')
        rec.check('incomplete_no_rows',count(),before,'Rejected status values never insert a completed experiment.','errors')
        partial=store.save_experiment('Explicit partial','synthetic',{'status':'partial','value':1},'/qc',{})
        rec.check('partial_status_explicit',store.experiment(partial)['metadata']['status'],'partial','Explicit partial snapshots never receive a complete label.','errors')
        duplicate_id='a'*32
        with patch.object(store.secrets,'token_hex',return_value=duplicate_id):store.save_experiment('first','qc',dict(value=1),'/qc',{})
        before=count();error=False
        try:
            with patch.object(store.secrets,'token_hex',return_value=duplicate_id):store.save_experiment('second','qc',dict(value=2),'/qc',{})
        except sqlite3.IntegrityError:error=True
        rec.check('duplicate_id_rejected',error,True,'Real SQLite primary key collision, not overwrite.','errors')
        rec.check('duplicate_id_original',store.experiment(duplicate_id)['payload'],dict(value=1),'Original unchanged after collision.','errors')
        rec.check('duplicate_id_count',count(),before,'No duplicate/partial row committed.','errors')
        # Corrupt the isolated raw DB, not any genuine experiment.
        with sqlite3.connect(out/'state/experiments.db') as db:db.execute('UPDATE experiments SET payload=? WHERE id=?',('{"value":NaN}',identifier))
        response=client.get('/experiments/'+identifier+'/record.json')
        rec.check('corruption_visible',response.status_code,500,'Stored nonstandard JSON is rejected rather than downloaded successfully.','errors')
        rec.check('corruption_no_leak','/private/' not in response.text and 'NaN' not in response.text,True,'Public failure body excludes sensitive data/paths.','security')
        with sqlite3.connect(out/'state/experiments.db') as db:actual_corrupt=db.execute('SELECT payload FROM experiments WHERE id=?',(identifier,)).fetchone()[0]
        rec.check('corruption_retained',actual_corrupt,'{"value":NaN}','No silent data repair or deletion.','errors')
        rec.check('corrupt_history_status','unreadable snapshot data' in client.get('/experiments',query_string={'q':'Synthetic'}).text,True,'History does not label a corrupt snapshot as a readable completed record.','history')
        intact=store.save_experiment('integrity fixture','qc',dict(value=1),'/qc',{})
        with sqlite3.connect(out/'state/experiments.db') as db:db.execute('UPDATE experiments SET payload=? WHERE id=?',('{"value":2}',intact))
        rec.check('valid_json_tamper_rejected',client.get('/experiments/'+intact+'/record.json').status_code,500,'Canonical payload fingerprint detects valid-JSON numerical corruption.','errors')
        for index,text in enumerate(['{"value":1e309}','{"value":','[1,2]']):
            broken=store.save_experiment('Malformed legacy fixture','synthetic',{'value':1},'/qc',{})
            with sqlite3.connect(out/'state/experiments.db') as db:db.execute('UPDATE experiments SET payload=?,metadata=? WHERE id=?',(text,'{}',broken))
            rec.check('legacy_corruption_'+str(index),client.get('/experiments/'+broken+'/record.json').status_code,500,'Reject overflow exponent/malformed JSON/wrong object root even without legacy checksum.','errors')
            with sqlite3.connect(out/'state/experiments.db') as db:retained=db.execute('SELECT payload FROM experiments WHERE id=?',(broken,)).fetchone()[0]
            rec.check('legacy_corruption_retained_'+str(index),retained,text,'Failed read never silently rewrites the original bad record.','errors')
        # Malicious text stays exact in JSON but never becomes executable markup.
        title='<script>alert("x")</script>';danger='unsafe\r\nX-Evil: yes/../../α'
        identifier=store.save_experiment(title,danger,dict(results={}),'/not-a-replay-url',dict(any='value'))
        response=client.get('/experiments/'+identifier+'/record.json')
        rec.check('filename_safe_status',response.status_code,200,'Unsafe metadata text cannot break attachment headers.','security')
        disposition=response.headers['Content-Disposition']
        rec.check('filename_safe_header',not any(c in disposition for c in ['\r','\n','/','\\']),True,'Download filename is a safe basename; metadata retains original workflow.','security')
        rec.check('workflow_unicode_preserved',response.get_json()['workflow'],danger,'Sanitize filename, never silently modify stored identity.','serialization')
        detail=client.get('/experiments/'+identifier)
        rec.check('history_title_escaped',title not in detail.text and '&lt;script&gt;' in detail.text,True,'Actual rendered title is escaped.','security')
        rec.check('replay_url_restricted',store.experiment(identifier)['path'],'','Only existing supported replay actions are retained.','security')
        # History ordering, pagination, search and filters operate on real rows.
        ordered=[]
        for i in range(34):ordered.insert(0,store.save_experiment(f'Gate6 page {i}','gate6-pagination',dict(index=i),'/qc',{}))
        page1=client.get('/experiments?workflow=gate6-pagination');page2=client.get('/experiments?workflow=gate6-pagination&page=2')
        def history_ids(text):return list(dict.fromkeys(re.findall(r'href="/experiments/([0-9a-f]{32})"',text)))
        rec.check('pagination_first',history_ids(page1.text),ordered[:30],'Actual 30-row page boundary, newest tie-broken by insertion rowid.','history')
        rec.check('pagination_second',history_ids(page2.text),ordered[30:],'Remaining four records, no drops/duplicates.','history')
        searched=client.get('/experiments?q=Gate6%20page%2033&workflow=gate6-pagination')
        rec.check('history_search',len(history_ids(searched.text)),1,'Implemented title/workflow filter only.','history')
        rec.check('history_status_label','complete' in page1.text,True,'Actual history cards show completion status.','history')
        from experiment_views import display_time
        app.config['DISPLAY_TIMEZONE']='Asia/Kolkata'
        rec.check('timezone_rollover',display_time('2024-02-29T23:45:00+00:00'),'01 Mar 2024 · 05:15 AM IST','Independent fixed UTC+05:30 leap-day rollover.','history')
        app.config['DISPLAY_TIMEZONE']='UTC'
        # Deletion requires CSRF and deletes only the requested owned snapshot.
        target=ordered[0];before=count()
        rec.check('delete_requires_csrf',client.post('/experiments/'+target+'/delete').status_code,400,'No deletion without valid token.','security')
        rec.check('delete_owned',client.post('/experiments/'+target+'/delete',data={'csrf_token':csrf}).status_code,302,'Explicit owner deletion, test records only.','history')
        rec.check('delete_count',count(),before-1,'Only one row deleted.','history')
        rec.check('deleted_id_404',client.get('/experiments/'+target+'/record.json').status_code,404,'Deleted record is no longer exposed.','history')
    with client:
        client.get('/experiments')
        import account_views
        accounts=account_views.database()
        accounts.execute("INSERT INTO users (name,email,password,active,session_version) VALUES ('A','gate6-a@example.invalid','unused-test-hash',1,0)")
        uid_a=accounts.execute('SELECT last_insert_rowid()').fetchone()[0]
        accounts.execute("INSERT INTO users (name,email,password,active,session_version) VALUES ('B','gate6-b@example.invalid','unused-test-hash',1,0)")
        uid_b=accounts.execute('SELECT last_insert_rowid()').fetchone()[0];accounts.commit()
        own_id=store.save_experiment('Account A fixture','qc',dict(mode='SYNTHETIC_ACCESS_FIXTURE',value=1),'/qc',{},owner_id='user:'+str(uid_a))
        a=app.test_client();b=app.test_client()
        for target,uid in [(a,uid_a),(b,uid_b)]:
            with target.session_transaction() as sess:sess['user_id']=uid;sess['session_version']=0
        rec.check('account_owner_access',a.get('/experiments/'+own_id+'/record.json').status_code,200,'Actual active-account authorization with isolated seeded test accounts.','security')
        rec.check('account_cross_access',b.get('/experiments/'+own_id+'/record.json').status_code,404,'A different authenticated account cannot read the record.','security')
        original_database=store.database
        class CommitFailure:
            def __init__(self):self.db=original_database()
            def __getattr__(self,name):return getattr(self.db,name)
            def commit(self):raise sqlite3.OperationalError('Gate6 commit interruption')
        before=count();failed=False
        try:
            with patch.object(store,'database',lambda:CommitFailure()):store.save_experiment('Uncommitted','qc',dict(value=3),'/qc',{})
        except sqlite3.OperationalError:failed=True
        rec.check('interrupted_commit_visible',failed,True,'Real INSERT followed by failed commit; method cannot return an ID.','errors')
        rec.check('interrupted_commit_rollback',count(),before,'Connection close rolls back uncommitted inserted row.','errors')
        with patch.object(store,'database',side_effect=sqlite3.OperationalError('Gate6 private read fault /private/secret.fa')):
            response=client.get('/experiments/'+ids['http_fasta']+'/record.json')
        rec.check('read_failure_visible',response.status_code,500,'Simulated DB read failure through actual download route.','errors')
        rec.check('read_failure_private','/private/' not in response.text,True,'Error body does not expose private diagnostic paths.','security')
        # Exercise metadata-column migration on a separate legacy-format DB.
        legacy=out/'state/legacy.db'
        with sqlite3.connect(legacy) as db:
            db.execute('CREATE TABLE experiments (id TEXT PRIMARY KEY,owner TEXT NOT NULL,title TEXT NOT NULL,workflow TEXT NOT NULL,created TEXT NOT NULL,payload TEXT NOT NULL,path TEXT NOT NULL,form TEXT NOT NULL,presentation TEXT NOT NULL)')
            old_row=('b'*32,owner,'Legacy fixture','qc','2024-02-29T12:00:00+00:00','{"value":1}','/qc','{}','{}')
            db.execute('INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?)',old_row)
        original_path=app.config['HISTORY_DATABASE'];app.config['HISTORY_DATABASE']=str(legacy)
        try:
            migrated=store.experiment('b'*32)
            rec.check('legacy_payload_unchanged',migrated['payload'],{'value':1},'Schema upgrade preserves original result, does not invent new provenance.','migration')
            rec.check('legacy_metadata_unknown',migrated['metadata'],{},'No retrospective inference identity/version/timestamp fabricated.','migration')
            with sqlite3.connect(legacy) as db:preserved=db.execute('SELECT id,owner,title,workflow,created,payload,path,form,presentation FROM experiments').fetchone()
            rec.check('legacy_original_columns',list(preserved),list(old_row),'All nine original columns unchanged by metadata migration.','migration')
            rec.check('legacy_history_unknown','unknown (legacy snapshot)' in client.get('/experiments').text,True,'Legacy status remains explicitly unknown, never silently complete.','migration')
        finally:app.config['HISTORY_DATABASE']=original_path
    # File-backed job lifecycle with synthetic tasks, never model training.
    rec.mode='SYNTHETIC_JOB_LIFECYCLE'
    with client:
        client.get('/experiments')
        started=threading.Event();release=threading.Event()
        task_result=dict(mode=rec.mode,value=-.25,warnings=['synthetic task, no inference'])
        def task_fn(jobid):started.set();release.wait(4);return task_result.copy()
        with patch.object(threading.Thread,'start',lambda self:None):jobid=lab_jobs.submit_job('Gate6 synthetic task',task_fn)
        rec.check('job_queued',client.get('/api/jobs/'+jobid).get_json()['state'],'queued','Actual initial persisted status before starting worker.','jobs')
        lab_jobs.workers[jobid].start();started.wait(4)
        rec.check('job_running',client.get('/api/jobs/'+jobid).get_json()['state'],'running','Actual running worker waits on test barrier.','jobs')
        release.set()
        for _ in range(80):
            response=client.get('/api/jobs/'+jobid);state=response.get_json()
            if state['state'] in ('complete','failed') and jobid not in lab_jobs.workers:break
            time.sleep(.025)
        rec.check('job_complete',state['state'],'complete','Actual thread/state-file/database archive lifecycle.','jobs')
        archive=client.get('/experiments/'+state['result']['saved_run_id']+'/record.json').get_json()
        rec.check('job_archive_status',archive['metadata']['status'],'complete','Two-store commit promotes pending snapshot after status-file success.','jobs')
        rec.check('job_exact_result',archive['payload'],task_result,'Independent synthetic task literal, no model prediction claim.','jobs')
        rec.check('job_owner_hidden','owner' not in state,True,'Status response omits owner token.','security')
        rec.check('job_other_owner',other.get('/api/jobs/'+jobid).status_code,404,'Other guest cannot access job result.','security')
        rec.check('missing_checkpoint',client.get('/jobs/'+jobid+'/checkpoint').status_code,404,'No checkpoint was trained/generated by this task.','files')
        artifact=b'Gate 6 routing fixture; NOT a trained checkpoint.'
        lab_jobs.update_job(jobid,result={**state['result'],'checkpoint':'../../outside-file'})
        rec.check('missing_output_file',client.get('/jobs/'+jobid+'/checkpoint').status_code,404,'Missing output file is explicit, not fabricated download success.','files')
        (lab_jobs.job_path(jobid)/'model.pt').write_bytes(artifact)
        response=client.get('/jobs/'+jobid+'/checkpoint')
        rec.check('job_file_bytes',response.data.decode(),artifact.decode(),'Actual controlled-file download, not a model download or trained model validation.','files')
        rec.check('job_file_name','grelu-trained-model.pt' in response.headers.get('Content-Disposition',''),True,'Fixed safe attachment filename; result path hint is never followed.','files')
        rec.check('job_file_other_owner',other.get('/jobs/'+jobid+'/checkpoint').status_code,404,'File route enforces same job ownership.','security')
        collision=False;old_status=(lab_jobs.job_path(jobid)/'status.json').read_bytes()
        try:
            with patch.object(lab_jobs.secrets,'token_hex',return_value=jobid):lab_jobs.submit_job('collision',lambda jobid:{})
        except lab_jobs.JobStateError:collision=True
        rec.check('job_duplicate_id',collision,True,'Existing directory collision is rejected without overwrite.','jobs')
        rec.check('job_duplicate_preserved',(lab_jobs.job_path(jobid)/'status.json').read_bytes().decode(),old_status.decode(),'Existing status bytes unchanged by repeated ID allocation.','jobs')
        missing='e'*32
        folder=out/'state/lab/jobs'/missing
        rec.check('missing_job_404',client.get('/api/jobs/'+missing).status_code,404,'Unknown job lookup.','jobs')
        rec.check('missing_job_no_directory',folder.exists(),False,'Read-only lookup no longer allocates job directories.','jobs')
        path=lab_jobs.job_path(jobid)/'status.json';old=path.read_bytes()
        real_replace=Path.replace
        def fail_replace(self,target):
            if self.name=='status.tmp':raise OSError('Gate6 simulated rename failure')
            return real_replace(self,target)
        failed=False
        try:
            with patch.object(Path,'replace',fail_replace):lab_jobs.update_job(jobid,message='not committed')
        except OSError:failed=True
        rec.check('job_atomic_failure',failed,True,'Failure occurs after temp write, before atomic rename.','jobs')
        rec.check('job_old_state_intact',path.read_bytes().decode(),old.decode(),'Prior complete status file remains byte-for-byte intact.','jobs')
        rec.check('job_temp_cleaned',path.with_suffix('.tmp').exists(),False,'Uncommitted temp status file cleaned safely.','jobs')
        def bad_task(jobid):raise RuntimeError('/private/secret.fa SECRET_DNA')
        failed_id=lab_jobs.submit_job('Gate6 failed task',bad_task)
        for _ in range(80):
            state=client.get('/api/jobs/'+failed_id).get_json()
            if state['state']=='failed' and failed_id not in lab_jobs.workers:break
            time.sleep(.025)
        rec.check('job_failed_state',state['state'],'failed','Failed task never reports completion.','jobs')
        rec.check('job_error_sanitized','SECRET_DNA' not in json.dumps(state) and '/private/' not in json.dumps(state),True,'Worker error details remain in server log, not API.','security')
        corrupt=json.loads(old_status);corrupt['result']=None
        path.write_text(json.dumps(corrupt));corrupt_bytes=path.read_bytes()
        rec.check('corrupt_job_visible',client.get('/api/jobs/'+jobid).status_code,500,'Malformed completed job state cannot be reported as success.','jobs')
        rec.check('corrupt_job_retained',path.read_bytes().decode(),corrupt_bytes.decode(),'Original corrupt bytes retained for diagnosis.','jobs')
        corrupt=json.loads(old_status);corrupt['result']={'value':'exponent-marker'}
        overflow_text=json.dumps(corrupt).replace('"exponent-marker"','1e309');path.write_text(overflow_text)
        rec.check('overflow_job_visible',client.get('/api/jobs/'+jobid).status_code,500,'Valid JSON exponent overflow cannot leak Infinity as a successful complete result.','jobs')
        rec.check('overflow_job_retained',path.read_text(),overflow_text,'Overflow corruption retained for diagnosis.','jobs')
        path.write_bytes(old_status)
        interrupted='f'*32;interrupted_state=json.loads(old_status)
        interrupted_state.update(id=interrupted,state='running',result=None,worker_pid=99999999)
        interrupted_path=lab_jobs.job_path(interrupted,create=True)/'status.json';interrupted_path.write_text(json.dumps(interrupted_state))
        rec.check('interrupted_worker_failed',client.get('/api/jobs/'+interrupted).get_json()['state'],'failed','Dead process liveness check marks interrupted work failed; signal 0 only.','jobs')
        # A status-file failure after archive insertion must not leave a completed row.
        original_update=lab_jobs.update_job;before=count()
        def fail_complete(identifier,**changes):
            if changes.get('state')=='complete':raise OSError('Gate6 simulated final status failure')
            return original_update(identifier,**changes)
        with patch.object(lab_jobs,'update_job',fail_complete):
            failed_id=lab_jobs.submit_job('Gate6 incomplete archive',lambda jobid:dict(mode=rec.mode,value=1))
            for _ in range(100):
                state=client.get('/api/jobs/'+failed_id).get_json()
                if state['state']=='failed' and failed_id not in lab_jobs.workers:break
                time.sleep(.025)
        rec.check('partial_job_failed',state['state'],'failed','Post-insertion status failure is visible.','jobs')
        rec.check('partial_archive_rollback',count(),before,'Owned pending archive removed; no misleading completed experiment.','jobs')
        before=count()
        with patch.object(lab_jobs,'update_job',fail_complete),patch.object(store,'delete_experiment',side_effect=sqlite3.OperationalError('Gate6 cleanup fault')):
            pending_id=lab_jobs.submit_job('Gate6 pending archive fixture',lambda jobid:dict(mode=rec.mode,value=2))
            for _ in range(100):
                pending=client.get('/api/jobs/'+pending_id).get_json()
                if pending['state']=='failed' and pending_id not in lab_jobs.workers:break
                time.sleep(.025)
        rec.check('double_fault_job_failed',pending['state'],'failed','Two-store failure is not reported complete.','jobs')
        rec.check('double_fault_cleanup_flag',pending.get('archive_cleanup_pending'),True,'Failed compensation is explicit in job status.','jobs')
        with sqlite3.connect(out/'state/experiments.db') as db:pending_archive=db.execute('SELECT id FROM experiments WHERE title=?',('Gate6 pending archive fixture',)).fetchone()[0]
        record=client.get('/experiments/'+pending_archive+'/record.json').get_json()
        rec.check('double_fault_archive_pending',record['metadata']['status'],'pending','Retained partial archive cannot claim completion after compensation failure.','jobs')
        store.delete_experiment(pending_archive,owner_id=owner)
        rec.check('double_fault_cleanup_test_only',count(),before,'Explicit cleanup only removes this test-owned pending row.','jobs')
    rec.mode='LIMITATIONS'
    for name,why,workflow in [('browser','No installed browser runtime / Playwright / Selenium / Node. Static client download code is not executed browser evidence.','browser'),
        ('client_csv_downloads','CSV Blob buttons execute only in JavaScript; no server CSV history endpoint exists. Generated engine CSV files are tested separately.','CSV'),
        ('raw_tensor_archiving','Snapshots contain processed results; native raw tensors/checkpoint weights are not archived by this app. Attribution matrices already in processed results are retained.','capabilities'),
        ('training_installation','No training, fine-tuning, model download or installation tasks executed; generic job protocol uses explicit synthetic tasks.','jobs'),
        ('full_scientific_workflows','Prior partial saturation/motif/design/model-comparison limitations remain. Saving/exporting does not validate untested science.','scope')]:rec.skip(name,why,workflow)
    save(out/'results.json',dict(checks=rec.rows,counts={s:sum(r['status']==s for r in rec.rows) for s in ['PASS','FAIL','SKIP','BLOCKED']},
        experiment_ids=ids,workflow_cases=[dict(id=c['id'],workflow=c['workflow']) for c in cases],step2_started=False,
        browser_executed=False,checkpoint_modified=False,training_executed=False))


if __name__=='__main__':
    if sys.argv[1]=='--restart':restart(Path(sys.argv[2]).resolve())
    else:main(Path(sys.argv[1]).resolve())
