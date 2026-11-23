"""Gate 2: real input handlers, stopped at string-to-inference boundaries.

No predictions, tensors, model training, or biological interpretation are used.
Run from the repository root with the project .venv Python.
"""
import base64
from contextlib import ExitStack
import copy
import csv
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import inspect
import io
import json
import os
import random
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GRELU_SECRET_KEY', 'isolated-gate2-input-verification')
os.environ['HF_HUB_OFFLINE'] = '1'
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(name,'1')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False, default=str)+'\n')


def protected():
    directories = ['Testing/testing-phase-1', 'test-results/gm12878-biological',
                   'test-results/gm12878-verified', 'Testing/testing-phase-2/input-output/runs']
    return {str(p.relative_to(ROOT)): digest(p) for name in directories
            for p in sorted((ROOT/name).rglob('*')) if p.is_file()}


def dna(n):
    # Nonperiodic context exposes unintended shifts/crops. No model scores.
    return 'ACGT'[:n] if n<=4 else 'ACGT'+''.join(random.Random(2901).choices('ACGT',k=n-4))


def changed(seq, position=1):
    i = position-1
    return seq[:i] + {'A':'C','C':'G','G':'T','T':'A'}[seq[i]] + seq[i+1:]


def csv_text(fields, rows):
    stream = io.StringIO(); writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(fields); writer.writerows(rows)
    return stream.getvalue()


def cases():
    tests = []
    def add(name, path, fields=None, *, expected='reject', final=None, ids=None, upload=None,
            json_body=None, notes='', scope='website', extra=None):
        tests.append({'id':name, 'path':path, 'fields':fields or {}, 'expected':expected,
                      'final_sequences':final, 'expected_ids':ids, 'upload':upload,
                      'json_body':json_body, 'notes':notes, 'scope':scope, **(extra or {})})
    base = dna(2114)
    for kind, sequence, expected in [
        ('uppercase',base,'boundary'),('lowercase',base.lower(),'boundary'),
        ('mixed_case',base[:50].lower()+base[50:],'boundary'),('empty','','reject'),
        ('wrapped',' \t'+base[:100]+' \r\n'+base[100:]+'\t ','boundary'),
        ('internal_space',base[:2]+' '+base[2:],'reject'),('invalid_X',base[:3]+'X'+base[4:],'reject'),
        ('number',base[:3]+'1'+base[4:],'reject'),('punctuation',base[:3]+'-'+base[4:],'reject'),
        ('N_context',base[:3]+'N'+base[4:],'boundary'),('all_N','N'*2114,'boundary'),
        ('unicode_base',base[:3]+'Ａ'+base[4:],'reject'),
        ('unicode_separator',base[:10]+'\u2028'+base[10:],'reject'),
        ('unicode_edge','\u00a0'+base,'reject'),('zero_width',base[:4]+'\u200b'+base[4:],'reject'),
        ('over_sequence_limit','A'*200001,'reject')]:
        final = base if kind in ['lowercase','mixed_case','wrapped'] else sequence
        add('dna_'+kind,'/predict',{'model':'human','fasta':sequence},expected=expected,
            final=[final] if expected=='boundary' else None,ids=['sequence_1'])
    for n in [0,1,200,2113,2114,2115,5000]:
        seq = dna(n)
        for path in ['/predict','/qc','/attribution','/mutagenesis','/design','/motif-simulation']:
            fields = {'model':'human', 'fasta' if path in ['/predict','/qc'] else 'text':seq,
                      'start':'1','end':'1','target':'0','motif':'TATAAA','positions':'1',
                      'orientation':'+','gaps':'0','iterations':'1','candidates':'1'}
            if path=='/motif-simulation':fields['controls']='on'
            outcome = 'response' if path=='/qc' and n else ('boundary' if n==2114 else 'reject')
            add(f'length_{path[1:]}_{n}',path,fields,expected=outcome,
                final=[seq] if outcome!='reject' else None, ids=['sequence_1'],
                extra={'motif_input_batch':True} if path=='/motif-simulation' and n==2114 else None)
        for path in ['/allele-specific','/prioritize','/multi-allelic']:
            alt = changed(seq) if seq else ''
            text = f'>ref\n{seq}\n>alt\n{alt}\n'
            add(f'length_{path[1:]}_{n}',path,{'model':'human','fasta':text},
                expected='boundary' if n==2114 else 'reject',final=[seq,alt] if n==2114 else None,ids=['ref','alt'])
        add(f'length_compare_api_{n}','/api/compare',json_body={'model':'human','reference':seq,'alternate':changed(seq) if seq else ''},
            expected='boundary' if n==2114 else 'reject',final=[seq,changed(seq)] if n==2114 else None,
            ids=['reference','alternate'])
    fasta_cases = [
        ('single',f'>record_A description\n{base}\n','boundary',[base],['record_A']),
        ('multi',f'>Z\n{base}\n>A\n{changed(base)}\n','boundary',[base,changed(base)],['Z','A']),
        ('wrapped',f'>wrap\n{base[:99]}\n{base[99:]}\n','boundary',[base],['wrap']),
        ('windows',f'>win\r\n{base[:99]}\r\n{base[99:]}\r\n','boundary',[base],['win']),
        ('unicode_identifier',f'>sample_α description\n{base}\n','boundary',[base],['sample_α']),
        ('duplicate',f'>same\n{base}\n>same\n{base}\n','reject',None,None),
        ('missing_identifier',f'>\n{base}\n','reject',None,None),
        ('preamble',f'ACGT\n>record\n{base}\n','reject',None,None),
        ('empty_record',f'>empty\n>full\n{base}\n','reject',None,None),
        ('invalid',f'>bad\n{base[:-1]}!\n','reject',None,None),
        ('mixed_lengths',f'>full\n{base}\n>short\nACGT\n','reject',None,None),
        ('thirteen_records',''.join(f'>r{i}\n{base}\n' for i in range(13)),'reject',None,None),
        ('raw_without_header',base,'boundary',[base],['sequence_1'])]
    for name, text, outcome, final, ids in fasta_cases:
        for uploaded in [False,True]:
            add('fasta_'+name+('_upload' if uploaded else '_text'),'/predict',
                {'model':'human',**({} if uploaded else {'fasta':text})},expected=outcome,final=final,ids=ids,
                upload={'field':'file','filename':name+'.fasta','text':text} if uploaded else None)
    add('fasta_upload_bom','/predict',{'model':'human'},expected='boundary',final=[base],ids=['bom'],
        upload={'field':'file','filename':'bom.fasta','text':'\ufeff>bom\n'+base})
    add('fasta_invalid_utf8','/predict',{'model':'human'},upload={'field':'file','filename':'bad.fasta','bytes_base64':'//4='})
    add('payload_too_large','/predict',{'model':'human','fasta':'A'*601000},extra={'expected_http':413})
    add('missing_field','/predict',{'model':'human'})
    for value in [None,7,[],{},True]:
        add('json_malformed_reference_'+str(type(value).__name__),'/api/compare',
            json_body={'model':'human','reference':value,'alternate':base})
    add('json_missing_reference','/api/compare',json_body={'model':'human','alternate':base})
    add('json_genomic_fields_not_supported','/api/compare',json_body={'model':'human','reference':base,
        'alternate':changed(base),'assembly':'hg19','chrom':'chr1','pos':1})
    add('variant_form','/variants',{'model':'human','reference':base.lower(),'alternate':changed(base).lower()},
        expected='boundary',final=[base,changed(base)],ids=['reference','alternate'])
    for p in [1,1057,2114]:
        text=csv_text(['variant_id','sequence','pos','ref','alt'],[['001',base,p,base[p-1],changed(base,p)[p-1]]])
        for upload in [False,True]:
            add(f'variant_csv_{p}_'+('upload' if upload else 'text'),'/batch-variants',
                {'model':'human','shuffles':'0',**({} if upload else {'text':text})},expected='boundary',
                final=[base,changed(base,p)],
                upload={'field':'file','filename':'variants.csv','text':text} if upload else None,
                extra={'variant_ids':['001'],'variant_position':p})
    for kind, pos, ref, alt in [('zero',0,'A','C'),('outside',2115,'A','C'),('mismatch',1,'C','G'),
        ('indel',1,'AC','A'),('multialt',1,'A','C,G'),('invalid',1,'A','X'),('same',1,'A','A')]:
        text=csv_text(['variant_id','sequence','pos','ref','alt'],[['v',base,pos,ref,alt]])
        add('variant_reject_'+kind,'/batch-variants',{'model':'human','shuffles':'0','text':text})
    twice = changed(changed(base,1),2)
    add('multi_allelic_supplied_combinations','/multi-allelic',{'model':'human',
        'fasta':f'>ref\n{base}\n>one\n{changed(base)}\n>two\n{twice}\n'},expected='boundary',
        final=[base,changed(base),twice],ids=['ref','one','two'])
    add('prioritize_reject_multiple_changes','/prioritize',{'model':'human','fasta':f'>ref\n{base}\n>alt\n{twice}\n'})
    add('allele_specific_indel_rejected','/allele-specific',{'model':'human','fasta':f'>r\n{base}\n>a\n{base}A\n'})
    for path in ['/design','/attribution','/mutagenesis']:
        text='N'+base[1:]
        add('N_policy_'+path[1:],path,{'model':'human','text':text,'start':'1','end':'1','target':'0'},
            expected='reject' if path=='/design' else 'boundary',final=None if path=='/design' else [text])
    for n in [1,200,2113,2114,2115,5000]:
        seq=dna(n)
        add('comparison_strict_'+str(n),'/model-comparison',{'model_a':'human','model_b':'human','text':seq,'policy':'strict'},
            expected='boundary' if n==2114 else 'reject',final=[seq] if n==2114 else None)
        left=(2114-n)//2 if n<2114 else (n-2114)//2
        expected= 'N'*left+seq+'N'*(2114-n-left) if n<2114 else seq[left:left+2114]
        add('comparison_center_'+str(n),'/model-comparison',{'model_a':'human','model_b':'human','text':seq,'policy':'center'},
            expected='boundary',final=[expected],extra={'explicit_resize':True,'expected_model_windows':[[expected],[expected]],
                'expected_offsets':[left if n<2114 else -left]*2})
    add('comparison_validate_second_model_before_inference','/model-comparison',
        {'model_a':'demo','model_b':'human','text':dna(200),'policy':'strict'})
    add('demo_short_policy','/api/compare',json_body={'model':'demo','reference':'ACGT','alternate':'CCGT'},
        expected='boundary',final=['N'*98+'ACGT'+'N'*98,'N'*98+'CCGT'+'N'*98],ids=['reference','alternate'],
        extra={'require_declared_demo_padding':True})
    for n in [0,1,200,2113,2114,2115,5000]:
        for path in ['/training','/fine-tuning']:
            seqs=[(dna(n-4)+f'{i:04b}'.translate(str.maketrans('01','AC'))) if n>=4 else dna(n) for i in range(8)]
            text=csv_text(['sequence','label'],[[s,i] for i,s in enumerate(seqs)])
            good=(path=='/fine-tuning' and n==2114) or (path=='/training' and n==200)
            add('labeled_csv_'+path[1:]+'_'+str(n),path,{'model':'human','text':text,'length':'200','epochs':'1','learning_rate':'0.001','target':'0'},
                expected='boundary' if good else 'reject',final=seqs if good else None)
    add('motifs_batch','/motifs',{'text':'>m1\nacgtn\n>m2\nTTTT\n','patterns':'m:ACG','k':'3'},
        expected='boundary',final=['ACGTN','TTTT'],ids=['m1','m2'])
    add('pwm_batch','/pwm-scan',{'text':'>p1\nacgtn\n','matrices':'>tiny\nA [ 1 0 0 ]\nC [ 0 1 0 ]\nG [ 0 0 1 ]\nT [ 0 0 0 ]',
        'threshold':'0.8','background':'0.25,0.25,0.25,0.25','smoothing':'0.1'},expected='boundary',final=['ACGTN'],ids=['p1'])
    add('dataset_duplicate_ids_preserved','/research/dataset',{'kind':'fasta','text':'>same\nACGT\n>same\nNNNN\n'},
        expected='response',final=['ACGT','NNNN'],ids=['same','same'])
    add('dataset_bad_sequence','/research/dataset',{'kind':'fasta','text':'>x\nACGX\n'})
    add('dataset_unicode_dna','/research/dataset',{'kind':'fasta','text':'>x\n\u00a0ACGT\n'})
    add('dataset_csv','/research/dataset',{'kind':'csv','text':'sequence,label\nacgtn,1\nCCCC,0\n'},
        expected='response',final=['ACGTN','CCCC'],ids=['record_1','record_2'])
    for name, params, outcome, final, ids in [
        ('empty_records',{'sequences':[{'id':'empty','sequence':''},{'id':'full','sequence':'ACGT'}]},'reject',None,None),
        ('nonstring',{'sequences':[3.5]},'reject',None,None),
        ('not_a_list',{'sequences':'ACGT'},'reject',None,None),
        ('missing_sequence_field',{'sequences':[{'id':'missing'}]},'reject',None,None),
        ('competing_sources',{'sequences':['ACGT'],'fasta':'unused.fasta'},'reject',None,None),
        ('excess_records',{'sequences':['ACGT','TTTT'],'max_sequences':1},'reject',None,None),
        ('invalid_base',{'sequences':['ACGX']},'reject',None,None),
        ('duplicate_ids',{'sequences':[{'id':'same','sequence':'ACGT'},{'id':'same','sequence':'TTTT'}]},'reject',None,None),
        ('valid_dict_batch',{'sequences':[{'id':'Z','sequence':'acgtn'},{'id':'A','sequence':'TTTT'}]},'helper',['ACGTN','TTTT'],['Z','A'])]:
        add('engine_'+name,'HELPER:read_sequences',params,expected=outcome,final=final,ids=ids,scope='engine helper; not separate website upload')
    for name,text,outcome,final,ids in [
        ('preamble','ACGT\n>real\nTTTT\n','reject',None,None),
        ('duplicate','>id\nACGT\n>id\nTTTT\n','reject',None,None),
        ('valid','>id\nacgt\n','helper',['ACGT'],['id'])]:
        add('engine_fasta_'+name,'HELPER:read_fasta',{'text':text},expected=outcome,final=final,ids=ids,scope='engine helper')
    for name,text,outcome,final,ids in [
        ('empty_cell','id,sequence\n001,\n','reject',None,None),
        ('leading_zero_id','id,sequence\n001,ACGT\n','helper',['ACGT'],['001'])]:
        add('engine_csv_'+name,'HELPER:sequence_csv',{'text':text},expected=outcome,final=final,ids=ids,scope='engine helper')
    add('qc_mixed_lengths','/qc',{'model':'human','fasta':f'>full\n{base}\n>short\nACGT\n'},
        expected='response',final=[base,'ACGT'],ids=['full','short'])
    add('variant_N_unchanged_context','/api/compare',json_body={'model':'human','reference':'N'+base[1:],
        'alternate':('N'+base[1:])[:-1]+changed(base,2114)[-1]},expected='boundary',
        final=['N'+base[1:],('N'+base[1:])[:-1]+changed(base,2114)[-1]],ids=['reference','alternate'])
    add('variant_N_changed_rejected','/api/compare',json_body={'model':'human','reference':'N'+base[1:],'alternate':'A'+base[1:]})
    add('fasta_file_priority','/predict',{'model':'human','fasta':'>ignored_text\n'+changed(base)},
        upload={'field':'file','filename':'chosen.fasta','text':'>chosen_upload\n'+base},
        expected='boundary',final=[base],ids=['chosen_upload'],notes='Existing explicit upload precedence: file supplies DNA; frontend preview normally clears the file field.')
    loci=json.loads((ROOT/'curated_loci.json').read_text())
    for key,locus in loci.items():
        add('curated_'+key,'/saturation',{'locus':key},expected='boundary',final=[locus['dna']],
            extra={'assembly':'hg38','reference_source':locus['source'],'coordinates':{'chrom':locus['chrom'],'start':locus['start'],'end':locus['end']}})
    add('curated_reject_DNA_override','/saturation',{'locus':next(iter(loci)),'sequence':'ACGT'})
    add('bed_content_is_not_fasta','/predict',{'model':'human'},upload={'field':'file','filename':'interval.bed','text':'chr1\t0\t2114\n'})
    for n in [0,1,200,2113,2114,2115,5000]:
        seq=dna(n)
        for path in ['/motifs','/pwm-scan','/research/dataset']:
            fields={'text':seq,'patterns':'m:ACG','k':'3','kind':'fasta','matrices':'>tiny\nA [ 1 0 0 ]\nC [ 0 1 0 ]\nG [ 0 0 1 ]\nT [ 0 0 0 ]',
                    'threshold':'0.8','background':'0.25,0.25,0.25,0.25','smoothing':'0.1'}
            add('length_independent_'+path.replace('/','_')+'_'+str(n),path,fields,
                expected='reject' if n==0 else 'response' if path=='/research/dataset' else 'boundary',
                final=[seq] if n else None,ids=['sequence_1'])
    for policy in ['strict','center']:
        add('comparison_empty_'+policy,'/model-comparison',{'model_a':'human','model_b':'human','text':'','policy':policy})
    add('training_cannot_configure_GM12878_window','/training',{'model':'human','length':'2114','text':'sequence,label\nACGT,1\n'})
    # Exercise each independently implemented upload handler, not just the
    # shared parsing helper or the prediction page's file branch.
    for path in ['/qc','/attribution','/mutagenesis','/design','/motif-simulation','/allele-specific',
                 '/prioritize','/multi-allelic','/model-comparison','/motifs','/pwm-scan','/research/dataset','/training','/fine-tuning']:
        original=next(c for c in tests if c['path']==path and c['expected'] in ['boundary','response'] and c['upload'] is None)
        sample=copy.deepcopy(original);sample['id']='upload_path_'+path.replace('/','_')
        field='fasta' if 'fasta' in sample['fields'] else 'text'
        content=sample['fields'].pop(field)
        upload_field='sequence-file' if path in ['/motif-simulation','/model-comparison','/pwm-scan'] else 'file'
        sample['upload']={'field':upload_field,'filename':'records.csv' if path in ['/training','/fine-tuning'] else 'records.fasta','text':content}
        tests.append(sample)
        invalid=copy.deepcopy(sample);invalid['id']='upload_invalid_utf8_'+path.replace('/','_')
        invalid.update(expected='reject',final_sequences=None,expected_ids=None)
        invalid['upload']={'field':upload_field,'filename':'invalid.bin','bytes_base64':'//4='}
        for key in ['expected_model_windows','expected_offsets','motif_input_batch','variant_ids','require_declared_demo_padding']:
            invalid.pop(key,None)
        tests.append(invalid)
    for path in ['/design','/attribution','/mutagenesis','/motif-simulation']:
        add('single_record_only_'+path[1:],path,{'model':'human','text':f'>one\n{base}\n>two\n{changed(base)}\n','motif':'TATAAA','positions':'1','start':'1','end':'1'})
    return tests


class BoundaryReached(BaseException):
    pass


class FormParser(HTMLParser):
    def __init__(self):
        super().__init__();self.names=set()
    def handle_starttag(self,tag,attrs):
        if tag in ['input','textarea','select']:
            values=dict(attrs)
            if values.get('name'):self.names.add(values['name'])


def main():
    global ACTIVE_RUN
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=BASE/'runs'/stamp;out.mkdir(parents=True)
    ACTIVE_RUN=out
    before=protected();save(out/'protected-before.json',before)
    tests=cases();save(BASE/'fixtures/cases.json',tests)
    from huggingface_hub import hf_hub_download
    checkpoint=Path(hf_hub_download('Genentech/GM12878_dnase-model','model.ckpt',
        revision='9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e',local_files_only=True))
    checkpoint_before=digest(checkpoint)
    expected_identity=json.loads((BASE.parent/'fixtures/model_identity.json').read_text())
    if checkpoint_before!=expected_identity['checkpoint_sha256']:
        raise ValueError('Gate 1 checkpoint identity anchor differs; Gate 2 cannot proceed on an unverified model.')
    os.environ.update(GRELU_HUMAN_CHECKPOINT=str(checkpoint),GRELU_HUMAN_INPUT_LENGTH='2114',GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from studio_web import app,parse_sequence
    from core.model_lengths import required_length,contract_for_spec
    import core.models,core.io,core.sequences,core.dna_inputs,core.dataset,engine,workbench,advanced_views,lab_views,exploration_views
    import workflows.predict,workflows.vep,workflows.design,workflows.scan,workflows.attribution,workflows.explore
    app.config.update(TESTING=True,LAB_ROOT=str(out/'state/lab'),ACCOUNT_DATABASE=str(out/'state/accounts.db'),HISTORY_DATABASE=str(out/'state/history.db'))
    (out/'state').mkdir()
    client=app.test_client();client.get('/experiments')
    with client.session_transaction() as session:csrf=session['csrf_token']
    current={}
    class ProbeAdapter:
        def __init__(self,spec):
            self.seq_len=required_length(spec);self.task_names=['ENCFF093VXI'];self.meta=dict(spec)
            self.backend='boundary-probe';self.device='cpu';self.notes=[];self.length_contract=contract_for_spec(spec)
            self.module=SimpleNamespace(data_params={},performance=None)
        def predict(self,sequences):
            frame=inspect.currentframe().f_back
            details={key:frame.f_locals[key] for key in ['records','details','transforms','effective','effective_inputs','prepared'] if key in frame.f_locals}
            current['boundary']={'stage':'pre-encoding string batch','function':frame.f_code.co_name,
                'sequences':list(sequences),'caller_input_details':details,'predictions_generated':False}
            raise BoundaryReached()
    def loader(spec,*args,**kwargs):return ProbeAdapter(spec),[]
    real_read=core.io.read_sequences
    def read(params):
        result=real_read(params);current['engine_records']=[{'id':i,'sequence':s} for i,s in result[0]]
        current['engine_warnings']=result[1];return result
    real_parse=parse_sequence
    def parse(raw,label):
        sequence=real_parse(raw,label)
        current.setdefault('normalizer_calls',[]).append({'label':label,'original':raw,'normalized':sequence})
        return sequence
    real_normalize=core.dna_inputs.normalize_dna
    def normalize(raw,*args,**kwargs):
        sequence=real_normalize(raw,*args,**kwargs)
        current.setdefault('normalization_operations',[]).append({'original':raw,'final':sequence,
            'record_label':args[0] if args else kwargs.get('label','input'),
            'original_character_length':len(raw),'final_base_length':len(sequence),
            'original_sha256_utf8':sha_bytes(raw.encode('utf-8')),'final_sha256_utf8':sha_bytes(sequence.encode('utf-8'))})
        return sequence
    real_batch=workbench.parse_batch
    def batch(*args,**kwargs):
        records=real_batch(*args,**kwargs);current['parsed_records']=[dict(r) for r in records];return records
    real_counter=core.dataset.Counter
    def counter(*args,**kwargs):
        frame=inspect.currentframe().f_back
        if 'records' in frame.f_locals:
            current['dataset_normalized_records']=[dict(row) for row in frame.f_locals['records']]
        return real_counter(*args,**kwargs)
    real_quality=workbench.quality_report
    def quality(records,length):
        current['parsed_records']=[dict(r) for r in records];return real_quality(records,length)
    real_execute=engine.execute_platform_workflow
    def execute(workflow,params):
        current['workflow']=workflow;current['engine_params']={k:v for k,v in params.items() if k not in ['output_dir']}
        if params.get('variants_csv'):current['serialized_variant_csv']=Path(params['variants_csv']).read_text()
        return real_execute(workflow,params)
    def record_stop(kind,records):
        current['boundary']={'stage':kind,'sequences':[r['sequence'] if isinstance(r,dict) else r[0] for r in records],
                             'records':records,'predictions_generated':False}
        raise BoundaryReached()
    def submit(kind,task,*args,**kwargs):return record_stop('before training/job submission',inspect.getclosurevars(task).nonlocals['rows'])
    def motifs(records,*args,**kwargs):return record_stop('before motif analysis',records)
    def pwm(records,*args,**kwargs):return record_stop('before PWM scanning',records)
    real_fit=workflows.explore.fit_with_offset
    def fit(sequence,length):
        value=real_fit(sequence,length);current.setdefault('resize_operations',[]).append({
            'original':sequence,'model_length':length,'final':value[0],'offset':value[1]});return value
    results=[]
    with ExitStack() as stack:
        for module in [core.models,workflows.predict,workflows.vep,workflows.design,workflows.scan,workflows.attribution,workflows.explore]:
            stack.enter_context(patch.object(module,'load_model',loader))
        stack.enter_context(patch('studio_web.parse_sequence',parse))
        stack.enter_context(patch.object(core.dna_inputs,'normalize_dna',normalize))
        stack.enter_context(patch.object(core.io,'normalize_dna',normalize))
        stack.enter_context(patch.object(workbench,'parse_batch',batch))
        stack.enter_context(patch.object(core.dataset,'Counter',counter))
        stack.enter_context(patch.object(core.io,'read_sequences',read))
        stack.enter_context(patch.object(workflows.predict,'read_sequences',read))
        stack.enter_context(patch.object(engine,'execute_platform_workflow',execute))
        stack.enter_context(patch.object(workbench,'quality_report',quality))
        stack.enter_context(patch.object(lab_views,'submit_job',submit))
        stack.enter_context(patch.object(lab_views,'motif_analysis',motifs))
        stack.enter_context(patch.object(exploration_views,'scan_pwm',pwm))
        stack.enter_context(patch.object(workflows.explore,'fit_with_offset',fit))
        stack.enter_context(patch.object(core.sequences,'one_hot',side_effect=AssertionError('Gate 3 encoding must not run')))
        stack.enter_context(patch.object(core.models,'one_hot',side_effect=AssertionError('Gate 3 encoding must not run')))
        for method in ['predict','_predict_torch','_predict_grelu','grad_attribution']:
            stack.enter_context(patch.object(core.models.ModelAdapter,method,side_effect=AssertionError('Real inference/encoding must not run in Gate 2')))
        for case in tests:
            current.clear();folder=out/'cases'/case['id'];folder.mkdir(parents=True)
            save(folder/'submitted.json',case)
            response=None;helper=None;boundary=False;failure=None;rejection=None
            try:
                path=case['path'];fields=dict(case['fields'])
                if path.startswith('HELPER:'):
                    if path=='HELPER:read_fasta':
                        file=folder/'input.fasta';file.write_text(fields['text']);helper=core.io.read_fasta(file)
                    elif path=='HELPER:sequence_csv':
                        file=folder/'input.csv';file.write_text(fields['text']);helper=real_read({'sequences_csv':str(file)})[0]
                    else:helper=real_read(fields)[0]
                    current['boundary']={'stage':'engine parsing helper return','sequences':[s for i,s in helper],
                                         'records':[{'id':i,'sequence':s} for i,s in helper]}
                else:
                    fields['csrf_token']=csrf
                    if case.get('upload'):
                        upload=case['upload'];data=base64.b64decode(upload['bytes_base64']) if 'bytes_base64' in upload else upload['text'].encode('utf-8')
                        (folder/'submitted-upload.bin').write_bytes(data)
                        current['upload_original_sha256_bytes']=sha_bytes(data)
                        current['upload_original_length_bytes']=len(data)
                        fields[upload['field']]=(io.BytesIO(data),upload['filename'])
                    if case['json_body'] is not None:
                        response=client.post(path,json=case['json_body'])
                    else:response=client.post(path,data=fields,content_type='multipart/form-data' if case.get('upload') else None)
                    (folder/'response.html').write_bytes(response.data)
                    current['response_status']=response.status_code
                    if path=='/research/dataset' and response.status_code==200:
                        # Trace actual normalized inputs at Counter construction below in later inspection;
                        # result records retain order/length. No inference is involved in this descriptive path.
                        from tests.run_gm12878_sample import record
                        current['dataset_result']=record(response)
            except BoundaryReached:boundary=True
            except ValueError as exc:rejection=str(exc)
            except Exception:failure=traceback.format_exc()
            observed=current.get('boundary',{}).get('sequences')
            if case['expected']=='response':
                if case['path']=='/qc':observed=[r['sequence'] for r in current.get('parsed_records',[])]
                if case['path']=='/research/dataset':
                    records=current.get('dataset_result',{}).get('records',[])
                    current['descriptive_input_verification']={'record_ids':[r['id'] for r in records],'lengths':[r['length'] for r in records]}
                    observed=[r['sequence'] for r in current.get('dataset_normalized_records',[])]
            ok=(boundary if case['expected']=='boundary' else
                helper is not None if case['expected']=='helper' else
                response is not None and response.status_code==200 if case['expected']=='response' else
                (rejection is not None or response is not None and response.status_code in [400,413,422]) and not boundary)
            if case['final_sequences'] is not None and observed is not None:
                ok=ok and (observed[:1]==case['final_sequences'] if case.get('motif_input_batch') else observed==case['final_sequences'])
            elif case['final_sequences'] is not None and case['path']!='/research/dataset':ok=False
            actual_ids=None
            if current.get('engine_records'):actual_ids=[r['id'] for r in current['engine_records']]
            elif helper is not None:actual_ids=[i for i,s in helper]
            elif current.get('parsed_records'):actual_ids=[r['id'] for r in current['parsed_records']]
            elif current.get('dataset_normalized_records'):actual_ids=[r['id'] for r in current['dataset_normalized_records']]
            if case.get('motif_input_batch') and observed:
                seq=case['final_sequences'][0]
                ok=ok and len(observed)==3 and observed[0]==seq and observed[1]=='TATAAA'+seq[6:] and observed[2][6:]==seq[6:] and sorted(observed[2][:6])==sorted('TATAAA')
            if case.get('expected_ids') and actual_ids is not None:ok=ok and actual_ids==case['expected_ids']
            if case.get('variant_ids') and boundary:
                records=current['boundary'].get('caller_input_details',{}).get('records',[])
                ok=ok and [r['variant_id'] for r in records]==case['variant_ids']
            if case.get('require_declared_demo_padding'):
                ok=ok and current.get('engine_params',{}).get('model',{}).get('input_policy')=='allow_short_with_n_pad'
            if case['path']=='/research/dataset' and case['expected']=='response':
                details=current.get('dataset_result',{}).get('records',[])
                ok=ok and [row.get('record_index') for row in details]==list(range(len(details)))
            if case.get('expected_model_windows'):
                operations=current.get('resize_operations',[])
                ok=ok and [[op['final']] for op in operations]==case['expected_model_windows']
                ok=ok and [op['offset'] for op in operations]==case['expected_offsets']
            if case.get('expected_http'):ok=ok and response is not None and response.status_code==case['expected_http']
            status='FAIL' if failure or not ok else 'PASS'
            explanation=f"Expected {case['expected']}; observed "+('boundary stop' if boundary else 'helper return' if helper is not None else 'validation rejection: '+rejection if rejection else f'HTTP {response.status_code}' if response is not None else 'no response')
            if failure:explanation=failure
            finals=[]
            if case.get('variant_ids') and boundary:
                ids=[r['variant_id'] for r in current['boundary'].get('caller_input_details',{}).get('records',[]) if r.get('status')=='ok']
                actual_ids=[identifier+':reference' for identifier in ids]+[identifier+':alternate' for identifier in ids]
            for i,sequence in enumerate(observed or []):
                finals.append({'id':actual_ids[i] if actual_ids and i<len(actual_ids) else f'input_{i+1}', 'record_index':i,
                    'sequence':sequence,'length':len(sequence),'sha256_utf8':sha_bytes(sequence.encode('utf-8'))})
            evidence={'case_id':case['id'],'status':status,'expected':case['expected'],'explanation':explanation,
                'original_submission_sha256_canonical_json':sha_bytes(canonical({key:case[key] for key in ['path','fields','json_body','upload']})),
                'original_fields':case['fields'],'original_json':case['json_body'],'upload':case['upload'],
                'original_text_character_lengths':{key:len(value) for key,value in case['fields'].items() if isinstance(value,str)},
                'normalized_input_records':current.get('parsed_records',current.get('engine_records',current.get('dataset_normalized_records',[]))),
                'final_records':finals,'final_records_sha256_canonical_json':sha_bytes(canonical(finals)),
                'input_observations':dict(current),'genome_assembly':case.get('assembly'),
                'assembly_note':'Immutable hg38 preset with explicit coordinates/reference source.' if case.get('assembly') else 'User-supplied sequence strings are not reference-genome extractions. Model training assembly does not establish input provenance.',
                'reference_source':case.get('reference_source'),'coordinates':case.get('coordinates'),
                'encoding_executed':False,'inference_executed':False,'mock_scope':'Metadata-only adapter and stop at pre-encoding .predict; no predictions returned.'}
            save(folder/'evidence.json',evidence)
            results.append({'id':case['id'],'path':case['path'],'status':status,'explanation':explanation,
                            'evidence':str((folder/'evidence.json').relative_to(out))})
        # Frontend schemas: actual rendered controls, no browser behavior claim.
        for path,fields in [('/predict',{'model','fasta','file'}),('/batch-variants',{'model','text','file'}),
                            ('/training',{'text','length','epochs'}),('/model-comparison',{'model_a','model_b','text','policy','sequence-file'})]:
            response=client.get(path);parser=FormParser();parser.feed(response.text)
            file=out/'forms'/(path[1:]+'.html');file.parent.mkdir(exist_ok=True);file.write_bytes(response.data)
            results.append({'id':'form_schema_'+path[1:],'path':path,'status':'PASS' if response.status_code==200 and fields<=parser.names else 'FAIL',
                'explanation':f'Required payload fields {sorted(fields)}; actual form controls {sorted(parser.names)}; browser execution not performed.',
                'evidence':str(file.relative_to(out))})
        for method in ['get','post']:
            response=getattr(client,method)('/research/bed')
            save(out/f'bed-{method}.json',{'status_code':response.status_code,'body':response.text})
            results.append({'id':'bed_route_absent_'+method,'path':'/research/bed','status':'PASS' if response.status_code==404 else 'FAIL',
                            'explanation':'Retired BED route must not be misrepresented as a working extraction feature.','evidence':f'bed-{method}.json'})
    for name in ['BED3_upload','multiple_BED_intervals','zero_based_half_open_extraction','off_by_one_extraction','invalid_chromosome',
                 'negative_start','end_not_after_start','incorrect_interval_length','missing_reference','hg19_hg38_mismatch','strand_handling']:
        results.append({'id':name,'path':'/research/bed','status':'SKIP','explanation':'Website BED upload/extraction is not implemented (GET/POST 404); no website sequence exists to compare. The CLI conversion helper is not a website feature.','evidence':'bed-post.json'})
    reference=ROOT/'datasets/gm12878/biological/source/hg38.2bit'
    reference_evidence={'assembly':'hg38','coordinate_convention':'zero-based start, exclusive end; forward reference strand',
        'source_url':'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit','local_file':str(reference),
        'comparison':'Stored curated loci originally retrieved via UCSC sequence API versus independent Bio.SeqIO TwoBit extraction; no network call or model inference.'}
    extracted_bases=0
    try:
        from Bio import SeqIO
        expected_hash=json.loads((ROOT/'datasets/gm12878/biological/sources.json').read_text())['sources']['hg38.2bit']['sha256']
        actual_hash=digest(reference)
        reference_evidence.update(sha256=actual_hash,expected_sha256=expected_hash)
        assert actual_hash==expected_hash,'Cached reference differs from the recorded reference identity'
        reader=SeqIO.parse(str(reference),'twobit');comparisons=[]
        for key,locus in json.loads((ROOT/'curated_loci.json').read_text()).items():
            sequence=str(reader.sequences[locus['chrom']][locus['start']:locus['end']]).upper()
            extracted_bases+=len(sequence)
            comparisons.append({'locus':key,'chrom':locus['chrom'],'start':locus['start'],'end':locus['end'],
                'length':len(sequence),'independently_extracted_sequence':sequence,'stored_sequence':locus['dna'],
                'extracted_sha256_utf8':sha_bytes(sequence.encode()),'stored_sha256_utf8':sha_bytes(locus['dna'].encode()),
                'equal':sequence==locus['dna']})
            results.append({'id':'independent_reference_'+key,'path':'/saturation','status':'PASS' if sequence==locus['dna'] else 'FAIL',
                'explanation':'Real cached hg38 reference extraction must exactly match the curated preset context and BED-style interval length. This is not website BED-upload coverage.',
                'evidence':'independent-reference.json'})
        reader.stream.close();reference_evidence['comparisons']=comparisons
    except (FileNotFoundError,ImportError) as exc:
        reference_evidence['skip_reason']=str(exc)
        results.append({'id':'independent_reference','path':'/saturation','status':'SKIP','explanation':str(exc),'evidence':'independent-reference.json'})
    except Exception:
        reference_evidence['failure']=traceback.format_exc()
        results.append({'id':'independent_reference','path':'/saturation','status':'FAIL','explanation':reference_evidence['failure'],'evidence':'independent-reference.json'})
    save(out/'independent-reference.json',reference_evidence)
    results.append({'id':'browser_javascript_validation','path':'frontend','status':'SKIP','explanation':'Rendered form schemas inspected; no actual browser/JavaScript execution.','evidence':'forms/predict.html'})
    after=protected();save(out/'protected-after.json',after)
    results.append({'id':'protected_results_unchanged','path':'preservation','status':'PASS' if before==after else 'FAIL',
                    'explanation':'Phase 1 and all existing Gate 1 evidence must remain unchanged.','evidence':'protected-after.json'})
    results.append({'id':'checkpoint_unchanged','path':'preservation','status':'PASS' if digest(checkpoint)==checkpoint_before else 'FAIL',
                    'explanation':'Pinned pretrained checkpoint content must remain unchanged.','evidence':'checkpoint.json'})
    save(out/'checkpoint.json',{'path':str(checkpoint),'sha256_before':checkpoint_before,'sha256_after':digest(checkpoint)})
    counts={s:sum(r['status']==s for r in results) for s in ['PASS','FAIL','SKIP','BLOCKED']}
    paths=sorted({r['path'] for r in results})
    coverage=[]
    for path in paths:
        rows=[r for r in results if r['path']==path]
        status='FAILED' if any(r['status']=='FAIL' for r in rows) else 'NOT TESTED' if all(r['status']=='SKIP' for r in rows) else 'PARTIAL' if any(r['status']=='SKIP' for r in rows) else 'VERIFIED'
        if path=='/research/bed':status='NOT TESTED'
        if path in ['/saturation','/model-comparison'] and status=='VERIFIED':status='PARTIAL'
        coverage.append({'path':path,'classification':status,'checks':len(rows),'scope':'Gate 2 input handling only; not numerical model correctness.'})
    source_files=['studio_web.py','workbench.py','core/dna_inputs.py','core/io.py','core/sequences.py','core/dataset.py','advanced_views.py',
                  'variant_views.py','lab_compute.py','lab_views.py','exploration_views.py','workflows/predict.py','workflows/vep.py','workflows/explore.py',
                  'static/model-inputs.js','templates/model_select.html','curated_scan.py']
    sources={p:digest(ROOT/p) for p in source_files}
    save(out/'source-hashes.json',sources)
    save(out/'coverage.json',coverage)
    result={'gate':2,'name':'DNA Input Integrity','status':'FAIL' if counts['FAIL'] else 'PASS','counts':counts,'checks':results,
        'canonical_hash_format':'Original submission: UTF-8 JSON over path/fields/json_body/upload, ensure_ascii=False, sort_keys=True, compact separators. Final sequences: exact UTF-8 bases. Final record list: ordered id/sequence/length/hash objects, same JSON conventions. CSRF tokens are test-session mechanics, excluded from the DNA payload hash; upload bytes are saved separately.',
        'inference_executed':False,'encoding_executed':False,'training_executed':False,'gate3_started':False,
        'fixtures_sha256':digest(BASE/'fixtures/cases.json'),'source_hashes':sources,'coverage':coverage}
    save(out/'results.json',result)
    import resource
    from importlib.metadata import version
    save(out/'environment.json',{'python':sys.version,'versions':{name:version(name) for name in ['torch','numpy','grelu','flask','pandas','biopython']},
        'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'model_inference_calls':0,'tensor_encoding_calls':0,'training_jobs_executed':0,
        'sequential_cases':True,'reference_downloads':0,'reference_extraction_bases':extracted_bases})
    (out/'runner-source.py.txt').write_text(Path(__file__).read_text())
    lines=['# Gate 2 — DNA input integrity','',f"**{result['status']}** — {counts}",'',
        'Real Flask parsing/validation and preprocessing were executed. Metadata-only adapters and explicit boundary exceptions stop before encoding/model inference; no mock prediction values are returned. This is not Gate 3/4 or a biological benchmark.','',
        '## Coverage','', '| Input path | Classification | Checks |','|---|---|---|']
    lines += [f"| {c['path']} | {c['classification']} | {c['checks']} |" for c in coverage]
    lines += ['', '## Checks','', '| Check | Status | Explanation | Evidence |','|---|---|---|---|']
    for row in results:
        explanation=row['explanation'].replace('|','/').replace('\n',' ')
        lines.append(f"| {row['id']} | {row['status']} | {explanation} | [evidence]({row['evidence']}) |")
    lines += ['', '## Limits','', 'BED is retired, not verified. Browser JavaScript is not executed. Curated preset, training inputs and descriptive input preservation need the explicit coverage recorded above; no unlisted path is automatically certified. No tensors or numeric predictions are tested.', '',
        'Original payloads, expected normalized strings, actual parser calls and exact final strings/hashes are saved per case. Original submission length can include headers/whitespace; final DNA length counts bases. Uploaded bytes are saved exactly. Sequence-only inputs have unknown assembly provenance; hg38 model metadata does not identify the input assembly.', '',
        'Reproduce from repository root:', '', '```bash', '.venv/bin/python Testing/testing-phase-2/input-output/gate-2/verify_gate2.py', '```','',
        'Prior runs are preserved. Exit code 1 indicates failed checks. See results.json, coverage.json and per-case evidence rather than assuming a route exists or a mock represents inference correctness.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(BASE/'latest.json',{'run':str(out.relative_to(BASE)),'status':result['status'],'counts':counts})
    print(json.dumps({'report':str(out/'REPORT.md'),'counts':counts,'failed':[r['id'] for r in results if r['status']=='FAIL']},indent=2),flush=True)
    return 1 if counts['FAIL'] else 0


if __name__=='__main__':
    try:
        sys.exit(main())
    except Exception:
        error=traceback.format_exc()
        if globals().get('ACTIVE_RUN'):
            save(ACTIVE_RUN/'controller-error.json',{'status':'BLOCKED','error':error,
                 'explanation':'Harness could not complete. No passing gate is claimed; partial evidence is retained.'})
        print(error,file=sys.stderr)
        sys.exit(2)
