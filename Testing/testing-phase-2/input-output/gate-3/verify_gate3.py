"""Gate 3 only: actual input tensors, independently calculated expectations.

The real pretrained architecture has a forward-pre-hook which saves input and
raises a BaseException sentinel. No convolution or model output is executed.
CPU worker is supervised for RAM/time; all prior evidence is retained.
"""
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
sys.path.insert(0,str(ROOT))
# Literal oracle, independent of Studio LUTs and all gReLU conversion helpers.
VECTORS={'A':(1.,0.,0.,0.),'C':(0.,1.,0.,0.),'G':(0.,0.,1.,0.),'T':(0.,0.,0.,1.),'N':(0.,0.,0.,0.)}
INDEX={'A':0,'C':1,'G':2,'T':3,'N':4}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def protected():
    names=['Testing/testing-phase-1','test-results/gm12878-biological','test-results/gm12878-verified',
           'Testing/testing-phase-2/input-output/runs','Testing/testing-phase-2/input-output/gate-2/runs']
    return {str(p.relative_to(ROOT)):sha(p) for name in names for p in sorted((ROOT/name).rglob('*')) if p.is_file()}


def oracle(sequences):
    import numpy as np
    if len({len(s) for s in sequences})!=1:raise ValueError('Oracle requires already prepared equal-length DNA.')
    expected=np.zeros((len(sequences),4,len(sequences[0])),dtype=np.float32)
    for row,sequence in enumerate(sequences):
        for position,base in enumerate(sequence):
            for channel,value in enumerate(VECTORS[base]):expected[row,channel,position]=value
    return expected


def expected_resize(sequence,length):
    if len(sequence)>length:
        left=(len(sequence)-length)//2
        return sequence[left:left+length]
    missing=length-len(sequence);left=missing//2
    return 'N'*left+sequence+'N'*(missing-left)


def expected_rc(sequence):
    letters={'A':'T','C':'G','G':'C','T':'A','N':'N'}
    return ''.join(letters[base] for base in reversed(sequence))


class Captured(BaseException):
    pass


def worker(out):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from huggingface_hub import hf_hub_download
    from core.models import load_model,ModelAdapter,_build_demo_model
    from core.sequences import one_hot,revcomp,fit_with_offset
    from core.dna_inputs import normalize_dna
    import grelu.sequence.format as fmt
    from grelu.data.augment import Augmenter
    identity=json.loads((BASE.parent/'fixtures/model_identity.json').read_text())
    path=Path(hf_hub_download(identity['repo_id'],identity['filename'],revision=identity['revision'],local_files_only=True))
    initial_sha=sha(path)
    if initial_sha!=identity['checkpoint_sha256']:raise ValueError('Pinned checkpoint identity mismatch')
    spec={'source':'checkpoint','path':str(path),'seq_len':2114}
    adapter,warnings=load_model(spec,'cpu',batch_size=2,backend='grelu')
    module=adapter.module
    demo_module=_build_demo_model(3,0)
    demo=ModelAdapter(demo_module,'cpu',200,['demo_0','demo_1','demo_2'],
                      {'source':'demo','seq_len':200,'input_policy':'allow_short_with_n_pad'},batch_size=1,backend='builtin')
    def state_hash(model):
        h=hashlib.sha256()
        for name,tensor in sorted(model.state_dict().items()):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    weights_before=state_hash(module)
    gate2=BASE.parent/'gate-2'
    latest2=json.loads((gate2/'latest.json').read_text());source2=gate2/latest2['run']
    def accepted(case):
        record=json.loads((source2/'cases'/case/'evidence.json').read_text())
        if record['status']!='PASS':raise ValueError('Source Gate 2 fixture was not accepted: '+case)
        return record['final_records']
    first=accepted('dna_uppercase')[0]['sequence']
    second=accepted('fasta_multi_text')[1]['sequence']
    unknown=accepted('dna_N_context')[0]['sequence']
    alln=accepted('dna_all_N')[0]['sequence']
    other=accepted('comparison_center_5000')[0]['sequence']
    records=[{'id':name,'sequence':sequence} for name,sequence in zip(['first','variant','unknown','all_N','cropped'],[first,second,unknown,alln,other])]
    save(out/'fixtures.json',{'source_gate2_run':str(source2),'source_results_sha256':sha(source2/'results.json'),
         'records':records,'oracle_vectors':VECTORS,'integer_mapping':INDEX,
         'fixture_note':'Accepted Gate 2 DNA; no prediction values or biology labels.'})
    (out/'fixtures.fasta').write_text(''.join(f">{r['id']}\n{r['sequence']}\n" for r in records))
    results=[];active={};weighted_attempts=[];augmentation_calls=[]
    def check(name,condition,explanation,evidence=None,kind='tensor',status=None,path='adapter'):
        row={'id':name,'status':status or ('PASS' if condition else 'FAIL'),'kind':kind,
             'explanation':explanation,'evidence':evidence or [],'workflow':path}
        results.append(row);print(row['status'],name,flush=True)
    def hook(layer,args):
        x=args[0]
        if not isinstance(x,torch.Tensor):raise TypeError('Actual model input is not a tensor')
        active['tensor']=x.detach().cpu().numpy().copy()
        active['metadata']={'shape':list(x.shape),'torch_dtype':str(x.dtype),'device':str(x.device),
            'requires_grad':x.requires_grad,'training':layer.training,'model_class':f'{type(layer).__module__}.{type(layer).__name__}',
            'boundary':'architecture forward_pre_hook, after Lightning.format_input and device transfer, before architecture forward body'}
        raise Captured()
    def weighted_guard(layer,args):
        weighted_attempts.append(type(layer).__name__)
        raise AssertionError('Weighted model execution is forbidden in Gate 3')
    handles=[module.model.register_forward_pre_hook(hook),demo_module.register_forward_pre_hook(hook)]
    for model in [module,demo_module]:
        for child in model.modules():
            if isinstance(child,(torch.nn.Conv1d,torch.nn.Linear)):
                handles.append(child.register_forward_pre_hook(weighted_guard))
    def compare(name,actual,expected,ids,metadata=None):
        folder=out/'captures'/name;folder.mkdir(parents=True,exist_ok=True)
        np.save(folder/'actual.npy',actual,allow_pickle=False);np.save(folder/'expected.npy',expected,allow_pickle=False)
        mismatches=[]
        shape_equal=actual.shape==expected.shape
        if shape_equal:
            for row,channel,position in np.argwhere(actual!=expected):
                mismatches.append({'sequence_id':ids[int(row)],'position_1based':int(position)+1,
                    'channel':int(channel),'channel_base':'ACGT'[int(channel)],
                    'expected':float(expected[row,channel,position]),'actual':float(actual[row,channel,position])})
        info={'sequence_ids':ids,'actual_shape':list(actual.shape),'expected_shape':list(expected.shape),
              'actual_dtype':str(actual.dtype),'expected_dtype':str(expected.dtype),'finite':bool(np.isfinite(actual).all()),
              'values_in_zero_one':bool(np.isin(actual,[0,1]).all()),'mismatches':mismatches,
              'metadata':metadata,'actual_sha256_file':sha(folder/'actual.npy'),'expected_sha256_file':sha(folder/'expected.npy')}
        save(folder/'comparison.json',info)
        valid=shape_equal and actual.dtype==np.float32 and np.isfinite(actual).all() and not mismatches
        if metadata:valid=valid and metadata['device']=='cpu' and metadata['torch_dtype']=='torch.float32' and not metadata['training']
        return bool(valid),str((folder/'comparison.json').relative_to(out))
    def capture(name,invoke,sequences,ids,path='adapter',requires_grad=None):
        active.clear();before=torch.random.get_rng_state().clone()
        try:
            invoke()
            check(name,False,'Forward boundary was not reached; no output may be fabricated.',kind='tensor',path=path)
            return None
        except Captured:
            pass
        except Exception:
            file=out/'captures'/name/'error.json';save(file,{'error':traceback.format_exc()})
            check(name,False,'Encoding/capture failed; saved traceback.',[str(file.relative_to(out))],path=path)
            return None
        actual=active['tensor'];metadata=active['metadata'];metadata['rng_unchanged']=bool(torch.equal(before,torch.random.get_rng_state()))
        if path=='/predict' and active.get('parsed_records') is not None:
            parsed=active['parsed_records']
            file=out/'captures'/name/'record-mapping.json'
            save(file,{'actual_parsed_records':parsed,'expected_ids':ids,'expected_sequences':sequences})
            check(name+'_id_mapping',[r[0] for r in parsed]==ids and [r[1] for r in parsed]==sequences,
                  'IDs and DNA order from actual workflow reader map exactly onto captured tensor rows.',[str(file.relative_to(out))],kind='record mapping',path=path)
        prepared=active.get('prepared_sequences')
        owner=active.get('adapter')
        remainder=[]
        if prepared is not None:
            file=out/'captures'/name/'prepared.json'
            save(file,{'actual_sequences':prepared,'independent_expected_sequences':sequences,'sequence_ids':ids,
                'actual_batch_size':owner.batch_size,'scope':'First real forward boundary stops the request. Remaining actual prepared chunks are replayed through the same real adapter without numerical outputs.'})
            check(name+'_prepared',prepared==sequences,'Full actual adapter input strings and order equal independent expected DNA before batching.',[str(file.relative_to(out))],kind='preparation',path=path)
            count=actual.shape[0]
            remainder=[(prepared[start:start+owner.batch_size],sequences[start:start+owner.batch_size],ids[start:start+owner.batch_size]) for start in range(count,len(prepared),owner.batch_size)]
            sequences=sequences[:count];ids=ids[:count]
        valid,evidence=compare(name,actual,oracle(sequences),ids,metadata)
        if requires_grad is not None:valid=valid and metadata['requires_grad']==requires_grad
        check(name,valid,'Actual forward-input tensor equals independent oracle at every channel/position; see captured arrays and explicit mismatch list.',[evidence],path=path)
        for index,(chunk,expected_chunk,chunk_ids) in enumerate(remainder):
            capture(name+'_remaining_'+str(index),lambda c=chunk:owner.predict(c),expected_chunk,chunk_ids,path=path)
        return actual
    # Installed mapping is checked, never used to construct the expected arrays.
    mapping_ok=fmt.STANDARD_BASES==['A','C','G','T'] and fmt.BASE_TO_INDEX_HASH==INDEX
    for sequence in ['A','C','G','T','N','ACGTN','GANTCTNNAG']:
        label='short_'+sequence
        expected=oracle([sequence])
        native=fmt.strings_to_one_hot([sequence]).cpu().numpy()
        builtin=one_hot([sequence],len(sequence))
        for backend,actual in [('native',native),('builtin',builtin)]:
            ok,evidence=compare(label+'_'+backend,actual,expected,['short'])
            check(label+'_'+backend,ok and mapping_ok,'Literal A/C/G/T channel vectors and N zero vector checked against actual encoded values.',[evidence],path='short encoders')
        indices=fmt.strings_to_indices([sequence])
        index_expected=np.array([[INDEX[b] for b in sequence]],dtype=np.int8)
        decoded=None;decode_error=None
        try:decoded=fmt.one_hot_to_strings(torch.from_numpy(native))
        except Exception:decode_error=traceback.format_exc()
        file=out/'integer-roundtrip'/(label+'.json')
        save(file,{'actual_indices':indices.tolist(),'dtype':str(indices.dtype),'expected_indices':index_expected.tolist(),'decoded':decoded,'expected_strings':[sequence],'decode_error':decode_error})
        check(label+'_indices',np.array_equal(indices,index_expected) and indices.dtype==np.int8,
              'Actual integer values and int8 dtype checked independently.',[str(file.relative_to(out))],path='native integer')
        check(label+'_roundtrip',decoded==[sequence],
              'Optional installed gReLU inverse conversion checked literally; errors or changed bases are failures. This inverse is not used by website prediction.',[str(file.relative_to(out))],path='native optional inverse')
    normalized=normalize_dna('aCgTn')
    ok,evidence=compare('lowercase_after_normalization',fmt.strings_to_one_hot([normalized]).numpy(),oracle(['ACGTN']),['lower'])
    check('lowercase_after_normalization',ok,'Actual Gate 2 normalization precedes native encoding; expected uppercase DNA is literal.',[evidence],path='short encoders')
    for text in ['X','A!','A1','Ａ','']:
        error=None
        try:one_hot([text],4)
        except ValueError as exc:error=str(exc)
        filename='invalid_'+str(len(results))+'.json';save(out/filename,{'input':text,'error':error})
        check('encoder_reject_'+repr(text),error is not None,'Malformed DNA must raise before numerical encoding, not turn into zero/N columns.',[filename],kind='negative guard')
    for size in [0,-1,True,1.5]:
        error=None
        try:ModelAdapter(module,'cpu',2114,['ENCFF093VXI'],spec,batch_size=size)
        except ValueError as exc:error=str(exc)
        filename='bad_batch_'+str(len(results))+'.json';save(out/filename,{'batch_size':size,'error':error})
        check('invalid_batch_size_'+repr(size),error is not None,'Reject invalid batch configuration instead of defaulting or dropping records.',[filename],kind='negative guard')
    for backend in ['grelu','builtin']:
        adapter.backend=backend;adapter._use_grelu_predict=backend=='grelu';adapter.batch_size=2
        capture('GM12878_single_'+backend,lambda:adapter.predict([first]),[first],['first'])
        adapter.batch_size=3
        capture('GM12878_three_'+backend,lambda:adapter.predict([first,second,unknown]),[first,second,unknown],['first','variant','unknown'])
        adapter.batch_size=2
        groups=list(adapter.iter_sequence_batches([r['sequence'] for r in records]))
        batches_ok=[len(g) for g in groups]==[2,2,1] and [s for g in groups for s in g]==[r['sequence'] for r in records]
        offset=0
        for i,group in enumerate(groups):
            ids=[r['id'] for r in records[offset:offset+len(group)]]
            captured=capture(f'GM12878_batches_{backend}_{i}',lambda g=group:adapter.predict(g),group,ids,path='batch preprocessing')
            batches_ok=batches_ok and captured is not None;offset+=len(group)
        file=out/f'batch-manifest-{backend}.json';save(file,{'ids':[r['id'] for r in records],'batch_sizes':[len(g) for g in groups],
            'drop_or_repeat':not batches_ok,'method':'Actual adapter iterator fully traversed; each chunk separately enters the real adapter and forward hook. Numerical output concatenation is deliberately not run.'})
        check('batch_schedule_'+backend,batches_ok,'All five IDs covered exactly once, batches 2/2/1, with exact captured tensors for each actual chunk.',[file.name],kind='batch invariant',path='batch preprocessing')
        adapter.batch_size=1
        for i,sequence in enumerate([first,second,unknown]):
            capture(f'batch_size_one_{backend}_{i}',lambda s=sequence:adapter.predict([s]),[sequence],[records[i]['id']],path='batch preprocessing')
    adapter.backend='grelu';adapter._use_grelu_predict=True;adapter.batch_size=3
    capture('duplicate_DNA_distinct_ids',lambda:adapter.predict([first,first,second]),[first,first,second],['copy_1','copy_2','variant'],path='batch preprocessing')
    adapter.batch_size=2
    original_augment=Augmenter.__call__
    def observed_augment(*args,**kwargs):
        augmentation_calls.append('called');return original_augment(*args,**kwargs)
    with patch.object(Augmenter,'__call__',observed_augment):
        a=capture('deterministic_a',lambda:adapter.predict([first,unknown]),[first,unknown],['first','unknown'],path='determinism')
        first_rng_unchanged=active['metadata']['rng_unchanged']
        b=capture('deterministic_b',lambda:adapter.predict([first,unknown]),[first,unknown],['first','unknown'],path='determinism')
    save(out/'augmentation.json',{'calls':augmentation_calls,'training_metadata':{k:module.data_params['train'].get(k) for k in ['rc','max_seq_shift','max_pair_shift']},
        'actual_inference_path':'predict_on_seqs → strings_to_one_hot, no SeqDataset/Augmenter',
        'no_inference_shift_or_random_rc':not augmentation_calls})
    check('deterministic_no_augmentation',a is not None and b is not None and np.array_equal(a,b) and not augmentation_calls and first_rng_unchanged and active['metadata']['rng_unchanged'],
        'Repeated tensor captures agree; actual inference path never calls Augmenter despite training metadata settings.',
        ['augmentation.json','captures/deterministic_a/comparison.json','captures/deterministic_b/comparison.json'],kind='tensor invariant',path='determinism')
    rc=revcomp(unknown)
    expected=expected_rc(unknown)
    capture('reverse_complement',lambda:adapter.predict([rc]),[expected],['unknown:RC'],path='reverse complement')
    capture('gradient_input_encoding',lambda:adapter.grad_attribution([unknown],[0],0),[unknown],['unknown'],path='attribution',requires_grad=True)
    # Real website handlers; only repeated model construction is replaced by
    # returning the same REAL preloaded module/adapter. No prediction method is
    # replaced, and no numerical output is returned.
    os.environ.update(GRELU_HUMAN_CHECKPOINT=str(path),GRELU_HUMAN_INPUT_LENGTH='2114',
                      GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase',GRELU_SECRET_KEY='gate3-isolated')
    import core.models,engine,workbench,advanced_views,lab_views,exploration_views
    import workflows.predict,workflows.vep,workflows.scan,workflows.attribution,workflows.design,workflows.explore
    from studio_web import app,MODELS,PROVENANCE
    state=out/'state';state.mkdir();app.config.update(TESTING=True,LAB_ROOT=str(state/'lab'),
        ACCOUNT_DATABASE=str(state/'accounts.db'),HISTORY_DATABASE=str(state/'accounts-history.db'))
    client=app.test_client();client.get('/experiments')
    with client.session_transaction() as session:csrf=session['csrf_token']
    def real_adapter(specification,device='cpu',batch_size=None,backend='auto'):
        from core.model_lengths import required_length
        if specification.get('source')=='demo':
            if required_length(specification)!=200:raise ValueError('This Gate 3 demo fixture verifies 200 only.')
            demo.batch_size=batch_size or 4;return demo,[]
        if Path(specification.get('path','')).resolve()!=path.resolve():raise ValueError('Unexpected model checkpoint')
        if required_length(specification)!=2114:raise ValueError('Native workflow mismatch')
        adapter.batch_size=batch_size or 4;adapter.backend='grelu' if backend!='builtin' else 'builtin'
        adapter._use_grelu_predict=adapter.backend=='grelu';return adapter,[]
    def post(url,fields=None,body=None,upload=None):
        if body is not None:return client.post(url,json=body)
        data={'csrf_token':csrf,**(fields or {})}
        if upload:data[upload[0]]=(io.BytesIO(upload[1].encode()),upload[2])
        return client.post(url,data=data,content_type='multipart/form-data' if upload else None)
    original_reader=workflows.predict.read_sequences
    def observed_reader(params):
        answer=original_reader(params)
        active['parsed_records']=answer[0]
        return answer
    original_predict=ModelAdapter.predict
    def observed_predict(owner,sequences):
        active['prepared_sequences']=list(sequences)
        active['adapter']=owner
        return original_predict(owner,sequences)
    with ExitStack() as stack:
        stack.enter_context(patch.object(ModelAdapter,'predict',observed_predict))
        stack.enter_context(patch.object(workflows.predict,'read_sequences',observed_reader))
        for target in [core.models,workflows.predict,workflows.vep,workflows.scan,workflows.attribution,workflows.design,workflows.explore]:
            stack.enter_context(patch.object(target,'load_model',real_adapter))
        for backend in ['grelu','builtin']:
            # For native website routes backend selection is auto. Builtin is
            # already verified explicitly above, not falsely claimed as the UI default.
            if backend=='builtin':continue
            capture('http_prediction_fasta',lambda:post('/predict',{'model':'human','fasta':f'>Z\n{first}\n>A\n{second}\n'}),
                    [first,second],['Z','A'],path='/predict')
            capture('http_prediction_upload',lambda:post('/predict',{'model':'human'},upload=('file',f'>Z\n{unknown}\n','input.fasta')),
                    [unknown],['Z'],path='/predict')
        capture('http_variant_pair',lambda:post('/api/compare',body={'model':'human','reference':first,'alternate':second}),
                [first,second],['reference','alternate'],path='/api/compare')
        for url in ['/allele-specific','/prioritize','/multi-allelic']:
            capture('http_'+url[1:],lambda u=url:post(u,{'model':'human','fasta':f'>R\n{first}\n>A\n{second}\n'}),
                [first,second],['R','A'],path=url)
        for url in ['/design','/attribution','/mutagenesis']:
            capture('http_'+url[1:],lambda u=url:post(u,{'model':'human','text':first,'start':'1','end':'1','target':'0'}),
                [first],['sequence_1'],path=url)
        csv='variant_id,sequence,pos,ref,alt\n001,'+first+',1,'+first[0]+','+second[0]+'\n'
        capture('http_variant_csv',lambda:post('/batch-variants',{'model':'human','text':csv,'shuffles':'0'}),
                [first,second],['001:reference','001:alternate'],path='/batch-variants')
        for i,position in enumerate([1,1057,2114]):
            replacement={'A':'C','C':'G','G':'T','T':'A'}[first[position-1]]
            mutant=first[:position-1]+replacement+first[position:]
            text=f'variant_id,sequence,pos,ref,alt\n001,{first},{position},{first[position-1]},{replacement}\n'
            capture('variant_position_'+str(position),lambda t=text:post('/batch-variants',{'model':'human','text':t,'shuffles':'0'}),
                    [first,mutant],['001:reference','001:alternate'],path='/batch-variants')
        motif='TATAAA';mutant=motif+first[len(motif):]
        capture('http_motif_simulation',lambda:post('/motif-simulation',{'model':'human','text':first,'motif':motif,
            'positions':'1','orientation':'+','gaps':'0','target':'0'}),[first,mutant],['reference','motif:position1'],path='/motif-simulation')
        for key,locus in json.loads((ROOT/'curated_loci.json').read_text()).items():
            capture('http_curated_'+key,lambda k=key:post('/saturation',{'locus':k}),[locus['dna']],[key],path='/saturation')
        # Invalid and duplicate inputs must be rejected before any hook fires.
        for name,text in [('invalid',first[:20]+'X'+first[21:]),('duplicate',f'>same\n{first}\n>same\n{second}\n')]:
            active.clear();response=post('/predict',{'model':'human','fasta':text})
            file=out/(name+'-rejection.json');save(file,{'status':response.status_code,'body':response.text,'model_hook_reached':'tensor' in active})
            check('http_reject_'+name,response.status_code==400 and 'tensor' not in active,
                  'Real Flask/validation path must reject before tensor/model boundary.',[file.name],kind='negative guard',path='/predict')
        # Comparison helper preflights every window before inference. Replay its
        # real prepared strings into each corresponding real adapter; a complete
        # two-model numerical request cannot finish without inference and is not claimed.
        for n in [1,2113,2114,2115,2117]:
            seq=(first+second)[:n]
            prepared=workflows.explore.prepare_comparison_inputs([MODELS['human'],MODELS['demo']],
                [{'id':'context','sequence':seq}],policy='center')
            save(out/f'comparison-prepared-{n}.json',prepared)
            detail_ok=True
            for item,length in zip(prepared,[2114,200]):
                offset=(length-len(seq))//2 if len(seq)<=length else -((len(seq)-length)//2)
                detail=item['details'][0]
                detail_ok=detail_ok and detail['offset']==offset and detail['original_length']==len(seq) and detail['model_length']==length and detail['sequence']==expected_resize(seq,length) and bool(detail['transformation'])
            check('explicit_transform_offsets_'+str(n),detail_ok,'Recorded original/final lengths, transformation descriptions and signed crop/padding offsets equal independent rules.',[f'comparison-prepared-{n}.json'],kind='transformation record',path='/model-comparison')
            for j,(target,length) in enumerate([(adapter,2114),(demo,200)]):
                target.batch_size=1
                capture(f'comparison_{n}_{length}',lambda t=target,item=prepared[j]:t.predict(item['sequences']),
                        [expected_resize(seq,length)],['context'],path='/model-comparison')
        for n in [2113,2114,2115]:
            seq=(first+second)[:n];error=None;actual_prepared=None
            try:actual_prepared=workflows.explore.prepare_comparison_inputs([MODELS['human']],[{'id':'strict','sequence':seq}],policy='strict')
            except ValueError as exc:error=str(exc)
            file=out/f'strict-{n}.json';save(file,{'input_length':n,'error':error,'prepared':actual_prepared})
            check('strict_window_'+str(n),(error is None and actual_prepared[0]['sequences']==[seq]) if n==2114 else error is not None,
                  'Strict policy accepts the verified workflow window and rejects shorter/longer inputs before encoding.',[file.name],kind='negative guard' if n!=2114 else 'preparation',path='/model-comparison')
        capture('http_duplicate_dna',lambda:post('/predict',{'model':'human','fasta':f'>copy_1\n{first}\n>copy_2\n{first}\n'}),[first,first],['copy_1','copy_2'],path='/predict')
        capture('http_comparison_first_handoff',lambda:post('/model-comparison',{'model_a':'human','model_b':'demo','text':first,'policy':'center'}),
                [first],['sequence_1'],path='/model-comparison')
        capture('http_demo_odd_padding',lambda:post('/api/compare',body={'model':'demo','reference':'ACGTA','alternate':'CCGTA'}),
                [expected_resize('ACGTA',200),expected_resize('CCGTA',200)],['reference','alternate'],path='demo comparison')
    for size,sequence in [(7,'ACGTN'),(4,'ACGTNTA'),(8,'ACGTNTA')]:
        expected=expected_resize(sequence,size)
        ok,evidence=compare(f'helper_resize_{size}_{len(sequence)}',one_hot([sequence],size),oracle([expected]),['resize'])
        check(f'helper_resize_{size}_{len(sequence)}',ok,'Actual helper center crop/N-padding compared at every base/channel, including odd differences.',[evidence],path='resize helper')
    check('no_weighted_model_execution',not weighted_attempts,'Architecture hooks stopped before all convolution/linear bodies.',
          ['safety.json'],kind='safety')
    check('weights_unchanged',state_hash(module)==weights_before and sha(path)==initial_sha,
          'Loaded GM12878 tensor bytes and pinned checkpoint SHA-256 unchanged.',['safety.json'],kind='preservation')
    for name,workflow,reason in [
        ('sequence_shift','augmentation','No website deterministic inference shift policy exists; training metadata is not an active inference transformation.'),
        ('left_right_crop','resize','Website supports center resize, not independent left/right crop policies. No behavior added.'),
        ('curated_mutant_tensors','/saturation','Mutation generation follows a baseline numerical prediction. Only preset/reference tensor captured; mutation tensors remain PARTIAL.'),
        ('post_reference_mutagenesis','/mutagenesis','Later scan mutants require the reference score. No fake prediction is returned to advance this branch.'),
        ('full_comparison_request','/model-comparison','Both model-specific prepared paths captured independently; continuous second HTTP handoff requires first prediction, so remains PARTIAL.'),
        ('training_job','training/fine-tuning','Jobs and optimizers are not executed. Shared encoding is checked but no full trainer path is certified.'),
        ('gpu_device','device','Website tested CPU path; CUDA/MPS transfers not run.'),
        ('unused_dataset_classes','library datasets','SeqDataset/VariantDataset are not used by these website predict_on_seqs paths; no artificial dataset coverage is claimed.')]:
        file=out/(name+'-scope.json');save(file,{'workflow':workflow,'reason':reason})
        check(name,False,reason,[file.name],kind='scope',status='SKIP',path=workflow)
    save(out/'safety.json',{'checkpoint_path':str(path),'checkpoint_sha256_before':initial_sha,'checkpoint_sha256_after':sha(path),
        'loaded_state_sha256_before':weights_before,'loaded_state_sha256_after':state_hash(module),
        'weighted_forward_attempts':weighted_attempts,'numeric_model_outputs_created':0,'optimizer_steps':0,
        'training_executed':False,'channel_order_verified':['A','C','G','T'],'N_policy':'index 4, all four one-hot channels zero',
        'website_adapter_reuse':'Actual preloaded checkpoint module reused to avoid repeated deserialization; only the loader factory is redirected. Actual adapter encoders, gReLU functions, device transfer, Lightning.format_input and architecture pre-hooks run.',
        'batch_scope':'Actual shared batch iterator traversed fully; each chunk enters an independent stopped real-adapter invocation. No concatenated numerical result is generated or compared.'})
    for handle in handles:handle.remove()
    return {'checks':results,'counts':{s:sum(r['status']==s for r in results) for s in ['PASS','FAIL','SKIP','BLOCKED']},
            'status':'FAIL' if any(r['status']=='FAIL' for r in results) else 'PASS',
            'source_gate2':str(source2),'versions':{name:version(name) for name in ['torch','numpy','grelu','flask','pandas']},
            'numeric_inference_executed':False,'gate4_started':False}


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=BASE/'runs'/stamp;out.mkdir(parents=True)
    before=protected();save(out/'protected-before.json',before)
    env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','HF_HUB_OFFLINE':'1',
         'MPLBACKEND':'Agg','MPLCONFIGDIR':str(out/'matplotlib-cache')}
    import psutil
    peak=0;blocked=None;start=time.monotonic()
    with (out/'worker.log').open('w') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker',str(out)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        tracked=psutil.Process(process.pid)
        while process.poll() is None:
            try:peak=max(peak,tracked.memory_info().rss)
            except psutil.NoSuchProcess:pass
            if peak>1_800_000_000:blocked='Worker exceeded 1.8 GB RSS ceiling.'
            if psutil.virtual_memory().available<650_000_000:blocked='Available memory below 650 MB reserve.'
            if time.monotonic()-start>240:blocked='Worker exceeded 240 second limit.'
            if blocked:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                break
            time.sleep(.2)
    if (out/'worker-results.json').exists():result=json.loads((out/'worker-results.json').read_text())
    else:result={'checks':[],'counts':{'PASS':0,'FAIL':0,'SKIP':0,'BLOCKED':0},'status':'BLOCKED' if blocked else 'FAIL'}
    if blocked or process.returncode!=0:
        status='BLOCKED' if blocked else 'FAIL';result['checks'].append({'id':'worker_completion','status':status,
            'kind':'execution','workflow':'runner','explanation':blocked or 'Worker failed; inspect saved traceback/log.','evidence':['worker.log']})
    after=protected();save(out/'protected-after.json',after)
    unchanged=all(after.get(k)==v for k,v in before.items())
    result['checks'].append({'id':'earlier_evidence_preserved','status':'PASS' if unchanged else 'FAIL','kind':'preservation',
        'workflow':'preservation','explanation':'All pre-existing Phase 1/Gate 1/Gate 2 evidence hashes unchanged; new run files are permitted.',
        'evidence':['protected-before.json','protected-after.json']})
    result['counts']={s:sum(c['status']==s for c in result['checks']) for s in ['PASS','FAIL','SKIP','BLOCKED']}
    result['status']='FAIL' if result['counts']['FAIL'] else 'BLOCKED' if result['counts']['BLOCKED'] else 'PASS'
    result.update(gate=3,peak_worker_rss_bytes=peak,elapsed_seconds=time.monotonic()-start)
    files=['core/models.py','core/sequences.py','core/dna_inputs.py','workflows/predict.py','workflows/explore.py','workflows/vep.py','workflows/attribution.py','lab_compute.py']
    result['source_hashes']={name:sha(ROOT/name) for name in files};result['runner_sha256']=sha(Path(__file__))
    installed=['grelu/sequence/format.py','grelu/lightning/__init__.py','grelu/data/dataset.py','grelu/data/augment.py']
    library=ROOT/'.venv/lib/python3.14/site-packages'
    save(out/'installed-source-inventory.json',{name:{'path':str(library/name),'sha256':sha(library/name)} for name in installed})
    save(out/'results.json',result)
    (out/'runner-source.py.txt').write_text(Path(__file__).read_text())
    save(BASE/'latest.json',{'run':str(out.relative_to(BASE)),'status':result['status'],'counts':result['counts']})
    coverage=json.loads((BASE/'workflow-coverage.json').read_text())
    for item in coverage:
        relevant=[r for r in result['checks'] if r['workflow'] in item['check_workflows']]
        item['check_ids']=[r['id'] for r in relevant]
        if any(r['status']=='FAIL' for r in relevant):item['classification']='FAILED'
    save(out/'workflow-coverage.json',coverage)
    matrix=['# Workflow tensor coverage','','Labels certify only the stated input/tensor scope, never prediction output correctness.','','| Workflow | Classification | Verified scope / limits |','|---|---|---|']
    for item in coverage:matrix.append(f"| {item['workflow']} | {item['classification']} | {item['scope']} |")
    (out/'WORKFLOW_COVERAGE.md').write_text('\n'.join(matrix)+'\n')
    lines=['# Gate 3 — preprocessing and tensor encoding','',f"**{result['status']} — {result['counts']}**",'',
        'Real model input tensors are captured by forward-pre-hooks. No weighted model computation or numeric prediction output is permitted. Expected tensors use literal independent A/C/G/T vectors; N is all zeros. This is not inference equivalence or biological analysis.','',
        '| Check | Kind | Status | Explanation | Evidence |','|---|---|---|---|---|']
    for row in result['checks']:
        refs=', '.join(f'[{p}]({p})' for p in row['evidence'])
        lines.append(f"| {row['id']} | {row['kind']} | {row['status']} | {row['explanation']} | {refs} |")
    lines += ['', '## Findings and trace','',
        'Five optional native inverse-conversion checks fail on single-base tensors in this installed environment. A/C/G/T decode as N and N raises IndexError. Forward encodings and captured website input tensors are checked separately; no inverse failure is relabeled as a skip. The website does not call this inverse function. See integer-roundtrip/*.json and the bugs report.', '',
        'Read [workflow coverage](WORKFLOW_COVERAGE.md), [actual pipeline and encoding specification](../../PREPROCESSING.md), and [bugs/fixes](../../BUGS_AND_FIXES.md). [Safety evidence](safety.json) records the checkpoint fingerprint before/after, loaded state identity, zero weighted forward attempts and zero numeric outputs. [Installed source inventory](installed-source-inventory.json) pins the inspected library code.', '',
        f"Versions: {result.get('versions')}. Peak worker RSS: {peak:,} bytes. Elapsed: {result['elapsed_seconds']:.1f} seconds.", '',
        '## Scope and reproducibility','',
        'Actual/expected .npy arrays and comparison JSON record sequence IDs, shape, dtype, device, finite values and every mismatch (position/channel/expected/actual). The fixture JSON identifies the accepted Gate 2 source DNA. The model boundary is the real architecture input after gReLU Lightning.format_input, before its forward body. No expected tensor comes from a Studio or gReLU encoder.', '',
        'Batching uses the same iterator consumed by real adapter prediction/gradient methods. To avoid fabricated outputs, each actual chunk is captured in a separate stopped invocation; full numerical output concatenation is not tested. Comparison windows for both models are encoded/captured from the actual preflight helper; second continuous HTTP handoff remains partial. Curated mutation tensors generated after a reference prediction remain partial.', '',
        'Reproduce from repository root:', '', '```bash', '.venv/bin/python Testing/testing-phase-2/input-output/gate-3/verify_gate3.py', '```','',
        'Worker log and resource limits are retained. Earlier runs are never overwritten. Read all SKIP/PARTIAL limits; no Gate 4 execution is authorized or started.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(out/'REPORT.md'),'status':result['status'],'counts':result['counts'],'peak_worker_rss_bytes':peak},indent=2),flush=True)
    return 1 if result['status']=='FAIL' else 2 if result['status']=='BLOCKED' else 0


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        target=Path(sys.argv[2])
        try:save(target/'worker-results.json',worker(target))
        except BaseException:
            (target/'worker-error.log').write_text(traceback.format_exc());traceback.print_exc();sys.exit(1)
    else:sys.exit(main())
