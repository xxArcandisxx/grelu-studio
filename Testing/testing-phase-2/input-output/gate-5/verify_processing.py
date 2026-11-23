"""Gate 5 application tests: saved real raw-output replay + explicit synthetic edges.

Replaying a previously verified tensor is a postprocessing boundary fixture,
not new inference. No replay or synthetic check claims pretrained inference.
"""
from contextlib import ExitStack
from html.parser import HTMLParser
import io
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + '\n')


class Tables(HTMLParser):
    def __init__(self):
        super().__init__(); self.tables=[]; self.row=None; self.cell=None; self.table=None
    def handle_starttag(self, tag, attrs):
        if tag=='table': self.table=[]
        if tag=='tr': self.row=[]
        if tag in ['td','th']: self.cell=''
    def handle_data(self, data):
        if self.cell is not None: self.cell+=data
    def handle_endtag(self, tag):
        if tag in ['td','th'] and self.row is not None: self.row.append(self.cell.strip()); self.cell=None
        if tag=='tr' and self.table is not None: self.table.append(self.row); self.row=None
        if tag=='table': self.tables.append(self.table); self.table=None


class Recorder:
    def __init__(self, out): self.out=out; self.rows=[]; self.context={}
    def check(self, name, actual, expected, formula, mode='SAVED_REAL_GATE4_OUTPUT', workflow='shared', tolerance=1e-6):
        if isinstance(actual,(int,float,np.number)) and isinstance(expected,(int,float,np.number)):
            absolute=abs(float(actual)-float(expected)); relative=absolute/max(abs(float(expected)),1e-300)
            ok=math.isfinite(float(actual)) and math.isclose(float(actual),float(expected),rel_tol=tolerance,abs_tol=1e-7)
        else: absolute=relative=None; ok=actual==expected
        path=self.out/'checks'/(name+'.json')
        evidence=dict(context=self.context.copy(),transformations=['native output already produced','task selection / single-output pass-through or explicit mean','Studio formula: '+formula,'display formatting only for HTML'],id=name,status='PASS' if ok else 'FAIL',actual=actual,expected=expected,
            absolute_error=absolute,relative_error=relative,rtol=tolerance,atol=1e-7,
            formula=formula,mode=mode,workflow=workflow,
            semantics='Native model score; no probability/log/unit conversion assumed.')
        save(path,evidence); self.rows.append(dict(id=name,status=evidence['status'],workflow=workflow,mode=mode,evidence=str(path.relative_to(self.out))))
        print(evidence['status'],name,flush=True)
    def skip(self,name,why,workflow):
        path=self.out/'checks'/(name+'.json'); save(path,dict(status='SKIP',reason=why))
        self.rows.append(dict(id=name,status='SKIP',workflow=workflow,mode='LIMITATION',evidence=str(path.relative_to(self.out))))


def main(out):
    from core.models import _norm3
    from core.sequences import aggregate
    from core.output_processing import predict_checked
    from report_builder import chart
    from variant_views import comparisons
    gate4=BASE.parent/'gate-4/runs/20261122T163955672475Z'
    fixture=json.loads((gate4/'fixtures.json').read_text())
    raw=np.load(gate4/'direct/all-batch1.npy',allow_pickle=False)
    lookup={r['sequence']:raw[i].copy() for i,r in enumerate(fixture['reference_queries'])}
    save(out/'fixture-provenance.json',dict(source=str(gate4),identity=fixture['identity'],
        mode='REPLAY_OF_REAL_SAVED_RAW_OUTPUTS',note='Loaded Gate 4 direct arrays; no new inference or fabricated prediction is used.'))
    os.environ.update(GRELU_HUMAN_CHECKPOINT=fixture['checkpoint_path'],GRELU_HUMAN_INPUT_LENGTH='2114',
                      GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase',GRELU_SECRET_KEY='gate5-isolated')
    import core.models, workflows.predict, workflows.vep, workflows.scan, workflows.explore, workflows.design, workflows.attribution
    import curated_scan,studio_services
    from studio_web import app,MODELS
    state=out/'state';state.mkdir(parents=True)
    app.config.update(TESTING=True,LAB_ROOT=str(state/'lab'),ACCOUNT_DATABASE=str(state/'accounts.db'),HISTORY_DATABASE=str(state/'history.db'))
    client=app.test_client();client.get('/experiments')
    with client.session_transaction() as session:csrf=session['csrf_token']
    current={'names':fixture['identity']['tasks'],'values':lookup,'notes':[],'fixture_mode':'SAVED_REAL_GATE4_OUTPUT'}
    source_calls=[]
    class Replay:
        def __init__(self,spec):
            self.seq_len=2114;self.task_names=current['names'];self.meta={**spec,'backend':'gate5_postprocessing_boundary','verification_fixture_mode':current['fixture_mode']}
            self.backend='gate5_saved_output_replay';self.device='cpu';self.notes=list(current['notes'])
            self.module=SimpleNamespace(data_params={},performance=None)
        def predict(self,sequences):
            values=np.stack([current['values'][s] for s in sequences]);source_calls.append(dict(sequences=sequences,values=values.tolist()))
            return values
    def load(spec,*a,**kw): return Replay(spec),['Gate 5 boundary fixture: '+current['fixture_mode']+'; synthetic values are not pretrained model predictions; replay is not fresh inference.']
    def post(case):
        if case['kind']=='json': return client.post(case['url'],json=case['body'])
        data={'csrf_token':csrf,**case.get('fields',{})}
        if case.get('upload'):data['file']=(io.BytesIO(case['upload'].encode()),'input.fasta')
        return client.post(case['url'],data=data,content_type='multipart/form-data' if case.get('upload') else None)
    rec=Recorder(out)
    with ExitStack() as stack:
        for target in [core.models,workflows.predict,workflows.vep,workflows.scan,workflows.explore,workflows.design,workflows.attribution]:
            stack.enter_context(patch.object(target,'load_model',load))
        for case in fixture['cases']:
            if case['kind']=='adapter' or case.get('fault') or case.get('stop_after_calls'):continue
            source_calls.clear();response=post(case);name=case['id']
            rec.context=dict(model_identity=fixture['identity'],task_names=fixture['identity']['tasks'],source_case=name,sequence_ids_in_input_order=[r['id'] for r in case['expected_calls'][0]],raw_origin='Gate 4 direct/all-batch1.npy; actual boundary recorded in '+name+'-raw-boundary.json')
            (out/(name+'.html')).write_text(response.text)
            if case['kind']=='json': result=response.get_json()
            else:
                match=re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>',response.text,re.S)
                result=json.loads(match[1]) if match else {}
            save(out/(name+'.json'),result);save(out/(name+'-raw-boundary.json'),source_calls)
            rec.context['warnings']=result.get('warnings',[])
            rec.check(name+'_status',response.status_code,200,'Real Flask output-processing handler receives saved real tensors.',workflow=case['workflow'])
            if response.status_code!=200:continue
            tables=Tables();tables.feed(response.text)
            expected_calls=case['expected_calls']
            def score(record): return float(lookup[record['sequence']][0,0])
            if 'predictions' in result.get('results',{}):
                predictions=result['results']['predictions']
                rec.check(name+'_native_semantics',result['model']['output_semantics']['output_positions'],1,'Verified native single output; no 1000-bin aggregation.',workflow=case['workflow'])
                rec.check(name+'_unknown_units',result['model']['output_semantics']['units'],'not established','No biological units or probability semantics invented.',workflow=case['workflow'])
                rec.check(name+'_ids',[p['id'] for p in predictions],[r['id'] for r in expected_calls[0]],'Preserve record order and IDs.',workflow=case['workflow'])
                for i,(observed,source) in enumerate(zip(predictions,expected_calls[0])):
                    for task in fixture['identity']['tasks']:
                        rec.check(f'{name}_score_{i}',observed['scores'][task],score(source),'Single native output is passed through unchanged.',workflow=case['workflow'])
                        rec.check(f'{name}_rc_{i}',observed['reverse_complement_scores'][task],score(expected_calls[1][i]),'Select corresponding raw reverse-complement record; no extra transformation.',workflow=case['workflow'])
                        rec.check(f'{name}_plot_{i}',observed['tracks'][task],[score(source)],'Single plot value retains full native precision; no spatial values invented.',workflow=case['workflow'])
                if case['kind']=='json':
                    raw_ref,raw_alt=map(score,expected_calls[0]);offset=next(i for i,(a,b) in enumerate(zip(case['body']['reference'],case['body']['alternate'])) if a!=b)
                    rec.check(name+'_position',result['inputs']['position'],offset+1,'Mutation position is one-based.',workflow=case['workflow'])
                    rec.check(name+'_alleles',result['inputs']['substitution'],case['body']['reference'][offset]+'>'+case['body']['alternate'][offset],'Alleles map to the input difference.',workflow=case['workflow'])
                    rec.check(name+'_delta_formula',predictions[1]['scores'][task]-predictions[0]['scores'][task],raw_alt-raw_ref,'Alternate minus reference, signed model-score change.',workflow=case['workflow'])
                elif case['workflow']=='/predict':
                    for i,source in enumerate(expected_calls[0]):
                        wanted=[source['id'],task,format(score(source),'.6g'),format(score(expected_calls[1][i]),'.6g')]
                        rec.check(f'{name}_html_{i}',any(row[:4]==wanted for table in tables.tables for row in table),True,'Actual rendered table: IDs/task and six significant digits; full precision remains in JSON.',workflow=case['workflow'])
                else:
                    effects=result['results']['comparisons'];expected=[]
                    for i,source in enumerate(expected_calls[0][1:],1):
                        expected.append((source['id'],score(source)-score(expected_calls[0][0]),score(expected_calls[1][i])-score(expected_calls[1][0])))
                    if case['workflow']=='/prioritize':expected=sorted(expected,key=lambda r:-abs(r[1]))
                    rec.check(name+'_ranking',[v['alternate'] for v in effects],[v[0] for v in expected],'Stable decreasing absolute delta for prioritize; source order otherwise.',workflow=case['workflow'])
                    for i,(observed,expected_row) in enumerate(zip(effects,expected)):
                        alternative=next(r for r in expected_calls[0] if r['id']==expected_row[0])
                        differences=sum(a!=b for a,b in zip(expected_calls[0][0]['sequence'],alternative['sequence']))
                        rec.check(f'{name}_difference_count_{i}',observed['differences'],differences,'Independent base-by-base mismatch count for this reference/alternate pair.',workflow=case['workflow'])
                        rec.check(f'{name}_delta_{i}',observed['delta'],expected_row[1],'alternate_score - reference_score; signed native-score effect.',workflow=case['workflow'])
                        rec.check(f'{name}_rcdelta_{i}',observed['reverse_complement_delta'],expected_row[2],'RC alternate - RC reference.',workflow=case['workflow'])
                        original=next(p for p in predictions if p['id']==observed['alternate'])
                        expected_cells=[task,predictions[0]['id'],original['id'],str(observed['differences']),format(observed['reference_score'],'.6g'),format(observed['alternate_score'],'.6g'),format(expected_row[1],'+.6g'),format(expected_row[2],'+.6g')]
                        if case['workflow']=='/prioritize':expected_cells.insert(0,str(i+1))
                        rec.check(f'{name}_html_effect_{i}',any(row==expected_cells for table in tables.tables for row in table),True,'Every rendered effect-table field, sign, rank, task, ID and six-digit precision.',workflow=case['workflow'])
            elif case['workflow']=='/batch-variants':
                rows=result['results']['variants'];n=len(rows);refs=expected_calls[0][:n];alts=expected_calls[0][n:]
                expected_deltas=[score(a)-score(r) for r,a in zip(refs,alts)]
                for i,row in enumerate(rows):
                    rec.check(f'{name}_reference_{i}',row['ref_scores'][task],score(refs[i]),'Reference row selects its own raw score.',workflow=case['workflow'])
                    rec.check(f'{name}_alternate_{i}',row['alt_scores'][task],score(alts[i]),'Alternate row selects its own raw score.',workflow=case['workflow'])
                    rec.check(f'{name}_delta_{i}',row['effects'][task],expected_deltas[i],'Alternate minus reference, computed in independent Python double precision.',workflow=case['workflow'])
                    expected_rc=score(expected_calls[1][n+i])-score(expected_calls[1][i])
                    agreement=float((expected_deltas[i]>0)-(expected_deltas[i]<0)==(expected_rc>0)-(expected_rc<0))
                    rec.check(f'{name}_rcsign_{i}',row['rc_sign_agreement'],agreement,'Equality of signed forward and RC effects; not biological accuracy.',workflow=case['workflow'])
                    rec.check(f'{name}_untested_p_{i}',row['pvalues'],None,'Zero shuffles means p-values are Not tested, never 0.',workflow=case['workflow'])
                    expected_cells=[row['variant_id'],task,format(score(refs[i]),'.6g'),format(score(alts[i]),'.6g'),format(expected_deltas[i],'+.6g'),'Not tested']
                    rec.check(f'{name}_html_values_{i}',any(cells==expected_cells for table in tables.tables for cells in table),True,'Actual detailed HTML reference, alternate, delta, task and missing-p label.',workflow=case['workflow'])
                expected_ids=[rows[i]['variant_id'] for i in sorted(range(n),key=lambda i:-abs(expected_deltas[i]))]
                rec.check(name+'_ranking',[r['variant_id'] for r in result['results']['ranked_variants']],expected_ids,'Stable absolute-effect ranking; no top-k filter is implemented.',workflow=case['workflow'])
                rec.check(name+'_not_tested_label','Not tested' in response.text,True,'HTML does not invent p=0 for an unperformed background test.',workflow=case['workflow'])
            elif case['workflow']=='/mutagenesis':
                observed=result['results'];reference=score(expected_calls[0][0]);position=int(case['fields']['start']);bases=[b for b in 'ACGT' if b!=expected_calls[0][0]['sequence'][position-1]]
                rec.check(name+'_baseline',observed['reference_score'],reference,'Reference raw native score.',workflow=case['workflow'])
                rec.check(name+'_index',observed['positions'][0]['position'],position,'One-based position alignment.',workflow=case['workflow'])
                for b,r in zip(bases,expected_calls[1]):rec.check(name+'_base_'+b,observed['positions'][0]['deltas'][b],score(r)-reference,'Mutant raw score - reference, correct A/C/G/T cell.',workflow=case['workflow'])
                rec.check(name+'_html_axes',tables.tables[0][0],['Position','Reference','A','C','G','T'],'Actual mutation table axes and base order.',workflow=case['workflow'])
                expected_cells=[str(position),expected_calls[0][0]['sequence'][position-1]]+[format(observed['positions'][0]['deltas'][b],'+.4g') if b in bases else 'Reference' for b in 'ACGT']
                rec.check(name+'_html_cells',tables.tables[0][1],expected_cells,'Actual heatmap cells: signed four significant digits and unchanged-base marker.',workflow=case['workflow'])
            elif case['workflow']=='/motif-simulation':
                observed=result['results'];reference=score(expected_calls[0][0]);mutant=score(expected_calls[0][1]);row=observed['conditions'][0]
                rec.check(name+'_baseline',observed['reference_score'],reference,'Original DNA native score.',workflow=case['workflow'])
                rec.check(name+'_score',row['score'],mutant,'Actual replacement DNA native score.',workflow=case['workflow'])
                rec.check(name+'_delta',row['delta'],mutant-reference,'Replacement minus unmodified reference.',workflow=case['workflow'])
                rec.check(name+'_control',row['control_score'],None,'No control requested: missing control is explicit, not numeric zero.',workflow=case['workflow'])
            elif case['workflow']=='/model-comparison':
                for i,(observed,source) in enumerate(zip(result['results']['comparisons'],expected_calls[0])):
                    for key in ['score_a','score_b']:rec.check(f'{name}_{key}_{i}',observed[key],score(source),'Both same-model selections retain the same raw score.',workflow=case['workflow'])
                    rec.check(f'{name}_difference_{i}',observed['raw_difference'],0.0,'Model B - A; same model, same input.',workflow=case['workflow'])
                rec.check(name+'_scale_warning','different scales' in response.text or 'scales or training objectives' in response.text,True,'Unjustified cross-model scale equivalence is not claimed.',workflow=case['workflow'])
        # Actual curated calculations and HTML for the Gate 4 verified one-position subset.
        rec.context=dict(model_identity=fixture['identity'],task_names=fixture['identity']['tasks'],source_case='curated-subset',sequence_ids_in_input_order=['reference','three alternate bases'],raw_origin='Verified Gate 4 curated reference and first substitutions')
        curated_case=next(c for c in fixture['cases'] if c['id']=='http_curated_subset')
        with patch.object(curated_scan,'POSITIONS',range(1053,1054)):
            response=post(curated_case)
        (out/'curated-subset.html').write_text(response.text)
        match=re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>',response.text,re.S);observed=json.loads(match[1])
        save(out/'curated-subset.json',observed);reference=score(curated_case['expected_calls'][0][0]);locus=curated_scan.LOCI[curated_case['fields']['locus']]
        rec.check('curated_baseline',observed['reference_score'],reference,'Actual handler; harness limits its constant position list to the previously verified subset.',workflow='/saturation')
        rec.check('curated_genomic_position',observed['positions'][0]['position'],locus['start']+1053+1,'Genome start (zero-based) + input index + 1.',workflow='/saturation')
        bases=[b for b in 'ACGT' if b!=locus['dna'][1053]]
        for b,r in zip(bases,curated_case['expected_calls'][1]):rec.check('curated_'+b,observed['positions'][0]['deltas'][b],score(r)-reference,'Correct mutant/base cell minus reference.',workflow='/saturation')
        current['notes']=json.loads((gate4/'website/http_fault_recovery.json').read_text())['calls'][0]['notes']
        with patch.object(curated_scan,'POSITIONS',range(1053,1054)):
            response=post(curated_case)
        (out/'curated-warning.html').write_text(response.text)
        match=re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>',response.text,re.S)
        warning_result=json.loads(match[1]);save(out/'curated-warning.json',warning_result)
        rec.check('curated_warning_metadata',all(w in warning_result.get('warnings',[]) for w in current['notes']),True,'Actual curated handler propagates saved loader/fallback warning metadata.',workflow='/saturation')
        rec.check('curated_warning_html',all(w in response.text for w in current['notes']),True,'Actual curated HTML presents warning text with its numeric subset.',workflow='/saturation')
        current['notes']=[]
        # Controlled synthetic processing fixtures only, never pretrained results.
        rec.context=dict(model_identity='SYNTHETIC_PROCESSING_FIXTURE_NOT_PRETRAINED',raw_origin='Explicit boundary numeric constants; no inference or optimization outcome claimed.')
        current['fixture_mode']='SYNTHETIC_PROCESSING_ONLY'
        task='synthetic_task';predictions=[dict(id='ref',scores={task:0},reverse_complement_scores={task:0})]
        predictions += [dict(id=name,scores={task:value},reverse_complement_scores={task:value}) for name,value in [('first',-2),('second',2),('zero',0),('tiny',1e-9)]]
        ranked=comparisons('prioritize',predictions,{p['id']:1 for p in predictions[1:]})
        rec.check('synthetic_signed_ties',[r['alternate'] for r in ranked],['first','second','tiny','zero'],'Stable tie order from original IDs, decreasing absolute signed delta.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='rankings')
        rec.check('synthetic_signed_values',[r['delta'] for r in ranked],[-2,2,1e-9,0],'No probability clipping, abs used only for ranking.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='rankings')
        for p in predictions:
            p['scores']['task2']=-p['scores'][task]
            p['reverse_complement_scores']['task2']=-p['reverse_complement_scores'][task]
        ranked2=comparisons('prioritize',predictions,{p['id']:1 for p in predictions[1:]})
        rec.check('synthetic_multitask_ranks',[r['rank_within_track'] for r in ranked2],[1,2,3,4,1,2,3,4],'Independent within-task ranks, stable absolute-effect ties.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='rankings')
        current['names']=['on','off'];first=fixture['cases'][0]['records'][0]['sequence']
        current['values']={first:np.array([[3.0],[1.0]],dtype=np.float32)}
        design=workflows.design.handle_design_workflow(dict(model=MODELS['human'],device='cpu',seeds=[first],on_targets=[0],off_targets=[1],max_iter=0,n_baseline=0,output_dir=str(out/'synthetic-objective')))
        rec.check('synthetic_on_off_objective',design['results']['designs'][0]['score_before'],2.0,'mean(on=3)-mean(off=1); zero iterations, no fabricated optimization outcome.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/design')
        comparison=workflows.explore.compare_models([MODELS['human'],MODELS['human']],[dict(id='synthetic',sequence=first)],[0,1])
        rec.check('different_task_scale_difference',comparison['results']['comparisons'][0]['raw_difference'],None,'Different/unverified task scales: omit subtraction instead of assuming comparability.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/model-comparison')
        rec.check('different_task_scale_policy',comparison['results']['scale_comparison']['comparable'],False,'Only identical model configurations and identical tasks permit a raw difference.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/model-comparison')
        synthetic_compare=dict(kind='form',url='/model-comparison',fields=dict(model_a='human',model_b='human',text=first,policy='strict',target_a='0',target_b='1'))
        response=post(synthetic_compare)
        (out/'synthetic_scale_policy.html').write_text(response.text)
        match=re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>',response.text,re.S)
        synthetic_result=json.loads(match[1]) if match else {}
        save(out/'synthetic_scale_policy.json',synthetic_result)
        rec.check('unverified_scales_html',response.status_code==200 and 'Model A native scores (scale not harmonized)' in response.text and 'Model B native scores (scale not harmonized)' in response.text,True,'Actual synthetic-boundary Flask HTML gives each unverified scale a separate chart, never a shared-scale comparison.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/model-comparison')
        rec.check('unverified_scales_missing_difference',synthetic_result['results']['comparisons'][0]['raw_difference'],None,'Actual response has an explicit missing unsupported difference, not zero.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/model-comparison')
        current['names']=fixture['identity']['tasks'];current['values']=lookup
        for label,values in [('zero',[[[0.0]]]),('negative',[[[-2.0]]]),('tiny',[[[1e-30]]]),('large',[[[1e30]]]),('overflow32',[[[1e38]*8]]),('multitask',[[[1,3],[2,6]]])]:
            array=np.array(values,dtype=np.float32);expected=[[math.fsum(float(x) for x in channel)/len(channel) for channel in record] for record in array]
            rec.check('aggregate_'+label,aggregate(array,0).tolist(),expected,'Explicit sum(values)/count; L=1 pass-through.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='arithmetic')
        actual_downsample=workflows.predict._downsample(np.array([1,2,3,4,5],np.float32),2)
        rec.check('downsample_odd_groups',actual_downsample.tolist(),[2.0,4.5],'Independent means of first three and last two positions; synthetic multi-position output only.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='arithmetic')
        huge=np.full(8,1e38,np.float32)
        rec.check('downsample_finite_mean',workflows.predict._downsample(huge,1).tolist(),[float(huge[0])],'Finite float32 chart mean must not overflow intermediate sum; no invented GM12878 profile.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='arithmetic')
        for label,array,n in [('wrong_batch',np.ones((2,1,1)),1),('empty',np.empty((0,1,1)),1),('empty_positions',np.empty((1,1,0)),1),('nan',np.array([[[np.nan]]]),1),('inf',np.array([[[np.inf]]]),1),('negative_inf',np.array([[[-np.inf]]]),1),('rank4',np.ones((1,1,1,1)),1)]:
            error=None
            try:_norm3(array,n)
            except ValueError as exc:error=str(exc)
            rec.check('reject_'+label,error is not None,True,'Invalid raw boundary explicitly rejects; no null/zero substitution.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='guards')
        for label,names in [('missing_none',None),('missing',[]),('wrong_count',['x','y']),('duplicate',['x','x'])]:
            adapter=SimpleNamespace(task_names=names,predict=lambda sequences:np.ones((len(sequences),2 if label=='duplicate' else 1,1)))
            error=None
            try:predict_checked(adapter,['ACGT'])
            except ValueError as exc:error=str(exc)
            rec.check('metadata_'+label,error is not None,True,'Missing/incompatible task metadata blocks processing.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='guards')
        for label,series in [('signed',[('signed',[-2.,0.,2.])]),('equal',[('equal',[0.,0.])]),('huge',[('huge',[-1e308,1e308])])]:
            result=chart(label,series);save(out/('chart_'+label+'.json'),result)
            from decimal import Decimal,localcontext
            with localcontext() as dc:
                dc.prec=80
                values=series[0][1];lo=min(map(lambda x:Decimal(str(x)),values));hi=max(map(lambda x:Decimal(str(x)),values))
                expected_points=[]
                for i,v in enumerate(values):
                    x=Decimal(70)+Decimal(i)*500/max(1,len(values)-1)
                    y=Decimal(100) if lo==hi else Decimal(180)-(Decimal(str(v))-lo)/(hi-lo)*155
                    expected_points.append(f'{x:.3f},{y:.3f}')
            rec.check('chart_'+label+'_coordinates',result['series'][0]['points'],' '.join(expected_points),'Independent Decimal affine mapping of actual signed scores to SVG coordinates; equal series centered.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='charts')
            rec.check('chart_'+label+'_finite','nan' not in result['series'][0]['points'].lower() and 'inf' not in result['series'][0]['points'].lower(),True,'Actual SVG coordinate builder preserves signed finite series without clipping scores.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='charts')
        try:chart('invalid',[('ok',[1.]),('bad',[float('inf')])]);error=False
        except ValueError:error=True
        rec.check('chart_invalid_rejected',error,True,'Do not silently discard invalid chart series.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='charts')
        # Actual HTTP synthetic small-score precision and non-finite failure behavior.
        pair=next(c for c in fixture['cases'] if c['id']=='http_variant_1');ref,alt=pair['body']['reference'],pair['body']['alternate']
        current['values']={**lookup,ref:np.array([[1e-8]],np.float32),alt:np.array([[-1e-8]],np.float32)}
        response=post(pair);observed=response.get_json();task=current['names'][0]
        save(out/'synthetic_precision_api.json',observed)
        rec.check('small_score_plot_precision',observed['results']['predictions'][0]['tracks'][task],[float(np.float32(1e-8))],'Explicit synthetic fixture: chart dataset keeps native small value instead of rounding to zero.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/api/compare')
        current['values'][ref]=np.array([[np.nan]],np.float32);response=post(pair);save(out/'synthetic_nonfinite_api.json',response.get_json())
        rec.check('nonfinite_api_fails',response.status_code,500,'Invalid model output returns explicit failure, not a successful null/zero score.',mode='SYNTHETIC_PROCESSING_ONLY',workflow='/api/compare')
        current['values']=lookup
        rec.context=dict(model_identity=fixture['identity'],task_names=fixture['identity']['tasks'],raw_origin='Saved real Gate 4 raw tensors and actual fallback notes; replay metadata only')
        current['fixture_mode']='SAVED_REAL_GATE4_OUTPUT'
        historical=json.loads((gate4/'website/http_fault_recovery.json').read_text());current['notes']=historical['calls'][0]['notes']
        response=post(pair);observed=response.get_json();save(out/'warning_replay_api.json',observed)
        rec.check('warning_propagation',all(w in observed['warnings'] for w in current['notes']),True,'Replay of actual Gate 4 fallback warning metadata through current API processing.',workflow='/api/compare')
    rec.context=dict(scope='Actual HTML and JavaScript static source only; no interactive browser claim')
    js=(ROOT/'static/studio.js').read_text();html=client.get('/variants').text
    rec.check('neutral_score_title','Predicted signal by output bin' not in html and 'Model prediction scores' in html,True,'Actual HTML scientific label; no per-base/profile/probability claim.',mode='HTML_STATIC',workflow='comparison UI')
    rec.check('single_output_js_label','no positional profile' in js and 'single ?' in js,True,'Source inspection only: singleton chart path and neutral axis text.',mode='JS_STATIC_ONLY',workflow='comparison UI')
    rec.check('frontend_nonfinite_guard','Number.isFinite(number)' in js,True,'Static source guard rejects missing/non-finite values; not a browser execution claim.',mode='JS_STATIC_ONLY',workflow='comparison UI')
    for name,why,workflow in [('browser','No installed browser, Node, Playwright or Selenium runtime. Interactive JS behavior is NOT TESTED.','browser'),('distinct_model_scales','No second checkpoint has verified identity and comparable units. Same-model calculations only.','/model-comparison'),('full_curated_scan','Only the prior verified first position is tested via an explicit harness subset; remaining positions are PARTIAL.','/saturation'),('probability_log_ratio','No probability, sigmoid, log, exponentiation, percentage or ratio transform exists in tested scoring paths; none is invented.','semantics'),('background_pvalues','Real cases use zero shuffles. Empirical p-values/background calibration are not certified.','/batch-variants')]:rec.skip(name,why,workflow)
    save(out/'results.json',dict(checks=rec.rows,counts={s:sum(r['status']==s for r in rec.rows) for s in ['PASS','FAIL','SKIP','BLOCKED']},mode='POSTPROCESSING_NOT_INFERENCE',gate=5,gate6_started=False))


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
