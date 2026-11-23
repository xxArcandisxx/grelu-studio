"""Small real GM12878 Gate 5 design and attribution calculations, CPU only."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch
import numpy as np

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('gate5_records',BASE/'verify_processing.py');records=importlib.util.module_from_spec(spec);spec.loader.exec_module(records)
save=records.save


def main(out):
    import torch
    torch.set_num_threads(1)
    source=BASE.parent/'gate-4/runs/20261122T163955672475Z'
    fixture=json.loads((source/'fixtures.json').read_text())
    os.environ.update(GRELU_HUMAN_CHECKPOINT=fixture['checkpoint_path'],GRELU_HUMAN_INPUT_LENGTH='2114',GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from core.models import load_model,ModelAdapter
    import core.models,workflows.design,workflows.attribution
    adapter,_=load_model(dict(source='checkpoint',path=fixture['checkpoint_path'],seq_len=2114),'cpu',batch_size=1,backend='grelu')
    def state_hash():
        h=hashlib.sha256()
        for name,tensor in sorted(adapter.module.state_dict().items()):h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial_state=state_hash()
    from studio_web import app
    state=out/'state';state.mkdir(parents=True)
    app.config.update(TESTING=True,LAB_ROOT=str(state/'lab'),ACCOUNT_DATABASE=str(state/'accounts.db'),HISTORY_DATABASE=str(state/'history.db'))
    client=app.test_client();client.get('/experiments')
    with client.session_transaction() as session:csrf=session['csrf_token']
    rec=records.Recorder(out);rec.context=dict(model_identity=fixture['identity'],task_names=fixture['identity']['tasks'],raw_origin='Fresh real model phase-raw-N.npy files; independently computed processing formula');calls=[];phase='design';original_predict=ModelAdapter.predict
    def observe(owner,sequences):
        result=original_predict(owner,sequences)
        calls.append(dict(sequences=list(sequences),raw=result.copy()))
        np.save(out/f'{phase}-raw-{len(calls)-1:02d}.npy',result,allow_pickle=False)
        return result
    def real_loader(specification,*a,**kw):
        if Path(specification['path']).resolve()!=Path(fixture['checkpoint_path']).resolve():raise ValueError('Unexpected checkpoint')
        return adapter,[]
    def post(url,fields):
        response=client.post(url,data={'csrf_token':csrf,'model':'human',**fields})
        (out/(url[1:]+'.html')).write_text(response.text)
        match=re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>',response.text,re.S)
        result=json.loads(match[1]) if match else {}
        save(out/(url[1:]+'.json'),result)
        return response,result
    with ExitStack() as stack:
        for target in [core.models,workflows.design,workflows.attribution]:stack.enter_context(patch.object(target,'load_model',real_loader))
        stack.enter_context(patch.object(ModelAdapter,'predict',observe))
        sequence=fixture['cases'][0]['records'][0]['sequence']
        response,result=post('/design',dict(text=sequence,iterations='1',candidates='2',edit_start='1057',edit_end='1058',target='0'))
        rec.check('fresh_design_status',response.status_code,200,'Actual real-model one-iteration/two-candidate HTTP design.',mode='FRESH_REAL_INFERENCE',workflow='/design')
        if response.status_code==200:
            design=result['results']['designs'][0]
            raw=[c['raw'] for c in calls]
            baseline=float(raw[0][0,0,0]);candidates=[float(v[0,0]) for v in raw[1]]
            best=max(range(len(candidates)),key=lambda i:candidates[i]);accepted=candidates[best]>baseline+1e-9
            expected_final=candidates[best] if accepted else baseline
            expected_sequence=calls[1]['sequences'][best] if accepted else sequence
            gate4_reference=np.load(source/'direct/all-batch1.npy',allow_pickle=False)
            query_index=next(i for i,q in enumerate(fixture['reference_queries']) if q['sequence']==sequence)
            rec.check('fresh_raw_gate4_regression',baseline,float(gate4_reference[query_index,0,0]),'Fresh actual native baseline equals retained independent Gate 4 raw score.',mode='FRESH_REAL_INFERENCE',workflow='raw regression')
            rec.check('design_raw_capture_identity',float(np.load(out/'design-raw-00.npy')[0,0,0]),baseline,'Saved design raw tensor retains its own phase/sequence identity.',mode='FRESH_REAL_INFERENCE',workflow='raw regression')
            rec.check('design_objective_before',design['score_before'],baseline,'One on-target, no off-target: native target score.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            rec.check('design_objective_after',design['score_after'],expected_final,'max candidate accepted only if > baseline+1e-9; ties retain baseline.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            rec.check('design_candidate_sequence',design['designed_sequence'],expected_sequence,'Best candidate maps to its actual DNA; no outcome fabricated.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            rec.check('design_edit_count',design['n_mutations'],int(accepted),'Exactly one accepted mutation if the actual candidate improved.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            rec.check('design_history',[h['score'] for h in design['history']],[baseline,expected_final] if accepted else [baseline],'History contains actual accepted scores, never a fabricated optimization curve.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            backgrounds=[float(v[0,0]) for v in raw[-1]]
            expected_percent=sum(v<expected_final for v in backgrounds)/len(backgrounds)*100
            rec.check('design_shuffled_percentile',design['percentile_vs_shuffled'],expected_percent,'100*count(background objective < final objective)/n; descriptive, not probability.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            rec.check('design_unedited_bases',design['designed_sequence'][:1056]+design['designed_sequence'][1058:],sequence[:1056]+sequence[1058:],'Candidate search preserves bases outside the 1-based inclusive editable region.',mode='FRESH_REAL_INFERENCE',workflow='/design')
            rec.check('design_html_precision',format(design['score_after'],'.5g') in response.text,True,'Actual HTML final objective, five significant digits.',mode='HTML_FRESH_REAL',workflow='/design')
            save(out/'design-oracle.json',dict(raw_baseline=baseline,raw_candidates=candidates,best_index=best,accepted=accepted,
                expected_final=expected_final,expected_sequence=expected_sequence,raw_backgrounds=backgrounds,
                operations=['select task 0','pass through single output','maximize candidate score','strict improvement >1e-9','compare backgrounds'],warnings=result.get('warnings')))
        calls.clear();phase='attribution'
        # N-containing accepted Gate 4 DNA; actual gradient values, not an invented matrix.
        sequence=next(q['sequence'] for q in fixture['reference_queries'] if 'N' in q['sequence'])
        response,result=post('/attribution',dict(text=sequence,start='1054',end='1060',target='0'))
        rec.check('fresh_attribution_status',response.status_code,200,'Real full-input gradient calculation; no optimizer or training.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
        if response.status_code==200:
            observed=result['results'];matrix=np.array(observed['matrix'],dtype=np.float64)
            # Independent literal encoding and explicit mathematical channel correction.
            expected_input=np.zeros((1,4,len(sequence)),np.float32)
            for i,b in enumerate(sequence):
                if b in 'ACGT':expected_input[0,'ACGT'.index(b),i]=1
            x=torch.tensor(expected_input,requires_grad=True)
            native=adapter.module(x)
            score=native[:,[0],:].mean(dim=(1,2)).sum()
            np.save(out/'independent-attribution-native-output.npy',native.detach().numpy(),allow_pickle=False)
            gradient=torch.autograd.grad(score,x)[0].detach().numpy()[0].astype(np.float64)
            mean_channel=np.sum(gradient,axis=0)/4
            expected=(gradient-mean_channel[None,:])*expected_input[0]
            contribution=np.sum(expected,axis=0)
            np.save(out/'independent-gradient.npy',gradient,allow_pickle=False)
            np.save(out/'independent-corrected-input.npy',expected,allow_pickle=False)
            np.save(out/'actual-attribution.npy',matrix,allow_pickle=False)
            error=np.abs(matrix-expected)
            save(out/'attribution-oracle.json',dict(input_length=len(sequence),shape=list(matrix.shape),
                max_absolute_error=float(error.max()),formula='(native input gradient - mean over four channels) * literal one-hot input',
                output_score_operation='task0 native output mean (one position)',rtol=1e-5,atol=1e-7,
                mismatched_positions=np.argwhere(~np.isclose(matrix,expected,rtol=1e-5,atol=1e-7)).tolist()))
            rec.check('attribution_matrix',bool(np.allclose(matrix,expected,rtol=1e-5,atol=1e-7)),True,'Every channel and input position compared to independent native-gradient formula.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            rec.check('attribution_axes',observed['matrix_bases'],list('ACGT'),'Correct matrix row order A/C/G/T.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            rec.check('attribution_positions',[r['position'] for r in observed['positions']],list(range(1,len(sequence)+1)),'Input positions 1..2114, no pooling-to-input remapping.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            rec.check('attribution_reference',[r['reference'] for r in observed['positions']],list(sequence),'Every actual base aligns with its contribution.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            actual_contribution=np.array([r['contribution'] for r in observed['positions']])
            rec.check('attribution_signed_sum',bool(np.allclose(actual_contribution,contribution,rtol=1e-5,atol=1e-7)),True,'Sum four corrected input channels; preserve sign.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            expected_top=sorted([i for i,b in enumerate(sequence) if b!='N'],key=lambda i:-abs(contribution[i]))[:20]
            rec.check('attribution_top20',[r['position'] for r in observed['top_positions']],[i+1 for i in expected_top],'Stable top-20 absolute contribution; excludes unknown bases.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            rec.check('attribution_display_region',[r['position'] for r in observed['display_positions']],list(range(1054,1061)),'Requested one-based inclusive region.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            rec.check('attribution_N_zero',bool(np.all(matrix[:,[i for i,b in enumerate(sequence) if b=='N']]==0)),True,'N has zero input channels and zero input contribution.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
            rec.check('attribution_raw_reference',observed['reference_score'],float(native.detach().numpy()[0,0,0]),'Baseline score remains native prediction, not transformed probability.',mode='FRESH_REAL_GRADIENTS',workflow='/attribution')
    save(out/'safety.json',dict(checkpoint_sha256=fixture['identity']['checkpoint_sha256'],state_before=initial_state,state_after=state_hash(),
        training_executed=False,optimizer_steps=0,gradient_optimization_steps=0,
        attribution_input_gradient_computed=True,design_iterations=1,design_candidates=2,
        weight_updates=0,notes='Model reused as real preloaded adapter to bound memory; all predictions are actual forward executions.'))
    rec.check('weights_preserved',state_hash(),initial_state,'Real loaded state bytes unchanged across design and attribution.',mode='FRESH_REAL_INFERENCE',workflow='safety')
    save(out/'results.json',dict(checks=rec.rows,counts={s:sum(r['status']==s for r in rec.rows) for s in ['PASS','FAIL','SKIP','BLOCKED']},gate=5,gate6_started=False))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
