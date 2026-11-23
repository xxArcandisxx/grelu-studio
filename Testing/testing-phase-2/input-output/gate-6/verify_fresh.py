"""One real pretrained request through storage and JSON download, no replay."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[3];sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('gate6_support',BASE/'verify_storage.py');support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)


def main(out):
    import torch
    torch.set_num_threads(1)
    app,fixture=support.configure(out)
    client=app.test_client();client.get('/experiments')
    case=next(c for c in fixture['cases'] if c['id']=='http_variant_1')
    raw=np.load(support.G4/'direct/all-batch1.npy',allow_pickle=False)
    lookup={q['sequence']:float(raw[i,0,0]) for i,q in enumerate(fixture['reference_queries'])}
    expected=dict(mode='FRESH_GM12878_REQUEST',identity=fixture['identity'],
        reference=lookup[case['body']['reference']],alternate=lookup[case['body']['alternate']],
        input_reference=case['body']['reference'],input_alternate=case['body']['alternate'])
    support.save(out/'expected-before-request.json',expected)
    rec=support.Recorder(out);rec.mode='FRESH_REAL_INFERENCE_STORAGE'
    response=client.post('/api/compare',json=case['body']);support.save(out/'actual-api.json',response.get_json())
    rec.check('fresh_request_status',response.status_code,200,'Unmodified real model loader/adapter and Flask handler.','fresh lifecycle')
    if response.status_code==200:
        payload=response.get_json();identifier=payload['saved_run_id']
        export=client.get('/experiments/'+identifier+'/record.json');(out/'actual-download.json').write_bytes(export.data)
        record=support.strict_json(export.text);task=fixture['identity']['tasks'][0]
        for i,name in enumerate(['reference','alternate']):
            rec.check('fresh_api_'+name,payload['results']['predictions'][i]['scores'][task],expected[name],'Independent retained direct-gReLU raw score, not read from stored data.','fresh lifecycle')
            rec.check('fresh_stored_'+name,record['payload']['results']['predictions'][i]['scores'][task],expected[name],'Exact stored/downloaded value after real inference.','fresh lifecycle')
        rec.check('fresh_checkpoint_identity',record['metadata']['models'][0]['checkpoint_sha256'],fixture['identity']['checkpoint_sha256'],'Actual model provenance persisted.','fresh lifecycle')
        rec.check('fresh_task_identity',record['metadata']['models'][0]['tasks'],[task],'Actual task identity persisted.','fresh lifecycle')
        rec.check('fresh_completed',record['metadata']['status'],'complete','Successful actual inference/archive result.','fresh lifecycle')
        rec.check('fresh_detail',client.get('/experiments/'+identifier).status_code,200,'Real request can reopen through history.','fresh lifecycle')
        with client:
            client.get('/experiments')
            from experiment_store import experiment
            rec.check('fresh_new_connection',experiment(identifier)['payload'],record['payload'],'Storage helper opens/closes a fresh SQLite connection for each read.','fresh lifecycle')
    support.save(out/'results.json',dict(checks=rec.rows,counts={s:sum(r['status']==s for r in rec.rows) for s in ['PASS','FAIL','SKIP','BLOCKED']},
        actual_model_inference_executed=True,replay=False,optimizer_steps=0,training_executed=False))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
