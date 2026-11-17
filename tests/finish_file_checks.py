"""Finish model checks one fresh process at a time, using only file fixtures."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['fine-tuning','model-zoo','persist-create','persist-verify'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'test-inputs/website-suite/manifest.json').read_text())
    settings=json.loads((ROOT/manifest['settings']).read_text())
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        GRELU_TRAIN_BATCH_SIZE='1',GRELU_SECRET_KEY='isolated-low-memory-test-key',HF_HUB_OFFLINE='1')
    if args.case in ('fine-tuning','model-zoo'):
        from huggingface_hub import hf_hub_download
        source=settings['model_zoo']
        checkpoint=hf_hub_download(source['repo'],source['filename'],revision=source['revision'],local_files_only=True)
        os.environ.update(GRELU_HUMAN_CHECKPOINT=checkpoint,GRELU_HUMAN_INPUT_LENGTH='2114',GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from studio_web import app
    started=time.monotonic()
    persistent=output/'persistence'
    with tempfile.TemporaryDirectory(prefix='grelu-light-') as temporary:
        state=persistent if args.case.startswith('persist-') else Path(temporary)
        state.mkdir(parents=True,exist_ok=True)
        app.config.update(TESTING=True,TRAIN_BATCH_SIZE=1,ACCOUNT_LOCAL_PREVIEW=True,
            ACCOUNT_DATABASE=str(state/'accounts.db'),HISTORY_DATABASE=str(state/'history.db'),LAB_ROOT=str(state/'lab'))
        with app.test_client() as client:
            client.get('/experiments')
            with client.session_transaction() as session:csrf=session['csrf_token']
            def post(path,fields=None,files=None):
                data={'csrf_token':csrf,**(fields or {})}
                for name,fixture_path in (files or {}).items():data[name]=(io.BytesIO((ROOT/fixture_path).read_bytes()),Path(fixture_path).name)
                return client.post(path,data=data,content_type='multipart/form-data')
            def await_job(response):
                assert response.status_code==302,response.text[:500]
                job_id=response.location.split('/')[-1]
                deadline=time.monotonic()+180
                while time.monotonic()<deadline:
                    job=client.get('/api/jobs/'+job_id).json
                    if job['state'] in ('complete','failed'):break
                    time.sleep(.1)
                assert job['state']=='complete',job
                return job
            if args.case=='fine-tuning':
                case=next(case for case in manifest['cases'] if case['id']=='fine-tuning-human')
                result=await_job(post(case['route'],{'model':'human',**settings['fine_tuning']},case['files']))
                assert result['result']['batch_size']==1
                assert result['result']['optimizer_steps']==6
                assert len(result['history'])==1
                checkpoint_response=client.get('/jobs/'+result['id']+'/checkpoint')
                assert checkpoint_response.status_code==200
                with (output/'fine-tuned-checkpoint.pt').open('wb') as stream:
                    for chunk in checkpoint_response.response:stream.write(chunk)
                checkpoint_response.close()
                response=post('/predict',{'model':result['result']['model_key']},
                    {'file':'test-inputs/trained-model-2114bp/06_Sequence_Prediction_three_records.fasta'})
                assert response.status_code==200,response.text[:500]
                from test_advanced import record
                prediction=record(response)
                assert len(prediction['results']['predictions'])==3
                assert client.get('/experiments/'+result['result']['saved_run_id']+'/report.html').status_code==200
                details={'id':'fine-tuning-human','mode':'sequential Flask HTTP integration, no browser',
                    'fixture':case['files']['file'],'optimizer_steps':result['result']['optimizer_steps'],
                    'batch_size':1,'checkpoint_reloaded_for_prediction':True,'metrics':result['result']}
            elif args.case=='model-zoo':
                source=settings['model_zoo']
                result=await_job(post('/model-zoo',{'action':'register-cached','repo':source['repo'],'revision':source['revision']}))
                assert result['result']['source']=='local_cache'
                key=result['result']['model_key']
                inspected=client.get('/api/model-info?model='+key)
                assert inspected.status_code==200 and inspected.json['seq_len']==2114
                prediction=post('/predict',{'model':key},{'file':'test-inputs/trained-model-2114bp/06_Sequence_Prediction_three_records.fasta'})
                assert prediction.status_code==200
                assert post('/model-zoo',{'action':'remove','key':key}).status_code==200
                assert client.get('/api/model-info?model='+key).status_code==400
                details={'id':'model-zoo-cache','mode':'sequential Flask HTTP integration, real cached weights',
                    'fixture':manifest['settings'],'registration':True,'inspection':True,'prediction':True,'removal':True,'remote_downloads_tested':False}
            elif args.case=='persist-create':
                response=post('/pwm-scan',settings['pwm'],{'sequence-file':'test-inputs/new-workflows/pwm-regions.txt',
                    'matrix-file':'test-inputs/new-workflows/pwm-motifs.jaspar'})
                assert response.status_code==200
                from test_advanced import record
                run_id=record(response)['saved_run_id']
                (state/'continuation.json').write_text(json.dumps({'run_id':run_id,'cookie':client.get_cookie('session').value}))
                details={'id':'history-persistence-create','fixture':'test-inputs/new-workflows/pwm-motifs.jaspar','run_id':run_id}
            else:
                continuation=json.loads((state/'continuation.json').read_text())
                client.set_cookie('session',continuation['cookie'])
                run_id=continuation['run_id']
                assert client.get('/experiments/'+run_id).status_code==200
                report=client.get('/experiments/'+run_id+'/report.html')
                assert report.status_code==200 and 'PWM motif scanning' in report.text
                (output/'persisted-report.html').write_bytes(report.data)
                details={'id':'history-persistence-verify','mode':'new Python process reads the prior process SQLite snapshot',
                    'persisted_snapshot_reopened':True,'report_exported':True}
    details.update(status='passed',seconds=round(time.monotonic()-started,2))
    (output/(args.case+'-result.json')).write_text(json.dumps(details,indent=2)+'\n')
    print(json.dumps({key:value for key,value in details.items() if key!='metrics'}),flush=True)


if __name__=='__main__':main()
