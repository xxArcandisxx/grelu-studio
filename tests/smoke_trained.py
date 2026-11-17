"""Optional offline smoke test using the pinned, already-cached human checkpoint."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fine-tuning',action='store_true')
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    checkpoint = hf_hub_download('Genentech/GM12878_dnase-model', 'model.ckpt',
        revision='9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e', local_files_only=True)
    os.environ.update(GRELU_HUMAN_CHECKPOINT=checkpoint, GRELU_HUMAN_INPUT_LENGTH='2114',
                      GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase')
    from advanced_views import example_sequence
    from studio_web import app
    from test_advanced import record

    with tempfile.TemporaryDirectory() as directory:
        app.config.update(LAB_ROOT=str(Path(directory)/'lab'), HISTORY_DATABASE=str(Path(directory) / 'history.db'), TESTING=True, ACCOUNT_DATABASE=str(Path(directory) / 'accounts.db'))
        with app.test_client() as client:
            started = time.monotonic()
            metadata = client.get('/api/model-info?model=human')
            assert metadata.status_code == 200, metadata.text
            assert metadata.json['seq_len'] == 2114
            print(f'Model metadata: PASS ({time.monotonic() - started:.2f}s)', flush=True)
            cases = [
                ('design', {'example': '1', 'iterations': '1', 'candidates': '2'}),
                ('batch-variants', {'example': '1'}),
                ('mutagenesis', {'text': example_sequence(2114), 'start': '1058', 'end': '1059'}),
                ('attribution', {'text': example_sequence(2114), 'start': '1008', 'end': '1107', 'randomized_check': 'on'}),
            ]
            for mode, values in cases:
                started = time.monotonic()
                response = client.post('/' + mode, data={'model': 'human', **values})
                assert response.status_code == 200, response.text
                result = record(response)
                assert result['ok'] and not result['demo']
                assert result['model']['provenance']['checkpoint_sha256']
                if mode == 'mutagenesis':
                    assert result['results']['substitutions'] == 6
                if mode == 'attribution':
                    assert len(result['results']['positions']) == 2114
                    assert result['results']['randomized_check']
                print(f'{mode}: PASS ({time.monotonic() - started:.2f}s)', flush=True)
            client.get('/experiments')
            with client.session_transaction() as session:
                csrf = session['csrf_token']
            for path, values in [
                ('/motif-simulation', {'model':'human','example':'1','positions':'1058','orientation':'+','controls':'on'}),
                ('/model-comparison', {'model_a':'demo','model_b':'human','policy':'center','example':'1'}),
            ]:
                started=time.monotonic()
                response=client.post(path,data={'csrf_token':csrf,**values})
                assert response.status_code==200,response.text
                saved=record(response)['saved_run_id']
                assert client.get('/experiments/'+saved+'/report.html').status_code==200
                print(f'{path}: PASS ({time.monotonic()-started:.2f}s)',flush=True)
            if args.fine_tuning:
                from lab_compute import training_example
                from lab_jobs import workers
                text='\n'.join(training_example(2114).splitlines()[:9])
                started=time.monotonic()
                response=client.post('/fine-tuning',data={'csrf_token':csrf,'model':'human','text':text,
                    'epochs':'1','learning_rate':'0.0001','target':'0'})
                assert response.status_code==302,response.text
                job_id=response.location.split('/')[-1]
                worker=workers.get(job_id)
                if worker:
                    worker.join(timeout=55)
                state=client.get('/api/jobs/'+job_id).json
                assert state['state']=='complete',state
                assert len(state['history'])==1 and state['result']['optimizer_steps']>0,state
                key=state['result']['model_key']
                predicted=client.post('/predict',data={'model':key,'fasta':example_sequence(2114)})
                assert predicted.status_code==200,predicted.text
                assert client.get('/experiments/'+state['result']['saved_run_id']+'/report.html').status_code==200
                print(f'Pretrained fine-tuning and checkpoint reload: PASS ({time.monotonic()-started:.2f}s)',flush=True)



if __name__ == '__main__':
    main()
