"""Training, motif analysis, and installed model management pages."""
import csv
import math
from pathlib import Path
import re
import secrets

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, send_file, session, url_for

from advanced_views import example_sequence, integer
from lab_compute import motif_analysis, train, training_example, training_rows
from lab_jobs import JobStateError, job_path, owner, read_job, submit_job, update_job
from model_registry import registrations, remove_registration, save_registration, sync_models
from core.model_lengths import required_length

lab = Blueprint('lab', __name__)


@lab.app_errorhandler(JobStateError)
def invalid_job_state(error):
    current_app.logger.exception('Invalid stored job state')
    return 'Stored job state is unavailable or invalid; no completed result is confirmed. Original data was retained.',500


def check_csrf():
    token = session.get('csrf_token', '')
    if not token or not secrets.compare_digest(token, request.form.get('csrf_token', '')):
        abort(400, description='The form expired. Reload and try again.')


def install_model(job_id, repo, revision=None, offline=False, filename="model.ckpt"):
    from huggingface_hub import HfApi, hf_hub_download
    from core.models import load_model
    from studio_web import inference_lock
    if offline:
        if not revision or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Provide the 40-character revision of an already cached checkpoint.')
        update_job(job_id, message='Reading the pinned checkpoint from the local cache.')
        path = hf_hub_download(repo, filename, revision=revision, local_files_only=True)
    else:
        update_job(job_id, message='Resolving the published checkpoint revision.')
        info = HfApi().model_info(repo, revision=revision or None, files_metadata=True)
        checkpoint = next((file for file in info.siblings if file.rfilename == filename), None)
        if checkpoint is None:
            raise ValueError(f'This repository does not publish {filename}.')
        if checkpoint.size and checkpoint.size > 2_000_000_000:
            raise ValueError('This local workspace accepts checkpoints up to 2 GB.')
        revision = info.sha
        update_job(job_id, message='Downloading checkpoint at pinned revision ' + revision[:12])
        path = hf_hub_download(repo, filename, revision=revision)
    update_job(job_id, message='Checking checkpoint metadata and input shape.')
    with inference_lock:
        adapter, _ = load_model({'source':'checkpoint','path':path}, 'cpu', batch_size=1)
    if adapter.seq_len > 200000:
        raise ValueError('This model exceeds the workspace input limit of 200,000 bases.')
    key = 'local-' + job_id[:12]
    save_registration(key, {'source':'checkpoint','path':path,'seq_len':adapter.seq_len}, repo, 'Hugging Face revision ' + revision)
    return {'model_key':key,'repository':repo,'revision':revision,'source':'local_cache' if offline else 'download','filename':filename,'input_length':adapter.seq_len,'tracks':len(adapter.task_names)}


@lab.route('/training', methods=['GET','POST'], defaults={'mode':'training'})
@lab.route('/fine-tuning', methods=['GET','POST'], defaults={'mode':'fine-tuning'})
@lab.route('/motifs', methods=['GET','POST'], defaults={'mode':'motifs'})
@lab.route('/model-zoo', methods=['GET','POST'], defaults={'mode':'model-zoo'})
def page(mode):
    from studio_web import MODELS, ZOO_MODELS, model_choices, parse_sequence, resolve_model
    from workbench import parse_batch
    sync_models()
    titles = {'training':'Model training','fine-tuning':'Fine-tuning','motifs':'Motif scanning & discovery','model-zoo':'Model Zoo'}
    values = {key:request.form.get(key, default) for key,default in {
        'text':'','patterns':'Motif_A:TATAAA\nMotif_B:GATA','k':'6','length':'200',
        'epochs':'5','learning_rate':'0.001','target':'0','model':'demo'}.items()}
    result, error, status = None, None, 200
    if request.method == 'POST':
        check_csrf()
        try:
            if mode == 'model-zoo':
                if request.form.get('action') == 'remove':
                    remove_registration(request.form.get('key',''))
                    sync_models()
                elif request.form.get('action') in ('install','register-cached'):
                    repo = request.form.get('repo','')
                    if repo not in ZOO_MODELS:
                        raise ValueError('Choose a model from the official catalog.')
                    revision = request.form.get('revision','').strip()
                    if revision and not re.fullmatch(r'[0-9a-f]{40}',revision):
                        raise ValueError('Revision must be a 40-character commit SHA, or blank for the latest published revision.')
                    offline = request.form.get('action') == 'register-cached'
                    if offline and not revision:
                        raise ValueError('A pinned revision is required for a cache-only installation.')
                    filename = {'Genentech/enformer-model':'human.ckpt','Genentech/borzoi-model':'human_rep0.ckpt'}.get(repo,'model.ckpt')
                    job_id = submit_job('Model installation', lambda job_id:install_model(job_id, repo, revision, offline, filename))
                    return redirect(url_for('lab.job', job_id=job_id))
            else:
                upload = request.files.get('file')
                if upload and upload.filename:
                    values['text'] = upload.read().decode('utf-8-sig')
                if mode == 'motifs':
                    if 'example' in request.form:
                        values['text'] = '>region_A\n' + example_sequence(80) + 'TATAAAGATATATAAA' + example_sequence(80) + '\n>region_B\n' + example_sequence(60) + 'TATAAATATAAA' + example_sequence(80)
                    records = parse_batch(values['text'], parse_sequence)
                    if sum(len(row['sequence']) for row in records) > 50000:
                        raise ValueError('Analyze at most 50,000 input bases per motif run.')
                    patterns = []
                    for index,line in enumerate(values['patterns'].splitlines(),1):
                        if not line.strip():
                            continue
                        name, _, sequence = line.strip().partition(':')
                        if not sequence:
                            sequence, name = name, f'motif_{index}'
                        sequence = sequence.upper()
                        if not re.fullmatch('[ACGT]{3,50}', sequence) or len(name) > 80:
                            raise ValueError('Use motif_name:ACGT with 3–50 known bases per motif.')
                        patterns.append((name,sequence))
                    if len(patterns)>32:
                        raise ValueError('Use at most 32 motif patterns.')
                    result = motif_analysis(records,patterns,integer(values['k'],'Discovery k-mer size',3,10))
                    result['inputs'] = records
                    from experiment_store import save_experiment
                    result['saved_run_id'] = save_experiment(titles[mode], mode, result, '/motifs',
                        {'text': values['text'], 'patterns': values['patterns'], 'k': values['k']})
                else:
                    if mode == 'training':
                        length = integer(values['length'],'Input length',50,2000)
                        spec = {'source':'demo','seq_len':length,'n_tasks':1,'seed':0}
                    else:
                        selected = resolve_model(values['model'])
                        if selected is None:
                            raise ValueError('Choose an installed model.')
                        spec = dict(MODELS[selected])
                        length = required_length(spec)
                        if length > 4096:
                            raise ValueError('CPU fine-tuning accepts model windows up to 4,096 bases.')
                    if 'example' in request.form:
                        values['text'] = training_example(length)
                    rows = training_rows(values['text'],length)
                    epochs = integer(values['epochs'],'Epochs',1,20)
                    rate = float(values['learning_rate'])
                    if not math.isfinite(rate) or not 0.00001 <= rate <= .01:
                        raise ValueError('Learning rate must be between 0.00001 and 0.01.')
                    target = integer(values['target'],'Target track index',0,100000)
                    data_source = 'synthetic GC-content regression example' if 'example' in request.form else 'uploaded numeric labels'
                    job_id = submit_job(titles[mode],lambda job_id:train(job_id,rows,spec,epochs,rate,target,data_source),
                        replay={'path': request.path, 'form': {**values}})
                    return redirect(url_for('lab.job',job_id=job_id))
        except (ValueError,UnicodeError,csv.Error) as exc:
            error,status = str(exc),400
    return render_template('lab.html',mode=mode,title=titles[mode],values=values,result=result,error=error,
        choices=model_choices(),selected=values['model'],catalog=ZOO_MODELS,installed=registrations()),status


def owned_job(job_id):
    try:
        state = read_job(job_id)
    except JobStateError:
        raise
    except ValueError:
        abort(404)
    if not state or state['owner'] != owner():
        abort(404)
    return state


@lab.get('/jobs/<job_id>')
def job(job_id):
    state = owned_job(job_id)
    return render_template('job.html',title=state['kind'],mode='training',job=state)


@lab.get('/api/jobs/<job_id>')
def job_status(job_id):
    state = dict(owned_job(job_id))
    state.pop('owner',None)
    response = jsonify(state)
    response.headers['Cache-Control'] = 'no-store'
    return response


@lab.get('/jobs/<job_id>/checkpoint')
def checkpoint(job_id):
    state = owned_job(job_id)
    if state['state'] != 'complete' or not state['result'].get('checkpoint'):
        abort(404)
    path=job_path(job_id)/'model.pt'
    if not path.is_file():abort(404)
    return send_file(path,as_attachment=True,download_name='grelu-trained-model.pt')
