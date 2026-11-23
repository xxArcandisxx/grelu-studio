"""Real Studio loaders, adapters and Flask requests with transparent observers."""
import argparse
import ctypes
import gc
from contextlib import ExitStack
import hashlib
import inspect
import io
import json
import os
import re
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


class StopSubset(BaseException):
    """Stop after a documented subset has produced real numerical outputs."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    fixture = json.loads((run / 'fixtures.json').read_text())
    os.environ.update(GRELU_HUMAN_CHECKPOINT=fixture['checkpoint_path'], GRELU_HUMAN_INPUT_LENGTH='2114',
                      GRELU_HUMAN_MODEL_ID='Genentech GM12878 DNase', GRELU_SECRET_KEY='gate4-isolated')
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from grelu.lightning import LightningModel
    from grelu.data.augment import Augmenter
    from core.models import ModelAdapter, load_model
    import core.models as models
    import workflows.predict as predict_workflow
    from studio_web import app, MODELS, PROVENANCE
    state = run / 'state'
    state.mkdir()
    app.config.update(TESTING=True, LAB_ROOT=str(state / 'lab'),
        ACCOUNT_DATABASE=str(state / 'accounts.db'), HISTORY_DATABASE=str(state / 'history.db'))
    client = app.test_client()
    client.get('/experiments')
    with client.session_transaction() as session:
        csrf = session['csrf_token']
    out = run / 'website'
    out.mkdir()
    current = {}
    all_cases = []
    augmentation = []
    original_forward = LightningModel.forward
    original_predict = ModelAdapter.predict
    original_norm = models._norm3
    original_reader = predict_workflow.read_sequences
    original_augment = Augmenter.__call__
    forwards = 0

    def state_hash(module):
        h = hashlib.sha256()
        for name, tensor in sorted(module.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def observed_forward(module, x, *args, **kwargs):
        nonlocal forwards
        if current.get('case', {}).get('fault') and (not current.get('fault_fired') or current['case']['fault'] == 'persistent'):
            current['fault_fired'] = True
            current.setdefault('faults', []).append('Gate 4 explicitly injected model RuntimeError before weighted forward')
            raise RuntimeError('Gate4 explicit model fault injection')
        result = original_forward(module, x, *args, **kwargs)
        forwards += 1
        call = current.get('call')
        if call is not None:
            index = len(call['forwards'])
            folder = out / current['case']['id'] / f'call-{call["index"]:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            name = f'forward-{index:02d}.npy'
            np.save(folder / name, result.detach().cpu().numpy().copy(), allow_pickle=False)
            tx = module.format_input(x)
            record = dict(file=str((folder / name).relative_to(out)),
                shape=list(result.shape), dtype=str(result.dtype), device=str(result.device),
                input_shape=list(tx.shape), input_dtype=str(tx.dtype), input_device=str(tx.device),
                input_sha256=hashlib.sha256(tx.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                training_modules=[n for n, m in module.named_modules() if m.training],
                dropout_modules={n: m.training for n, m in module.named_modules() if isinstance(m, torch.nn.Dropout)},
                grad_enabled=torch.is_grad_enabled(), inference_mode=torch.is_inference_mode_enabled(),
                output_requires_grad=result.requires_grad,
                activation=type(module.activation).__name__, transform=type(module.transform).__name__)
            call['forwards'].append(record)
        return result

    def observed_norm(y, n):
        call = current.get('call')
        result = original_norm(y, n)
        if call is not None:
            folder = out / current['case']['id'] / f'call-{call["index"]:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            index = len(call['normalizations'])
            raw_name = f'pre-norm-{index:02d}.npy'
            norm_name = f'normalized-{index:02d}.npy'
            np.save(folder / raw_name, np.asarray(y), allow_pickle=False)
            np.save(folder / norm_name, result, allow_pickle=False)
            call['normalizations'].append(dict(raw=str((folder / raw_name).relative_to(out)),
                normalized=str((folder / norm_name).relative_to(out)), n=n))
        return result

    def observed_predict(adapter, seqs):
        case = current['case']
        call = dict(index=len(current['calls']), input_sequences=list(seqs),
            backend=adapter.backend, native_enabled_before=adapter._use_grelu_predict,
            model_meta=adapter.meta.copy(), device=adapter.device, batch_size=adapter.batch_size,
            tasks=list(adapter.task_names), sequence_length=adapter.seq_len,
            state_sha256_before=state_hash(adapter.module), forwards=[], normalizations=[],
            caller=inspect.currentframe().f_back.f_code.co_name,
            caller_file=inspect.currentframe().f_back.f_code.co_filename)
        current['calls'].append(call)
        current['call'] = call
        try:
            result = original_predict(adapter, seqs)
            folder = out / case['id'] / f'call-{call["index"]:02d}'
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / 'adapter-return.npy', result, allow_pickle=False)
            call.update(result=str((folder / 'adapter-return.npy').relative_to(out)),
                shape=list(result.shape), dtype=str(result.dtype), notes=list(adapter.notes),
                native_enabled_after=adapter._use_grelu_predict,
                state_sha256_after=state_hash(adapter.module))
        except Exception:
            call['exception'] = traceback.format_exc()
            raise
        finally:
            current.pop('call', None)
        stop = case.get('stop_after_calls')
        if stop is not None and len(current['calls']) >= stop:
            raise StopSubset()
        return result

    def observed_reader(params):
        answer = original_reader(params)
        current.setdefault('reader_records', []).append(answer[0])
        return answer

    def observed_augment(*args, **kwargs):
        augmentation.append(current.get('case', {}).get('id'))
        return original_augment(*args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(LightningModel, 'forward', observed_forward))
        stack.enter_context(patch.object(ModelAdapter, 'predict', observed_predict))
        stack.enter_context(patch.object(models, '_norm3', observed_norm))
        stack.enter_context(patch.object(predict_workflow, 'read_sequences', observed_reader))
        stack.enter_context(patch.object(Augmenter, '__call__', observed_augment))
        # Real loader is never replaced. No prediction method fabricates values.
        for case in fixture['cases']:
            current.clear()
            current.update(case=case, calls=[], reader_records=[])
            print('WEBSITE', case['id'], flush=True)
            observation = dict(id=case['id'], workflow=case['workflow'])
            try:
                if case['kind'] == 'adapter':
                    adapter, warnings = load_model(MODELS['human'], 'cpu', case['batch_size'], backend=case['backend'])
                    adapter.predict([r['sequence'] for r in case['records']])
                    observation['loader_warnings'] = warnings
                    del adapter
                elif case['kind'] == 'engine_design':
                    from engine import execute_platform_workflow
                    outcome = execute_platform_workflow('design', dict(model=MODELS['human'], device='cpu',
                        seeds=[r['sequence'] for r in case['records']], max_iter=0, n_baseline=0,
                        batch_size=1, on_targets=[0], output_dir=str(out / case['id'] / 'engine-output')))
                    observation['engine_outcome'] = outcome
                    if not outcome['ok']:
                        observation['error'] = outcome.get('traceback', outcome.get('error'))
                elif case['kind'] == 'json':
                    response = client.post(case['url'], json=case['body'])
                    observation.update(status=response.status_code, response_json=response.get_json(silent=True))
                    (out / (case['id'] + '.response.json')).write_text(response.text)
                else:
                    data = {'csrf_token': csrf, **case.get('fields', {})}
                    if case.get('upload'):
                        data['file'] = (io.BytesIO(case['upload'].encode()), 'gate4-input.fasta')
                    response = client.post(case['url'], data=data,
                        content_type='multipart/form-data' if case.get('upload') else None)
                    observation['status'] = response.status_code
                    (out / (case['id'] + '.response.html')).write_text(response.text)
                    match = re.search(r'<script[^>]*id="run-record"[^>]*>(.*?)</script>', response.text, re.S)
                    if match:
                        observation['response_record'] = json.loads(match[1])
            except StopSubset:
                observation['deliberately_stopped_after_real_outputs'] = True
            except Exception:
                observation['error'] = traceback.format_exc()
            observation.update(calls=current['calls'], reader_records=current['reader_records'],
                faults=current.get('faults', []))
            save(out / (case['id'] + '.json'), observation)
            all_cases.append(observation)
            print('DONE', case['id'], observation.get('status'), len(current['calls']), flush=True)
            # The harness reloads checkpoints across independent cases. Release
            # unreachable models and free CPU allocator pages between cases.
            # No tensor, settings or numeric prediction is changed.
            gc.collect()
            ctypes.CDLL('libc.so.6').malloc_trim(0)
    save(out / 'manifest.json', dict(cases=all_cases, augmentation_calls=augmentation,
        numerical_forward_count=forwards, selected_spec=MODELS['human'], provenance=PROVENANCE['human'],
        observers='Original loader, forward, adapter, normalization and reader are executed; observers save copies and return values unchanged. Explicit fault/subset cases are labeled separately.',
        torch_threads=torch.get_num_threads(), checkpoint_path=fixture['checkpoint_path'],
        optimizer_steps=0, training_executed=False, gradients_computed=False,
        between_case_memory_cleanup='gc.collect and glibc malloc_trim(0); no inference settings change'))


if __name__ == '__main__':
    main()
