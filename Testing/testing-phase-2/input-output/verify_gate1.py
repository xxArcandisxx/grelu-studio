"""Gate 1 only: actual checkpoint/module identity, not inference correctness.

Run from repository root:
  .venv/bin/python Testing/testing-phase-2/input-output/verify_gate1.py
Sequential CPU workers use real weights; no DNA, predictions, or training.
Returns 0=PASS, 1=FAIL, 2=BLOCKED. Evidence is retained for every run.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT))
FIXTURE = BASE / 'fixtures/model_identity.json'
PHASE1 = ['Testing/testing-phase-1', 'test-results/gm12878-biological', 'test-results/gm12878-verified']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def snapshot():
    return {str(path.relative_to(ROOT)): sha(path) for directory in PHASE1
            for path in sorted((ROOT / directory).rglob('*')) if path.is_file()}


def allow_numpy_metadata():
    # Serialization compatibility only. Neither branch uses website DNA encoding.
    import numpy as np
    return [np.dtype, np.ndarray, (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct')] + [
        getattr(np.dtypes, key) for key in ['BoolDType', 'Float32DType', 'Float64DType',
        'Int32DType', 'Int64DType', 'ObjectDType', 'StrDType']]


def data_metadata(data):
    keys = ['seq_len', 'genome', 'label_len', 'bin_size', 'label_aggfunc',
            'label_transform_func', 'rc', 'max_seq_shift', 'max_pair_shift', 'n_seqs']
    return {'tasks': data.get('tasks'), **{split: {key: value for key, value in data.get(split, {}).items()
             if key in keys} for split in ['train', 'val', 'test']}}


def tensors(state):
    import torch
    output = {}
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        output[name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'numel': tensor.numel(),
                        'finite': bool(torch.isfinite(tensor).all()),
                        'sha256': hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
    return output


def module_identity(module):
    return {'lightning_class': f'{type(module).__module__}.{type(module).__qualname__}',
            'architecture_class': f'{type(module.model).__module__}.{type(module.model).__qualname__}',
            'model_params': module.model_params, 'data_metadata': data_metadata(module.data_params),
            'parameter_count': sum(p.numel() for p in module.parameters()),
            'training': module.training,
            'training_submodules': [name for name, child in module.named_modules() if child.training],
            'parameter_devices': sorted({str(p.device) for p in module.parameters()}),
            'parameter_dtypes': sorted({str(p.dtype) for p in module.parameters()}),
            'activation_class': type(module.activation).__name__, 'prediction_transform_class': type(module.transform).__name__,
            'state_tensors': tensors(module.state_dict())}


class EmbeddedRecord(HTMLParser):
    def __init__(self):
        super().__init__(); self.active = False; self.parts = []; self.cells = []; self.cell = None
    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'run-record': self.active = True
        if tag == 'td': self.cell = ''
    def handle_endtag(self, tag):
        if tag == 'script': self.active = False
        if tag == 'td' and self.cell is not None: self.cells.append(self.cell); self.cell = None
    def handle_data(self, text):
        if self.active: self.parts.append(text)
        if self.cell is not None: self.cell += text


def worker(role, path, out, expected):
    import torch
    torch.set_num_threads(1)
    if role == 'regressions':
        import importlib.util
        spec = importlib.util.spec_from_file_location('phase2_length_regressions', BASE/'test_length_contract.py')
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
        return tests.run_recorded()
    if role == 'raw':
        with torch.serialization.safe_globals(allow_numpy_metadata()):
            payload = torch.load(path, map_location='cpu', weights_only=True)
        return {'checkpoint_sha256': sha(path), 'metadata': {
                'model_params': payload['hyper_parameters']['model_params'],
                'data_metadata': data_metadata(payload['data_params']),
                'training_task': payload['hyper_parameters']['train_params'].get('task'),
                'training_loss': payload['hyper_parameters']['train_params'].get('loss'),
                'training_grelu_version': payload['hyper_parameters']['train_params'].get('grelu_version'),
                'checkpoint_lightning_version': payload.get('pytorch-lightning_version')},
                'state_tensors': tensors(payload['state_dict'])}
    if role == 'direct':
        from grelu.lightning import LightningModel
        with torch.serialization.safe_globals(allow_numpy_metadata()):
            module = LightningModel.load_from_checkpoint(str(path), map_location='cpu', weights_only=True)
        module.eval()
        return {'loader': 'grelu.lightning.LightningModel.load_from_checkpoint (direct)',
                'website_model_module_imported': 'core.models' in sys.modules,
                'website_sequence_module_imported': 'core.sequences' in sys.modules,
                'identity': module_identity(module), 'checkpoint_sha256_after': sha(path),
                'dna_inputs': None, 'raw_prediction_shape': None,
                'inference_executed': False, 'reason': 'Model Identity only; independent numerical inference is deferred.'}
    # A fresh website instance, configured exactly as the pinned launcher's GM12878 branch.
    # No existing account/history/model registry is used.
    os.environ.update(GRELU_HUMAN_CHECKPOINT=str(path), GRELU_HUMAN_INPUT_LENGTH=str(expected['seq_len']),
                      GRELU_HUMAN_MODEL_ID=expected['website_label'], GRELU_SECRET_KEY='isolated-phase2-gate1')
    import core.models as website_models
    original_loader = website_models.load_model
    observations = []
    def observe_real_loader(*args, **kwargs):
        adapter, warnings = original_loader(*args, **kwargs)
        observations.append({'spec': dict(args[0]), 'identity': module_identity(adapter.module),
                             'adapter': {'seq_len': adapter.seq_len, 'tasks': adapter.task_names,
                                         'batch_size': adapter.batch_size, 'backend': adapter.backend,
                                         'device': adapter.device, 'notes': list(adapter.notes)},
                             'warnings': warnings})
        return adapter, warnings
    # Instrumentation records the real loader's return value. No model, metadata,
    # forward method, prediction method, or return value is substituted.
    website_models.load_model = observe_real_loader
    from studio_web import app, MODELS, PROVENANCE
    state = out / 'isolated-website-state'
    state.mkdir(exist_ok=True)
    app.config.update(TESTING=True, LAB_ROOT=str(state/'lab'),
                      ACCOUNT_DATABASE=str(state/'accounts.db'), HISTORY_DATABASE=str(state/'history.db'))
    client = app.test_client()
    before = {'spec': dict(MODELS['human']), 'provenance': dict(PROVENANCE['human']),
              'demo_spec': dict(MODELS['demo'])}
    inspector = client.post('/models', data={'model': 'human'})
    (out/'model-inspector.html').write_bytes(inspector.data)
    parser = EmbeddedRecord(); parser.feed(inspector.text)
    inspector_record = json.loads(''.join(parser.parts)) if parser.parts else None
    api = client.get('/api/model-info?model=human')
    valid_observations = list(observations)
    unknown = {}
    for name in ['phase2-model-that-is-not-installed', expected['repo_id']]:
        response = client.get('/api/model-info', query_string={'model': name})
        unknown[name] = {'status': response.status_code, 'body': response.get_json()}
    # Deliberately conflicting declaration, confined to this test process.
    # This tests identity/configuration consistency; it sends no DNA to the model.
    MODELS['human'] = {**MODELS['human'], 'seq_len': expected['negative_declared_length']}
    try:
        bad = client.get('/api/model-info?model=human')
        bad_inspector = client.post('/models', data={'model': 'human'})
        (out/'incompatible-model-inspector.html').write_bytes(bad_inspector.data)
        mismatch = {'configured_length': expected['negative_declared_length'],
                    'native_checkpoint_length': valid_observations[0]['identity']['data_metadata']['train']['seq_len'],
                    'status': bad.status_code, 'body': bad.get_json(),
                    'inspector_status': bad_inspector.status_code,
                    'inspector_error_visible': 'Input length mismatch' in bad_inspector.text,
                    'invalid_200_window_advertised': 'requires 200 bases per record' in bad_inspector.text,
                    'loader_observations': observations[len(valid_observations):]}
    finally:
        MODELS['human'] = before['spec']
    # Real small gReLU checkpoint fixture, untrained and explicitly synthetic.
    # Used solely for metadata/display integration, never prediction evidence.
    from grelu.lightning import LightningModel
    torch.manual_seed(0)  # Only the following untrained metadata fixture is initialized.
    fixture_params = {'model_type': 'ConvModel', 'n_tasks': 1, 'stem_channels': 4,
                      'n_conv': 1, 'channel_init': 4}
    fixture_train = {'task': 'regression', 'loss': 'mse', 'logger': False}
    fixture_module = LightningModel(fixture_params, fixture_train)
    fixture_data = {'tasks': {'name': ['metadata_fixture']}, 'train': {'seq_len': 128}}
    payload = {'state_dict': fixture_module.state_dict(), 'hyper_parameters': {
        'model_params': fixture_params, 'train_params': fixture_train}, 'data_params': fixture_data,
        'performance': {}, 'pytorch-lightning_version': version('pytorch-lightning')}
    fixture_path = out/'fixtures/other-model-128.ckpt'
    fixture_path.parent.mkdir(exist_ok=True)
    torch.save(payload, fixture_path)
    unknown_path = out/'fixtures/missing-length-metadata.ckpt'
    torch.save({**payload, 'data_params': {'tasks': {'name': ['metadata_fixture']}}}, unknown_path)
    cases = {}
    for key, checkpoint_path in [('phase2_other_128', fixture_path), ('phase2_unknown', unknown_path)]:
        MODELS[key] = {'source': 'checkpoint', 'path': str(checkpoint_path), 'seq_len': 128}
        PROVENANCE[key] = {'model_id': 'Untrained metadata integration fixture', 'checkpoint_sha256': sha(checkpoint_path)}
        response = client.get('/api/model-info', query_string={'model': key})
        page = client.post('/models', data={'model': key})
        (out/f'{key}-inspector.html').write_bytes(page.data)
        fixture_parser = EmbeddedRecord(); fixture_parser.feed(page.text)
        cases[key] = {'api_status': response.status_code, 'api_body': response.get_json(),
                      'inspector_status': page.status_code, 'table_cells': fixture_parser.cells,
                      'unknown_notice_visible': 'Input length unknown' in page.text,
                      'configured_128_advertised': 'requires 128 bases per record' in page.text,
                      'has_prediction_record': bool(fixture_parser.parts),
                      'checkpoint_sha256': sha(checkpoint_path)}
    # Prove native metadata is used if the website omits a declaration entirely.
    original_spec = MODELS['human']
    MODELS['human'] = {key: value for key, value in original_spec.items() if key != 'seq_len'}
    try:
        inferred = client.get('/api/model-info?model=human')
        missing_declaration = {'status': inferred.status_code, 'body': inferred.get_json()}
    finally:
        MODELS['human'] = original_spec
    return {'loader_instrumentation': 'Records real returned modules; no mocked predictions or metadata.',
            'configuration': before, 'inspector_status': inspector.status_code,
            'inspector_record': inspector_record, 'rendered_table_cells': parser.cells,
            'api': {'status': api.status_code, 'body': api.get_json()},
            'valid_loader_observations': valid_observations, 'unavailable_selections': unknown,
            'contradictory_length': mismatch, 'checkpoint_sha256_after': sha(path),
            'length_integration_cases': cases, 'native_length_without_website_declaration': missing_declaration,
            'inference_executed': False, 'raw_prediction_shape': None,
            'website_scope': 'Isolated Flask requests, not a running server or browser session.'}


def run_worker(role, path, out, env):
    import psutil
    log = out / f'{role}.log'
    target = out / f'{role}.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', role,
               '--checkpoint', str(path), '--worker-output', str(out)]
    started, peak, reason = time.monotonic(), 0, None
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        tracked = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = tracked.memory_info().rss + sum(c.memory_info().rss for c in tracked.children(recursive=True))
                peak = max(peak, rss)
                if rss > 1_800_000_000: reason = 'Owned worker exceeded 1.8 GB RSS limit.'
                if psutil.virtual_memory().available < 650_000_000: reason = 'Available RAM below 650 MB reserve.'
                if time.monotonic() - started > 180: reason = 'Owned worker exceeded 180-second timeout.'
            except psutil.NoSuchProcess:
                pass
            if reason:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
                break
            time.sleep(.2)
    evidence = json.loads(target.read_text()) if target.exists() else None
    return {'exit_code': process.returncode, 'elapsed_seconds': time.monotonic()-started,
            'peak_rss_bytes': peak, 'resource_block': reason, 'evidence_file': target.name,
            'log_file': log.name, 'data': evidence}


def main(args):
    expected = json.loads(FIXTURE.read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = BASE / 'runs' / stamp
    out.mkdir(parents=True)
    def historical_runs():
        return {str(p.relative_to(BASE)): sha(p) for p in sorted((BASE/'runs').rglob('*'))
                if p.is_file() and out not in p.parents}
    history_before = historical_runs()
    save(out/'historical-runs-before.json', history_before)
    checks, workers = [], {}
    before = snapshot()
    save(out/'phase1-before.json', before)
    save(out/'expected-identity.json', expected)
    def check(name, condition, detail, evidence, required=True, status=None):
        entry = {'id': name, 'status': status or ('PASS' if condition else 'FAIL'),
                 'required': required, 'explanation': detail, 'evidence': evidence}
        checks.append(entry)
        print(entry['status'], name, detail, flush=True)
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1', 'MPLBACKEND': 'Agg',
           'MPLCONFIGDIR': str(out/'matplotlib-cache')}
    versions = {'python': platform.python_version(), 'platform': platform.platform(), 'byteorder': sys.byteorder}
    for package in ['torch','grelu','numpy','pandas','pytorch-lightning','flask','huggingface-hub','psutil']:
        try: versions[package] = version(package)
        except PackageNotFoundError: versions[package] = 'NOT INSTALLED'
    save(out/'environment.json', {'versions': versions, 'settings': {
        'device': 'cpu', 'threads': 1, 'workers': 'sequential', 'download_allowed': False,
        'training_allowed': False, 'dna_inference_executed': False, 'input_shape_observed': None,
        'output_shape_observed': None, 'shapes_reason': 'No DNA inference belongs to Gate 1.',
        'rss_ceiling_bytes': 1800000000, 'available_ram_reserve_bytes': 650000000},
        'ambient_configuration': {key: os.environ.get(key) for key in [
            'GRELU_HUMAN_CHECKPOINT','GRELU_HUMAN_INPUT_LENGTH','GRELU_HUMAN_MODEL_ID']}})
    try:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(expected['repo_id'], expected['filename'],
                                   revision=expected['revision'], local_files_only=True))
        checkpoint_before = sha(path)
        save(out/'checkpoint.json', {'repo_id': expected['repo_id'], 'revision': expected['revision'],
             'cache_path': str(path), 'resolved_path': str(path.resolve()), 'bytes': path.stat().st_size,
             'sha256': checkpoint_before, 'expected_sha256': expected['checkpoint_sha256']})
        check('checkpoint_hash', checkpoint_before == expected['checkpoint_sha256'],
              'Cached file content must match the pinned identity anchor.', ['checkpoint.json','expected-identity.json'])
        launcher = (ROOT/'run_studio.sh').read_text()
        check('launcher_pin', all(s in launcher for s in [expected['repo_id'],expected['revision'],expected['filename'],'local_files_only=True']),
              'Launcher statically references this repository, revision, filename and cache-only lookup.', ['source-inventory.json'])
        if checkpoint_before != expected['checkpoint_sha256']:
            raise ValueError('Checkpoint identity mismatch: refusing all deserialization.')
        for role in ['raw','direct','website','regressions']:
            print(f'Running real {role} identity worker (no inference)...', flush=True)
            workers[role] = run_worker(role, path, out, env)
            result = workers[role]
            state = 'SKIP' if result['resource_block'] else ('PASS' if result['exit_code']==0 and result['data'] and result['data'].get('ok') else 'FAIL')
            check(f'{role}_worker', state == 'PASS', result['resource_block'] or 'Worker must finish and save evidence; inspect its log for any error.',
                  [result['evidence_file'],result['log_file']], status=state)
            if state != 'PASS':
                continue
        def evidence(role):
            return workers.get(role,{}).get('data',{}).get('result')
        raw, direct, web = evidence('raw'), evidence('direct'), evidence('website')
        regressions = evidence('regressions')
        if regressions:
            for case in regressions['checks']:
                check('regression.'+case['id'], case['status']=='PASS', case['explanation'],
                      ['regressions.json','regressions.log'], status=case['status'])
        if raw:
            meta = raw['metadata']
            check('checkpoint_metadata', meta['model_params']==expected['model_params'] and
                  meta['data_metadata']['train']['seq_len']==expected['seq_len'] and
                  meta['data_metadata']['train']['genome']==expected['genome'] and
                  meta['data_metadata']['tasks']['name']==expected['tasks'],
                  'Native checkpoint architecture, window, task and assembly must match the fixture.', ['raw.json'])
            check('checkpoint_tensors_finite', all(t['finite'] for t in raw['state_tensors'].values()),
                  'Every persisted checkpoint tensor must contain finite values.', ['raw.json'])
        if direct and raw:
            identity = direct['identity']
            check('direct_loader_independence', not direct['website_model_module_imported'] and not direct['website_sequence_module_imported'],
                  'Direct gReLU identity loader must not import website model/preprocessing modules. This is not a numerical inference test.', ['direct.json'])
            check('direct_weights_match_checkpoint', identity['state_tensors']==raw['state_tensors'],
                  'Every named loaded tensor, dtype, shape and byte hash must match the raw checkpoint.', ['raw.json','direct.json'])
        if web and raw:
            observations = web['valid_loader_observations']
            check('website_weights_match_checkpoint', bool(observations) and all(o['identity']['state_tensors']==raw['state_tensors'] for o in observations),
                  'Actual website loaders must return the real checkpoint tensors, never demo or altered weights.', ['website.json','raw.json'])
            check('website_evaluation_mode', bool(observations) and all(not o['identity']['training'] and
                  not o['identity']['training_submodules'] and o['identity']['parameter_devices']==['cpu'] for o in observations),
                  'All actual loaded website modules must be in CPU evaluation mode.', ['website.json'])
            config = web['configuration']
            check('website_provenance', Path(config['spec']['path']).resolve()==path.resolve() and
                  config['provenance']['checkpoint_sha256']==checkpoint_before and
                  config['provenance']['model_id']==expected['website_label'] and config['spec']['source']=='checkpoint' and
                  config['demo_spec']['source']=='demo',
                  'The explicit human selection must reference this checkpoint and remain distinct from the random demo.', ['website.json','checkpoint.json'])
            api = web['api']['body'] or {}
            check('website_metadata_api', web['api']['status']==200 and api.get('seq_len')==expected['seq_len'] and
                  api.get('n_tasks')==1 and api.get('tasks')==expected['tasks'] and api.get('train_genome')=='hg38' and
                  api.get('model_id')==expected['website_label'] and api.get('backend')=='grelu',
                  'Metadata API must report the selected native model window, task, assembly and backend.', ['website.json'])
            record = web['inspector_record'] or {}
            cells = web['rendered_table_cells']
            pairs = list(zip(cells[::2],cells[1::2]))
            check('website_model_inspector', web['inspector_status']==200 and record.get('selected_model')=='human' and
                  record.get('actual_model')==expected['website_label'] and
                  record.get('provenance',{}).get('checkpoint_sha256')==checkpoint_before and
                  ('Input length','2114') in pairs and ('Output tracks','1') in pairs,
                  'Saved real inspector HTML and embedded record must identify the chosen checkpoint consistently.', ['model-inspector.html','website.json'])
            check('no_selection_fallback', all(v['status']==400 and v['body'].get('error') for v in web['unavailable_selections'].values()),
                  'Unknown and uninstalled catalog selections must be rejected rather than mapped to demo or another model.', ['website.json'])
            mismatch = web['contradictory_length']
            error = str((mismatch['body'] or {}).get('error','')).lower()
            rejected = mismatch['status'] in [400,422,503] and any(word in error for word in ['length','window','mismatch','2114'])
            check('contradictory_length_guard', rejected,
                  f"Native {expected['seq_len']}-base checkpoint with deliberately declared length {expected['negative_declared_length']}: HTTP {mismatch['status']}, response {mismatch['body']}. Must reject the contradiction explicitly.", ['website.json'])
            check('incompatible_inspector_consistency', mismatch['inspector_status']==400 and
                  mismatch['inspector_error_visible'] and not mismatch['invalid_200_window_advertised'],
                  'Inspector must reject 200 with a visible mismatch error, not advertise it as a valid requirement.',
                  ['website.json','incompatible-model-inspector.html'])
            other = web['length_integration_cases']['phase2_other_128']
            pairs_other = list(zip(other['table_cells'][::2],other['table_cells'][1::2]))
            check('different_model_128_display_api_consistency', other['api_status']==200 and
                  other['api_body'].get('seq_len')==128 and other['inspector_status']==200 and
                  ('Input length','128') in pairs_other and other['configured_128_advertised'],
                  'A distinct untrained gReLU metadata fixture must consistently report its own native 128-base window; no predictions used.',
                  ['website.json','phase2_other_128-inspector.html','fixtures/other-model-128.ckpt'])
            unknown_case = web['length_integration_cases']['phase2_unknown']
            unknown_body = unknown_case['api_body'] or {}
            check('missing_length_metadata_blocked', unknown_case['api_status']==400 and unknown_body.get('seq_len') is None and
                  unknown_body.get('verified_compatibility') is False and
                  unknown_body.get('input_length_contract',{}).get('status')=='unknown' and
                  unknown_case['inspector_status']==400 and unknown_case['unknown_notice_visible'] and
                  not unknown_case['configured_128_advertised'] and not unknown_case['has_prediction_record'],
                  'Without authoritative length metadata, both API and inspector must say unknown and block compatibility claims even with configured 128.',
                  ['website.json','phase2_unknown-inspector.html','fixtures/missing-length-metadata.ckpt'])
            missing_decl = web['native_length_without_website_declaration']
            check('native_window_inferred_without_declaration', missing_decl['status']==200 and
                  missing_decl['body'].get('seq_len')==expected['seq_len'],
                  'Omitting the website length must use the native checkpoint window, not a global constant.', ['website.json'])
            contract = api.get('input_length_contract',{})
            check('architecture_vs_workflow_constraints', contract.get('architecture_length_mode')=='variable' and
                  contract.get('workflow_length_mode')=='fixed_window' and contract.get('verified_compatibility') is True,
                  'GM12878 is convolutional/variable in architecture, but its Studio window is checkpoint-defined; these constraints must remain distinct.',
                  ['website.json','raw.json'])
        if web and direct:
            comparisons = [{'website_call':i, 'same_state_tensors':o['identity']['state_tensors']==direct['identity']['state_tensors'],
                            'same_architecture':o['identity']['architecture_class']==direct['identity']['architecture_class'],
                            'same_model_params':o['identity']['model_params']==direct['identity']['model_params'],
                            'same_data_metadata':o['identity']['data_metadata']==direct['identity']['data_metadata']}
                           for i,o in enumerate(web['valid_loader_observations'])]
            save(out/'identity-comparison.json', comparisons)
            check('direct_website_identity_agreement', bool(comparisons) and all(all(v for k,v in row.items() if k!='website_call') for row in comparisons),
                  'Website and independent direct gReLU loads must agree on architecture, native metadata and every state tensor.', ['identity-comparison.json','website.json','direct.json'])
        check('checkpoint_file_unchanged', sha(path)==checkpoint_before,
              'Checkpoint SHA-256 after all workers must equal its starting hash.', ['checkpoint.json','raw.json','direct.json','website.json'])
    except Exception:
        (out/'controller-error.log').write_text(traceback.format_exc())
        check('controller_completion', False, 'Controller could not complete required work; see saved traceback.', ['controller-error.log'])
    after = snapshot(); save(out/'phase1-after.json', after)
    check('phase1_results_unchanged', before==after,
          'All existing Phase 1 result files must keep the same paths and content hashes.', ['phase1-before.json','phase1-after.json'])
    history_after = historical_runs(); save(out/'historical-runs-after.json', history_after)
    check('earlier_gate1_evidence_unchanged', history_before==history_after,
          'All previous Gate 1 runs, including the failed run, must remain byte-for-byte unchanged.',
          ['historical-runs-before.json','historical-runs-after.json'])
    check('live_server_identity', False, 'No running-server URL/process was supplied or inspected. Isolated Flask evidence is not a live-server attestation.', [], required=False, status='SKIP')
    check('browser_identity_display', False, 'Saved server-rendered HTML was checked; no browser/JavaScript execution was performed.', ['model-inspector.html'], required=False, status='SKIP')
    inventory = []
    files = ['run_studio.sh','core/models.py','core/model_lengths.py','core/io.py','core/sequences.py','workbench.py','studio_web.py','studio_services.py',
             'workflows/info.py','workflows/predict.py','variant_views.py','advanced_views.py','exploration_views.py','lab_views.py',
             'model_registry.py','experiment_views.py','experiment_store.py','templates/model_select.html','templates/analysis.html',
             'static/analysis.js','static/studio.js','static/advanced.js','static/exploration.js','static/model-inputs.js',
             'templates/lab.html','templates/exploration.html','Testing/testing-phase-2/input-output/test_length_contract.py']
    for name in files:
        file = ROOT/name
        functions = [{'name':node.name,'line':node.lineno} for node in ast.walk(ast.parse(file.read_text()))
                     if isinstance(node,(ast.FunctionDef,ast.ClassDef))] if file.suffix=='.py' else []
        inventory.append({'file':name,'sha256':sha(file),'symbols':functions})
    save(out/'source-inventory.json', inventory)
    summary = {status:sum(c['status']==status for c in checks) for status in ['PASS','FAIL','SKIP']}
    overall = 'FAIL' if summary['FAIL'] else ('BLOCKED' if any(c['required'] and c['status']=='SKIP' for c in checks) else 'PASS')
    result = {'phase':2,'step':1,'gate':1,'name':'Model Identity','status':overall,'created_utc':stamp,
              'scope':'Pinned launcher-equivalent isolated Flask website. No DNA inference, training, preprocessing verification, or biological benchmarking.',
              'summary':summary,'checks':checks,'workers':{k:{a:b for a,b in v.items() if a!='data'} for k,v in workers.items()},
              'versions':versions,'fixture_sha256':sha(FIXTURE),'runner_sha256':sha(Path(__file__)),
              'next_gate':'NOT STARTED. User approval and definition required.',
              'deferred':['Independent numerical gReLU inference comparison','Valid/invalid DNA and FASTA boundaries',
                          'Genome coordinate extraction','Prediction tensor input/output shapes','Postprocessing, plots and numerical exports']}
    save(out/'results.json', result)
    (out/'runner-source.py.txt').write_text(Path(__file__).read_text())
    (out/'regression-source.py.txt').write_text((BASE/'test_length_contract.py').read_text())
    report = ['# Phase 2, Step 1 — Gate 1: Model Identity','',f'**Status: {overall}**', '', result['scope'], '',
              'This verifies software/model identity, not biological accuracy. Gates 2–6 have not been implemented or run. The user will define them after reviewing Gate 1.', '',
              '## Automated checks','', '| Check | Status | Explanation | Evidence |','|---|---|---|---|']
    for c in checks:
        links = ', '.join(f'[{name}]({name})' for name in c['evidence']) or 'No runtime evidence; scope limitation explicitly recorded'
        report.append(f"| {c['id']} | {c['status']} | {c['explanation']} | {links} |")
    report += ['', '## Identity and provenance','', f"Repository: `{expected['repo_id']}`", f"Revision: `{expected['revision']}`",
               f"Expected checkpoint SHA-256: `{expected['checkpoint_sha256']}`", '',
               'The fixture anchors identity to the existing pinned launcher/curated model hash and earlier evidence. It is not a new remote authenticity or signature check.', '',
               'Raw checkpoint state tensors are compared individually (name, dtype, shape, finiteness and SHA-256) against actual website and direct gReLU loaded modules. The direct worker never imports website model or sequence modules. Serialization allowlists and fingerprint helpers are shared; no website DNA preprocessing is used by the direct loader.', '',
               'The website worker records actual loader returns through a delegating observation wrapper. No predictions, model parameters, metadata or return values are mocked. The wrapper is process-local and does not modify production source files. The metadata requests and inspector use real pretrained weights.', '',
               'The conflicting-length test changes only this isolated process configuration, restores it afterward, and sends no DNA. A 200 response without a mismatch warning exposes an identity/configuration consistency issue; it is not silently accepted as a pass.', '',
               '## Configuration fix and constraint provenance', '',
               'Root cause: the previous loader preferred website model.seq_len over native checkpoint metadata without checking equality; startup/UI also repeated that declaration. The fix resolves the native workflow window first and rejects incompatible declarations rather than silently replacing either value.', '',
               'For GM12878, 2,114 comes from checkpoint data_params.train.seq_len and agrees with its published input specification. DilatedConvModel is convolutional with cropping and pooling: the architecture is not classified as an exact fixed-width network. Studio still uses this checkpoint-defined reference window for its fixed-window workflows. Other valid architectural widths are not automatically verified for this scientific workflow.', '',
               'The length contract separates architecture_length_mode from workflow_length_mode. ConvModel/DilatedConvModel are recognized as structurally variable (not a promise that every positive width is valid). ConvMLPModel constructor seq_len defines a canonical length-dependent head window; pooling may admit nearby widths. Unreviewed model classes are not assumed fixed or unrestricted. Missing authoritative metadata remains unknown even if an administrator provides a number; metadata API and inspector block verified compatibility.', '',
               'Production files changed for this fix: core/model_lengths.py; core/models.py; studio_services.py; studio_web.py; advanced_views.py; workbench.py; exploration_views.py; example_views.py; lab_views.py; templates/model_select.html; templates/lab.html; templates/exploration.html; static/model-inputs.js. README.md documents the revised environment-variable rule. These changes validate configuration and report provenance without changing pretrained weights or redesigning pages. Tests added: test_length_contract.py and additional real metadata integration checks in verify_gate1.py.', '',
               'The 128-base and missing-metadata integration fixtures are explicitly untrained small gReLU checkpoints. They verify metadata/HTTP/display behavior only, do not substitute for the real pretrained GM12878 identity checks, and produce no predictions. Metadata unit tests also make no inference-correctness claims.', '',
               'Source references: [official model card](https://huggingface.co/Genentech/GM12878_dnase-model) and [gReLU architecture documentation](https://genentech.github.io/gReLU/autoapi/grelu/model/models/index.html). Installed gReLU source was also inspected; source hashes and regression evidence are saved.', '',
               '## Settings, shapes and limits','',
               'Workers run sequentially, CPU, one thread, no downloads, with 1.8 GB worker RSS ceiling and 650 MB available-RAM reserve. Actual metadata-route adapter batch sizes and modes are recorded per call in website.json; they are not inference settings used to predict DNA in this gate.', '',
               'No DNA fixtures or raw prediction outputs are needed for Model Identity. Observed inference input/output shapes are explicitly null in environment.json because inference was not run. Checkpoint parameter tensor shapes are recorded. Numerical inference equivalence, preprocessing, coordinates, transformations, visualizations and exports remain unverified and are deferred to user-defined later gates.', '',
               'Live server identity and browser execution are SKIP, not implicit passes. The isolated instance uses the pinned startup branch; this does not prove the model selected in an already open browser. The existing direct Flask startup without the launcher may have only the demo configured if the model environment variables are absent.', '',
               '## Reproduce','', 'From repository root:', '', '```bash',
               '.venv/bin/python Testing/testing-phase-2/input-output/verify_gate1.py', '```','',
               'Every run writes a new timestamped directory under runs/ and leaves prior evidence intact. Exit codes: 0=PASS, 1=FAIL, 2=BLOCKED. Inspect worker logs even when a check fails. No Gate 2 action is triggered automatically.', '',
               'Machine-readable results: [results.json](results.json). Source hashes and function lines: [source-inventory.json](source-inventory.json). Environment: [environment.json](environment.json).', '',
               '## Library versions','', '```json', json.dumps(versions,indent=2), '```','']
    (out/'REPORT.md').write_text('\n'.join(report))
    save(BASE/'latest.json',{'run':str(out.relative_to(BASE)),'status':overall,'report':str((out/'REPORT.md').relative_to(BASE))})
    print(json.dumps({'gate_status':overall,'summary':summary,'report':str(out/'REPORT.md')},indent=2),flush=True)
    return 1 if overall=='FAIL' else (2 if overall=='BLOCKED' else 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=['raw','direct','website','regressions'], help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        target = args.worker_output/f'{args.worker}.json'
        try:
            result = worker(args.worker,args.checkpoint,args.worker_output,json.loads(FIXTURE.read_text()))
            save(target,{'ok':True,'result':result})
        except Exception:
            save(target,{'ok':False,'traceback':traceback.format_exc()})
            traceback.print_exc(); sys.exit(1)
    else:
        sys.exit(main(args))
