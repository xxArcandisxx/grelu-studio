"""Standalone optional gReLU dependency test; no Studio imports or model loading.

Run with Python -I from /tmp to exclude the website from import paths.
FAIL is deliberately returned while native inverse conversion is broken.
"""
import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
import os
import sys
import traceback
import warnings


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    import grelu.sequence.format as fmt
    torch.set_num_threads(1)
    vectors = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
               'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def tensor(sequences):
        return torch.tensor([[vectors[b] for b in s] for s in sequences],
                            dtype=torch.float32).transpose(1, 2).contiguous()

    def attempt(fn):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                value = fn()
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    value = value.tolist()
                return {'value': value, 'error': None,
                        'warnings': [str(w.message) for w in caught]}
            except Exception:
                return {'value': None, 'error': traceback.format_exc(),
                        'warnings': [str(w.message) for w in caught]}

    checks = []
    for dna in ['A', 'C', 'G', 'T', 'N', 'ACGTN', 'GANTCTNNAG']:
        x = tensor([dna])
        native = fmt.strings_to_one_hot([dna])
        observed = attempt(lambda: fmt.one_hot_to_strings(x))
        correct = observed['value'] == [dna] and observed['error'] is None
        row = {'id': 'short_' + dna + '_roundtrip',
               'status': 'PASS' if correct else 'FAIL', 'scope': 'optional dependency compatibility',
               'dna': dna, 'expected': [dna], 'actual': observed, 'shape': list(x.shape),
               'dtype': str(x.dtype), 'device': str(x.device),
               'accepted_by_library_check': bool(fmt.check_one_hot(x)),
               'detected_format': fmt.get_input_type(x),
               'equals_native_encoder': bool(torch.equal(x, native))}
        np.save(out / (dna + '-input.npy'), x.numpy(), allow_pickle=False)
        save(out / (dna + '.json'), row)
        checks.append(row)
        print(row['status'], row['id'], flush=True)

    # Dimension and batch experiments are supplementary diagnostics, not a
    # replacement for any failed primary compatibility test.
    diagnostics = []
    for sequences in [['A'], ['N'], ['A', 'C', 'G', 'T', 'N'],
                      ['AC'], ['AN'], ['NN'], ['AC', 'NN'], ['ACGTN']]:
        full = tensor(sequences)
        for batched in ([True, False] if len(sequences) == 1 else [True]):
            x = full if batched else full[0]
            expected = sequences if batched else sequences[0]
            integer = x.argmax(axis=-2).numpy().astype(np.int8)
            mask = x.max(axis=-2).values == 0
            def mask_assign(use_numpy):
                result = integer.copy()
                result[mask.numpy() if use_numpy else mask] = 4
                return result
            diagnostics.append({'sequences': sequences, 'shape': list(x.shape),
                'accepted_by_library_check': bool(fmt.check_one_hot(x)),
                'expected_strings': expected,
                'expected_indices': [[indices[b] for b in s] for s in sequences] if batched else [indices[b] for b in sequences[0]],
                'argmax_before_N_handling': integer.tolist(),
                'torch_boolean_mask': mask.tolist(), 'mask_dtype': str(mask.dtype),
                'native_inverse_indices': attempt(lambda: fmt.one_hot_to_indices(x)),
                'native_inverse_strings': attempt(lambda: fmt.one_hot_to_strings(x)),
                'same_source_operation_torch_mask': attempt(lambda: mask_assign(False)),
                'diagnostic_only_numpy_mask': attempt(lambda: mask_assign(True)),
                'note': 'NumPy-mask variant is an isolated cause experiment; installed functions and website code are not patched.'})
    save(out / 'dimension-and-mask-diagnostics.json', diagnostics)
    source_file = Path(inspect.getfile(fmt)).resolve()
    symbols = ['check_one_hot', 'one_hot_to_indices', 'one_hot_to_strings', 'indices_to_strings']
    sources = {name: {'line': inspect.getsourcelines(getattr(fmt, name))[1],
                      'source': inspect.getsource(getattr(fmt, name))} for name in symbols}
    save(out / 'installed-source.json', {'path': str(source_file),
         'sha256': hashlib.sha256(source_file.read_bytes()).hexdigest(),
         'module_format_contract': fmt.__doc__, 'functions': sources})
    forbidden = sorted(name for name in sys.modules if name == 'engine' or name == 'studio_web'
                       or name == 'core' or name.startswith(('core.', 'workflows.')))
    environment = {'python': sys.version, 'versions': {n: version(n) for n in ['grelu', 'torch', 'numpy']},
        'cwd': os.getcwd(), 'sys_path': sys.path, 'isolated_python': bool(sys.flags.isolated),
        'studio_modules_loaded': forbidden, 'models_loaded': 0, 'inference_executed': False,
        'installed_package_modified': False}
    save(out / 'environment.json', environment)
    counts = {s: sum(row['status'] == s for row in checks) for s in ['PASS', 'FAIL', 'SKIP', 'BLOCKED']}
    result = {'category': 'optional external dependency compatibility', 'counts': counts,
              'status': 'FAIL' if counts['FAIL'] or forbidden else 'PASS', 'checks': checks,
              'isolation_ok': not forbidden and bool(sys.flags.isolated),
              'diagnostics': 'dimension-and-mask-diagnostics.json', 'source': 'installed-source.json'}
    save(out / 'results.json', result)
    print(json.dumps({'status': result['status'], 'counts': counts, 'isolation_ok': result['isolation_ok']}), flush=True)
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
