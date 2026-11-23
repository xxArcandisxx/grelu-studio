"""Independent installed-gReLU inference. No Studio imports or normalization."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    import numpy as np
    import torch
    from grelu.lightning import LightningModel
    torch.set_num_threads(1)
    fixture = json.loads((run / 'fixtures.json').read_text())
    identity = fixture['identity']
    checkpoint = fixture['checkpoint_path']
    if digest(checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('Reference checkpoint fingerprint mismatch')
    # Deserialization compatibility only; no Studio utilities are imported.
    allowed = [np.dtype, np.ndarray,
               (np._core.multiarray._reconstruct, 'numpy.core.multiarray._reconstruct')]
    allowed += [getattr(np.dtypes, name) for name in ['BoolDType', 'Float32DType',
                'Float64DType', 'Int32DType', 'Int64DType', 'ObjectDType', 'StrDType']]
    with torch.serialization.safe_globals(allowed):
        model = LightningModel.load_from_checkpoint(checkpoint, map_location='cpu', weights_only=True)
    model = model.cpu().eval()

    def state_hash():
        h = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    initial_state = state_hash()
    queries = fixture['reference_queries']
    output = run / 'direct'
    output.mkdir()
    observations = []
    for size in [1, 2]:
        batches = []
        for start in range(0, len(queries), size):
            chunk = queries[start:start + size]
            # Native conversion/forward/activation/transform only. Deliberately
            # no reshaping, aggregation, result normalization, or Studio helper.
            result = model.predict_on_seqs([q['sequence'] for q in chunk], device='cpu')
            file = f'batch{size}-{start:03d}.npy'
            np.save(output / file, result, allow_pickle=False)
            observations.append(dict(batch_size=size, start=start,
                query_ids=[q['id'] for q in chunk], shape=list(result.shape),
                dtype=str(result.dtype), finite=bool(np.isfinite(result).all()),
                device='cpu', evaluation_mode=not model.training,
                file=file, grad_enabled=torch.is_grad_enabled(),
                inference_mode=torch.is_inference_mode_enabled()))
            batches.append(result)
            print('DIRECT', size, start, list(result.shape), flush=True)
        # Concatenation stores ordered native batches; it does not transform
        # axes, select tasks, or aggregate any values.
        np.save(output / f'all-batch{size}.npy', np.concatenate(batches, axis=0), allow_pickle=False)
    forbidden = sorted(name for name in sys.modules if name == 'engine' or name == 'studio_web'
                       or name == 'core' or name.startswith(('core.', 'workflows.')))
    metadata = dict(checkpoint=checkpoint, checkpoint_sha256=digest(checkpoint),
        state_sha256_before=initial_state, state_sha256_after=state_hash(),
        loader='grelu.lightning.LightningModel.load_from_checkpoint',
        predictor='grelu.lightning.LightningModel.predict_on_seqs',
        model_class=f'{type(model.model).__module__}.{type(model.model).__name__}',
        model_params=model.model_params, tasks=list(model.data_params['tasks']['name']),
        train_window=model.data_params['train']['seq_len'],
        activation=type(model.activation).__name__, transform=type(model.transform).__name__,
        cpu=True, evaluation_mode=not model.training, torch_threads=torch.get_num_threads(),
        versions={name: version(name) for name in ['grelu', 'torch', 'numpy']},
        studio_modules_loaded=forbidden, isolated_python=bool(sys.flags.isolated),
        import_path=sys.path, output_axes=['sequence', 'task', 'position'],
        output_definition='LightningModel.forward return after native activation/transform, before any Studio processing.',
        observations=observations)
    save(output / 'manifest.json', metadata)
    if forbidden or initial_state != state_hash():
        raise RuntimeError('Independent reference isolation or weight preservation failed')


if __name__ == '__main__':
    main()
