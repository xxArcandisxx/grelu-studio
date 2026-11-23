from core.output_processing import predict_checked
"""In silico saturation mutagenesis for a bounded uploaded-sequence region."""
import math

from core.models import load_model
from core.sequences import aggregate
from core.utils import card


def handle_scan_workflow(params):
    sequence = params['sequence']
    start, end, target = int(params['start']), int(params['end']), int(params.get('track_index', 0))
    if not 1 <= start <= end <= len(sequence) or end - start + 1 > 24:
        raise ValueError('Choose a valid 1-based inclusive region of at most 24 positions.')
    adapter, warnings = load_model(params['model'], 'cpu', batch_size=1)
    if len(sequence) != adapter.seq_len or any(base not in 'ACGTN' for base in sequence):
        raise ValueError('Use one full model-length A/C/G/T/N sequence.')
    if not 0 <= target < len(adapter.task_names):
        raise ValueError('Choose an available output track.')
    reference = float(aggregate(predict_checked(adapter, [sequence]), 0)[0, target])
    positions = []
    for index in range(start - 1, end):
        original = sequence[index]
        if original == 'N':
            positions.append({'position': index + 1, 'reference': original, 'deltas': {}, 'skipped': 'Unknown reference base'})
            continue
        bases = [base for base in 'ACGT' if base != original]
        mutants = [sequence[:index] + base + sequence[index + 1:] for base in bases]
        scores = aggregate(predict_checked(adapter, mutants), 0)[:, target]
        deltas = {base: float(score) - reference for base, score in zip(bases, scores, strict=True)}
        if not math.isfinite(reference) or not all(math.isfinite(value) for value in deltas.values()):
            raise ValueError('The model returned a non-finite score.')
        positions.append({'position': index + 1, 'reference': original, 'deltas': deltas})
    count = sum(len(row['deltas']) for row in positions)
    return {'model': {**adapter.meta, 'tasks': adapter.task_names, 'seq_len': adapter.seq_len}, 'device': 'cpu',
            'results': {'track': adapter.task_names[target], 'reference_score': reference, 'positions': positions,
                        'substitutions': count, 'skipped_positions': sum(row['reference'] == 'N' for row in positions)},
            'evidence': [card('Scoring', 'info', 'Each delta is mutant minus reference, averaged over model output bins.'),
                         card('Scope', 'info', 'These are model predictions. They do not establish an experimental regulatory effect.')],
            'warnings': warnings + adapter.notes}
