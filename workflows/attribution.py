from core.output_processing import predict_checked
"""Local gradient sensitivity with optional randomized-weight sanity comparison."""
import numpy as np

from core.models import load_model
from core.stats import pearson
from core.utils import card


def handle_attribution_workflow(params):
    sequence = params['sequence']
    target = int(params.get('track_index', 0))
    start, end = int(params.get('start', 1)), int(params.get('end', len(sequence)))
    adapter, warnings = load_model(params['model'], 'cpu', batch_size=1)
    if len(sequence) != adapter.seq_len or len(sequence) > 20000 or any(base not in 'ACGTN' for base in sequence):
        raise ValueError('Use one full model-length A/C/G/T/N sequence of at most 20,000 bases.')
    if not 0 <= target < len(adapter.task_names):
        raise ValueError('Choose an available output track.')
    if not 1 <= start <= end <= len(sequence) or end - start + 1 > 200:
        raise ValueError('Display a valid region of at most 200 positions.')
    score = float(predict_checked(adapter, [sequence])[0, target].mean())
    matrix = adapter.grad_attribution([sequence], [target], 0)[0]
    contributions = matrix.sum(axis=0)
    if not np.isfinite(matrix).all() or not np.isfinite(score):
        raise ValueError('The model returned non-finite gradients or scores.')
    positions = [{'position': index + 1, 'reference': base, 'contribution': float(contributions[index])}
                 for index, base in enumerate(sequence)]
    randomized_correlation = None
    if params.get('randomized_check'):
        randomized = adapter.randomized_copy(seed=1)
        reference = randomized.grad_attribution([sequence], [target], 0)[0].sum(axis=0)
        if not np.isfinite(reference).all():
            raise ValueError('The randomized-weight model returned non-finite gradients.')
        randomized_correlation = pearson(contributions, reference)
    evidence = [
        card('Method', 'info', 'Gradient × input with channel-mean correction, using the mean selected-track output over all bins.'),
        card('Interpretation', 'info', 'Signed values describe local input sensitivity. They are not mutation deltas, causal effects, or DeepLIFT/SHAP scores.'),
        card('Unknown bases', 'info', f"{sequence.count('N')} unknown N positions have zero input contribution."),
    ]
    if params.get('randomized_check'):
        evidence.append(card('Randomized-weight check', 'info',
                             'Pearson correlation with a randomized copy: ' +
                             (f'{randomized_correlation:.4g}. This is a diagnostic, not a pass/fail certificate.'
                              if randomized_correlation is not None else 'undefined because a vector is constant.')))
    return {'model': {**adapter.meta, 'tasks': adapter.task_names, 'seq_len': adapter.seq_len}, 'device': 'cpu',
            'results': {'track': adapter.task_names[target], 'reference_score': score,
                        'method': 'channel_mean_corrected_gradient_x_input', 'positions': positions,
                        'matrix': matrix.tolist(), 'matrix_bases': list('ACGT'),
                        'display_positions': positions[start - 1:end],
                        'top_positions': sorted((row for row in positions if row['reference'] != 'N'),
                                                key=lambda row: -abs(row['contribution']))[:20],
                        'randomized_check': bool(params.get('randomized_check')),
                        'randomized_correlation': randomized_correlation},
            'evidence': evidence, 'warnings': warnings + adapter.notes}
