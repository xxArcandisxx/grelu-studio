from core.output_processing import predict_checked
"""CPU-bounded motif placement and explicit cross-model comparisons."""
import math
import random

from core.models import load_model
from core.sequences import aggregate, fit_with_offset, revcomp
from core.stats import pearson
from core.model_lengths import required_length


def prepare_comparison_inputs(specifications, records, policy):
    """Validate and record ALL model windows before any model can see input."""
    if policy not in ('strict','center'):
        raise ValueError('Choose strict input lengths or explicit center-crop/N-pad handling.')
    prepared=[]
    for spec in specifications:
        expected=required_length(spec)
        if expected>20000:
            raise ValueError('CPU comparison supports model windows up to 20,000 bases.')
        sequences,details=[],[]
        for row in records:
            original=row['sequence'];length=len(original)
            if policy=='strict' and length!=expected:
                raise ValueError(f"Record {row['id']} has {length:,} bases; this model needs {expected:,}. Choose explicit center-crop/N-pad handling or matching model windows.")
            sequence,offset=fit_with_offset(original,expected)
            description=('Unchanged' if length==expected else
                f'Center-cropped from {length:,} to {expected:,} bases; removed {-offset} bases on the left' if length>expected else
                f'N-padded from {length:,} to {expected:,} bases; added {offset} bases on the left')
            sequences.append(sequence)
            details.append({'id':row['id'],'original_length':length,'model_length':expected,'offset':offset,
                            'transformation':description,'sequence':sequence})
        prepared.append({'sequences':sequences,'details':details})
    return prepared


def simulate_motif(spec, sequence, motif, positions, target=0, strands=('+',), partner='', gaps=(0,), controls=True):
    rng = random.Random(23)
    letters = list(motif)
    rng.shuffle(letters)
    scrambled = ''.join(letters)
    conditions = []
    for position in positions:
        start = position - 1
        for strand in strands:
            primary = motif if strand == '+' else revcomp(motif)
            control = scrambled if strand == '+' else revcomp(scrambled)
            for gap in (gaps if partner else (0,)):
                end = start + len(primary) + (gap + len(partner) if partner else 0)
                if start < 0 or end > len(sequence):
                    raise ValueError(f'Motif placement at {position} extends beyond the input window.')
                mutant = sequence[:start] + primary + sequence[start+len(primary):]
                null = sequence[:start] + control + sequence[start+len(control):]
                if partner:
                    partner_start = start + len(primary) + gap
                    mutant = mutant[:partner_start] + partner + mutant[partner_start+len(partner):]
                    null = null[:partner_start] + partner + null[partner_start+len(partner):]
                conditions.append({'position':position,'strand':strand,'partner_gap':gap if partner else None,
                                   'sequence':mutant,'control_sequence':null if controls else None})
    if not conditions or len(conditions)>12:
        raise ValueError('Choose at most 12 position/orientation/spacing combinations.')
    adapter,warnings = load_model(spec,'cpu',batch_size=1)
    if len(sequence) != adapter.seq_len:
        raise ValueError(f'Use exactly {adapter.seq_len:,} input bases.')
    if not 0 <= target < len(adapter.task_names):
        raise ValueError('Choose an available output track.')
    sequences = [sequence] + [row['sequence'] for row in conditions]
    if controls:
        sequences += [row['control_sequence'] for row in conditions]
    scores = aggregate(predict_checked(adapter, sequences), 0)[:,target]
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError('The model returned non-finite predictions.')
    reference = float(scores[0])
    for index,row in enumerate(conditions):
        row['score'] = float(scores[index+1])
        row['delta'] = row['score'] - reference
        row['control_score'] = float(scores[1+len(conditions)+index]) if controls else None
        row['motif_minus_control'] = row['score']-row['control_score'] if controls else None
    return {'results':{'conditions':conditions,'reference_score':reference,'track':adapter.task_names[target],
                        'scrambled_motif':scrambled if controls else None,'control_identical':scrambled==motif if controls else False},
            'model':{**{key:value for key,value in adapter.meta.items() if key!='path'},'tasks':adapter.task_names,'seq_len':adapter.seq_len},
            'settings':{'motif':motif,'positions':positions,'strands':list(strands),'partner':partner,'partner_strand':'+',
                        'gaps':list(gaps) if partner else [],'track_index':target,'scrambled_controls':controls,'seed':23},
            'method':'Motifs replace equal-length segments; input length and all other bases are preserved. Deltas are mean selected-track prediction minus the unmodified input.',
            'warnings':warnings+adapter.notes,
            'evidence':[{'message':'Placement effects depend on the supplied sequence context. A scrambled control preserves primary-motif base counts but can coincide with the original motif.'}]}


def compare_models(specifications, records, targets, policy='strict'):
    prepared=prepare_comparison_inputs(specifications,records,policy)
    all_predictions,transforms = [],[]
    for spec,target,item in zip(specifications,targets,prepared,strict=True):
        effective,details=item['sequences'],item['details']
        adapter,warnings = load_model(spec,'cpu',batch_size=1)
        if not 0 <= target < len(adapter.task_names):
            raise ValueError('Choose available output track indices for both models.')
        values = aggregate(predict_checked(adapter, effective), 0)
        if not all(math.isfinite(float(value)) for row in values for value in row):
            raise ValueError('A model returned non-finite predictions.')
        all_predictions.append({'track':adapter.task_names[target],'selected_scores':[float(row[target]) for row in values],
            'all_scores':[{name:float(value) for name,value in zip(adapter.task_names,row,strict=True)} for row in values],
            'tasks':adapter.task_names,'warnings':warnings+adapter.notes})
        transforms.append(details)
    first,second = all_predictions
    comparable = specifications[0] == specifications[1] and targets[0] == targets[1]
    scale_policy = {'comparable': comparable, 'policy': 'Raw differences require identical model configurations and the same selected task; no scale harmonization is assumed.'}
    rows = [{'id':record['id'],'score_a':first['selected_scores'][index],'score_b':second['selected_scores'][index],
             'raw_difference':second['selected_scores'][index]-first['selected_scores'][index] if comparable else None,
             'transformation_a':transforms[0][index]['transformation'],'transformation_b':transforms[1][index]['transformation']}
            for index,record in enumerate(records)]
    correlation = pearson(first['selected_scores'],second['selected_scores']) if len(records)>=3 else None
    return {'results':{'comparisons':rows,'models':all_predictions,'effective_inputs':transforms,'pearson':correlation, 'scale_comparison':scale_policy},
            'settings':{'input_policy':policy,'track_a':targets[0],'track_b':targets[1]},
            'method':'Mean selected-track scores on explicitly recorded model input windows. Cross-model score differences are descriptive; track scales and training objectives can differ.',
            'warnings':first['warnings']+second['warnings']+([] if comparable else ['Scores use different or unverified model/task scales. Raw differences are not computed; no cross-model scale equivalence is established.'])}
