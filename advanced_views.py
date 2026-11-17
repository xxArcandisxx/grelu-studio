"""Bounded, reproducible design, batch variant, and uploaded-sequence scan workflows."""
import csv
import io
import random
import re
from pathlib import Path
from tempfile import TemporaryDirectory

from flask import Blueprint, current_app, jsonify, make_response, render_template, request

from studio_services import length_error, model_metadata

advanced = Blueprint('advanced', __name__)
PAGES = {
    'attribution': ('Gradient attribution', 'Explore which input bases the model is locally sensitive to, with signed contributions and a randomized-weight diagnostic.'),
    'design': ('Sequence design', 'Optimize a DNA sequence for a selected output track and inspect every accepted mutation.'),
    'batch-variants': ('Batch variant analysis', 'Score a table of single-nucleotide substitutions, compare output tracks, and rank predicted effects.'),
    'mutagenesis': ('Sequence mutagenesis', 'Map all three substitutions at each position in a selected region of your own sequence.'),
}


def integer(value, label, low, high):
    if not re.fullmatch(r'\d+', str(value)):
        raise ValueError(f'{label} must be a whole number from {low} to {high}.')
    number = int(value)
    if not low <= number <= high:
        raise ValueError(f'{label} must be between {low} and {high}.')
    return number


def example_sequence(length):
    rng = random.Random(19)
    return ''.join(rng.choices('ACGT', k=length))


def variant_example(length):
    sequence = example_sequence(length)
    output = io.StringIO()
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['variant_id', 'sequence', 'pos', 'ref', 'alt'])
    for i in range(3):
        position = (length // 2 + i) % length
        ref = sequence[position]
        writer.writerow([f'variant_{i + 1}', sequence, position + 1, ref, next(base for base in 'ACGT' if base != ref)])
    return output.getvalue()


def validate_variants(text, length, limit=24):
    """Validate every row before running; do not silently crop, truncate, or skip records."""
    from studio_web import parse_sequence

    reader = csv.DictReader(io.StringIO(text), strict=True)
    expected = {'variant_id', 'sequence', 'pos', 'ref', 'alt'}
    headers = [header.strip().lower() for header in reader.fieldnames or []]
    if len(headers) != len(set(headers)) or set(headers) != expected:
        raise ValueError('Use exactly these CSV columns: variant_id, sequence, pos, ref, alt.')
    reader.fieldnames = headers
    records, ids = [], set()
    for line, row in enumerate(reader, 2):
        if len(records) >= limit:
            raise ValueError(f'Use at most {limit} variants per browser run.')
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f'CSV line {line}: the number of values does not match the header.')
        record = {key: value.strip() for key, value in row.items()}
        vid = record['variant_id']
        if not vid or len(vid) > 100 or vid in ids:
            raise ValueError(f'CSV line {line}: use a unique variant_id of 1–100 characters.')
        sequence = parse_sequence(record['sequence'], f'CSV line {line}')
        if len(sequence) != length:
            raise ValueError(f'CSV line {line}: ' + length_error(length, [{'id': vid, 'sequence': sequence}]))
        position = integer(record['pos'], f'CSV line {line} position', 1, length)
        ref, alt = record['ref'].upper(), record['alt'].upper()
        if ref not in 'ACGT' or alt not in 'ACGT' or len(ref) != 1 or len(alt) != 1 or ref == alt:
            raise ValueError(f'CSV line {line}: ref and alt must be different single A, C, G or T bases.')
        if sequence[position - 1] != ref:
            raise ValueError(f'CSV line {line}: ref {ref} does not match base {sequence[position - 1]} at position {position}.')
        records.append({'variant_id': vid, 'sequence': sequence, 'pos': position, 'ref': ref, 'alt': alt})
        ids.add(vid)
    if not records:
        raise ValueError('Provide at least one variant row after the CSV header.')
    return records


def sanitized_result(outcome, model, selection, inputs, settings):
    from studio_web import PROVENANCE

    if not outcome['ok']:
        current_app.logger.error('Advanced workflow %s failed: %s', outcome['workflow'], outcome.get('error'))
        raise RuntimeError('The model could not complete this analysis. Check the server log and model installation.')
    outcome['model'].pop('path', None)
    for key in ('csv', 'fasta'):
        outcome['results'].pop(key, None)
    outcome.update(inputs=inputs, settings=settings, selected_model=selection, demo=model == 'demo')
    outcome['model']['provenance'] = PROVENANCE[model]
    outcome['actual_model'] = PROVENANCE[model]['model_id']
    return outcome


@advanced.get('/api/model-info')
def model_info():
    from studio_web import PROVENANCE, resolve_model

    model = resolve_model(request.args.get('model', 'demo'))
    if model is None:
        return jsonify(error='Choose an installed local model.'), 400
    try:
        card, warnings = model_metadata(model)
        return jsonify(**card, model_id=PROVENANCE[model]['model_id'], demo=model == 'demo', warnings=warnings)
    except Exception:
        current_app.logger.exception('Could not inspect model %s', model)
        return jsonify(error='Could not load model metadata. Check the server installation.'), 503


@advanced.get('/api/workflow-example/<mode>')
def workflow_example(mode):
    from studio_web import MODELS, resolve_model

    if mode not in PAGES:
        return jsonify(error='Unknown workflow.'), 404
    model = resolve_model(request.args.get('model', 'demo'))
    if model is None:
        return jsonify(error='Choose an installed local model.'), 400
    length = MODELS[model]['seq_len']
    if mode == 'batch-variants':
        return jsonify(text=variant_example(length))
    start = max(1, length // 2 - 49) if mode == 'attribution' else length // 2 + 1
    end = min(length, start + (99 if mode == 'attribution' else 7))
    return jsonify(text='>synthetic_example\n' + example_sequence(length), start=start, end=end)


@advanced.route('/attribution', methods=['GET', 'POST'], defaults={'mode': 'attribution'})
@advanced.route('/design', methods=['GET', 'POST'], defaults={'mode': 'design'})
@advanced.route('/batch-variants', methods=['GET', 'POST'], defaults={'mode': 'batch-variants'})
@advanced.route('/mutagenesis', methods=['GET', 'POST'], defaults={'mode': 'mutagenesis'})
def run(mode):
    from engine import execute_platform_workflow
    from studio_web import MODELS, inference_lock, model_choices, parse_sequence, resolve_model
    from workbench import parse_batch

    title, description = PAGES[mode]
    selection = request.form.get('model', 'demo')
    model = resolve_model(selection)
    choices = model_choices()
    values = {'text': request.form.get('text', ''), 'target': request.form.get('target', '0'),
              'off_target': request.form.get('off_target', ''), 'iterations': request.form.get('iterations', '4'),
              'candidates': request.form.get('candidates', '8'), 'shuffles': request.form.get('shuffles', '0'),
              'start': request.form.get('start', '51' if mode == 'attribution' else '101'),
              'end': request.form.get('end', '150' if mode == 'attribution' else '108'),
              'randomized_check': 'randomized_check' in request.form,
              'edit_start': request.form.get('edit_start', ''), 'edit_end': request.form.get('edit_end', '')}
    result, error, status, heatmap_scale = None, None, 200, 0
    try:
        if model is None:
            raise ValueError('Choose an installed local model.')
        length = MODELS[model]['seq_len']
        if request.method == 'POST':
            upload = request.files.get('file')
            if upload and upload.filename:
                values['text'] = upload.read().decode('utf-8-sig')
            if 'example' in request.form:
                values['text'] = variant_example(length) if mode == 'batch-variants' else '>synthetic_example\n' + example_sequence(length)
                region_start = max(1, length // 2 - 49) if mode == 'attribution' else length // 2 + 1
                values.update(start=str(region_start), end=str(min(length, region_start + (99 if mode == 'attribution' else 7))))
            if mode == 'batch-variants':
                records = validate_variants(values['text'], length)
                shuffles = integer(values['shuffles'], 'Background shuffles', 0, 100)
                budget = 256 if model == 'demo' else 48
                if len(records) * (4 + 2 * shuffles) > budget:
                    raise ValueError(f'This CPU run allows {budget} sequence evaluations. Reduce the number of variants or shuffles, or choose the compact CNN.')
                with inference_lock, TemporaryDirectory(prefix='grelu-batch-') as path:
                    csv_path = Path(path) / 'variants.csv'
                    with csv_path.open('w', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=['variant_id', 'sequence', 'pos', 'ref', 'alt'])
                        writer.writeheader()
                        writer.writerows(records)
                    outcome = execute_platform_workflow('vep', {
                        'model': MODELS[model], 'device': 'cpu', 'variants_csv': str(csv_path),
                        'output_dir': path, 'center_bins': 0, 'n_shuffles': shuffles, 'seed': 0, 'batch_size': 1,
                    })
                result = sanitized_result(outcome, model, selection, records, {'n_shuffles': shuffles, 'seed': 0, 'aggregation': 'Mean across all output bins'})
                ranked = sorted(result['results']['variants'], key=lambda row: -abs(row['top_effect']))
                for rank, row in enumerate(ranked, 1):
                    row['rank'] = rank
                result['results']['ranked_variants'] = ranked
            else:
                records = parse_batch(values['text'], parse_sequence)
                if len(records) != 1:
                    raise ValueError('Provide exactly one DNA sequence or FASTA record for this workflow.')
                if len(records[0]['sequence']) != length:
                    raise ValueError(length_error(length, records))
                card, _ = model_metadata(model)
                target = integer(values['target'], 'Output track index', 0, card['n_tasks'] - 1)
                if mode == 'design':
                    if 'N' in records[0]['sequence']:
                        raise ValueError('Sequence design requires known A, C, G and T bases; remove unknown N bases first.')
                    off = [] if values['off_target'] == '' else [integer(values['off_target'], 'Off-target index', 0, card['n_tasks'] - 1)]
                    if target in off:
                        raise ValueError('Choose different on-target and off-target tracks.')
                    iterations = integer(values['iterations'], 'Iterations', 1, 12)
                    candidates = integer(values['candidates'], 'Candidates per iteration', 1, 32)
                    budget = 384 if model == 'demo' else 48
                    if iterations * candidates > budget:
                        raise ValueError(f'This CPU run allows {budget} candidate evaluations. Reduce iterations or candidates.')
                    edit_start = integer(values['edit_start'] or '1', 'First editable position', 1, length)
                    edit_end = integer(values['edit_end'] or str(length), 'Last editable position', edit_start, length)
                    settings = {'edit_start': edit_start, 'edit_end': edit_end, 'on_targets': [target], 'off_targets': off, 'max_iter': iterations,
                                'candidates_per_iter': candidates, 'n_baseline': 5, 'seed': 0, 'center_bins': 0}
                    with inference_lock, TemporaryDirectory(prefix='grelu-design-') as path:
                        outcome = execute_platform_workflow('design', {**settings, 'model': MODELS[model],
                            'device': 'cpu', 'seeds': [records[0]['sequence']], 'output_dir': path, 'batch_size': 1})
                    result = sanitized_result(outcome, model, selection, records, settings)
                    result['results']['fasta_text'] = '\n'.join(
                        f">design_{row['seed_index'] + 1}_score_{row['score_after']:.6g}\n{row['designed_sequence']}"
                        for row in result['results']['designs']) + '\n'
                else:
                    start = integer(values['start'], 'First position', 1, length)
                    end = integer(values['end'], 'Last position', start, length)
                    region_limit = 200 if mode == 'attribution' else 24
                    if end - start + 1 > region_limit:
                        raise ValueError(f'Choose at most {region_limit} positions for this view. Positions are 1-based and inclusive.')
                    if mode == 'attribution' and length > 20000:
                        raise ValueError('Gradient attribution supports model windows up to 20,000 bases on this CPU interface.')
                    settings = {'start': start, 'end': end, 'track_index': target, 'seed': 0}
                    if mode == 'attribution':
                        settings['randomized_check'] = values['randomized_check']
                    with inference_lock:
                        outcome = execute_platform_workflow('attribution' if mode == 'attribution' else 'scan', {**settings, 'model': MODELS[model], 'sequence': records[0]['sequence']})
                    result = sanitized_result(outcome, model, selection, records, settings)
                    if mode != 'attribution':
                        heatmap_scale = max((abs(value) for row in result['results']['positions'] for value in row['deltas'].values()), default=0)
    except (ValueError, UnicodeDecodeError, csv.Error) as exc:
        error, status = str(exc), 400
    except RuntimeError as exc:
        error, status = str(exc), 500
    if result is not None:
        from experiment_store import save_experiment
        result['saved_run_id'] = save_experiment(title, mode, result, request.path, {**values, 'model': selection})
    response = make_response(render_template('advanced.html', mode=mode, title=title, description=description,
        choices=choices, selected=selection, values=values, result=result, error=error, heatmap_scale=heatmap_scale), status)
    response.headers['Cache-Control'] = 'no-store'
    return response
