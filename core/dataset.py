"""CPU-light dataset summaries; no model loading or training."""
from collections import Counter
import csv
import io
import math
import statistics

MAX_RECORDS = 5000
MAX_BASES = 200_000


def histogram(values, kind):
    """Count observations in explicit, inclusive/exclusive labeled bins."""
    if not values:
        return []
    low, high = min(values), max(values)
    if kind == 'gc':
        width, low, size = 10, 0, 10
    elif low == high:
        return [{'label': f'{low:g}', 'count': len(values), 'width': 100.0}]
    elif kind == 'length':
        width = max(1, math.ceil((high - low + 1) / 8))
        size = math.ceil((high - low + 1) / width)
    else:
        width, size = (high - low) / 8 or (high - low), 8
    counts = [0] * size
    for value in values:
        counts[min(size - 1, max(0, int((value - low) / width)))] += 1
    result = []
    for index, count in enumerate(counts):
        start = low + index * width
        end = min(high, start + width - 1) if kind == 'length' else start + width
        if kind == 'length':
            label = f'{start:g}' if start == end else f'{start:g}–{end:g}'
        elif kind == 'gc':
            label = f'{start:g}–{"<" if index < size - 1 else ""}{end:g}%'
        else:
            label = f'[{start:.4g}, {end:.4g}{"]" if index == size - 1 else ")"}'
        result.append({'label': label, 'count': count, 'width': 100 * count / max(counts)})
    return result


def dataset_overview(text, kind='fasta'):
    if kind not in {'fasta', 'csv'}:
        raise ValueError('Choose FASTA / DNA or labeled CSV.')
    if len(text.encode('utf-8')) > 550_000:
        raise ValueError('Use a dataset under 550 KB.')
    text = text.lstrip('\ufeff')
    records, total = [], 0

    def add(identifier, sequence, label=None):
        nonlocal total
        from core.dna_inputs import normalize_dna
        sequence = normalize_dna(sequence, max_length=MAX_BASES)
        if not sequence or any(base not in 'ACGTN' for base in sequence):
            raise ValueError(f'Record {len(records) + 1} must contain nonempty A/C/G/T/N DNA.')
        if len(records) >= MAX_RECORDS:
            raise ValueError('Use at most 5,000 sequences per overview.')
        total += len(sequence)
        if total > MAX_BASES:
            raise ValueError('Use at most 200,000 total sequence bases per overview.')
        records.append({'id': identifier, 'sequence': sequence, 'label': label})

    if kind == 'csv':
        reader = csv.DictReader(io.StringIO(text), strict=True)
        if reader.fieldnames != ['sequence', 'label']:
            raise ValueError('Labeled CSV needs exactly sequence,label columns, in that order.')
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'CSV row {line} has missing or extra fields.')
            try:
                label = float(row['label'])
            except ValueError:
                raise ValueError(f'CSV row {line} needs a numeric label.') from None
            if not math.isfinite(label) or abs(label) > 1e6:
                raise ValueError(f'CSV row {line} needs a finite label with magnitude at most 1,000,000.')
            add(f'record_{len(records) + 1}', row['sequence'], label)
    else:
        from core.dna_inputs import parse_records
        for row in parse_records(text,max_records=MAX_RECORDS,max_length=MAX_BASES,duplicate_ids=True):
            add(row['id'],row['sequence'])
    if not records:
        raise ValueError('Provide at least one DNA sequence or labeled CSV record.')

    sequences = Counter(row['sequence'] for row in records)
    identifiers = Counter(row['id'] for row in records)
    bases = Counter(''.join(row['sequence'] for row in records))
    lengths = [len(row['sequence']) for row in records]
    gc_values, details = [], []
    for record_index,row in enumerate(records):
        sequence = row['sequence']
        unknown = sequence.count('N')
        known = len(sequence) - unknown
        gc = 100 * (sequence.count('G') + sequence.count('C')) / known if known else None
        if gc is not None:
            gc_values.append(gc)
        details.append({'id': row['id'], 'record_index': record_index, 'length': len(sequence), 'gc_percent': gc,
                        'unknown_bases': unknown, 'duplicate_sequence': sequences[sequence] > 1,
                        'label': row['label']})
    duplicates = len(records) - len(sequences)
    known_total = total - bases['N']
    summary = {
        'records': len(records), 'total_bases': total, 'unique_sequences': len(sequences),
        'duplicate_records': duplicates, 'duplicate_groups': sum(count > 1 for count in sequences.values()),
        'repeated_identifiers': len(records) - len(identifiers),
        'length_min': min(lengths), 'length_max': max(lengths),
        'length_mean': statistics.mean(lengths), 'length_median': statistics.median(lengths),
        'unknown_bases': bases['N'], 'records_with_unknowns': sum(row['unknown_bases'] > 0 for row in details),
        'all_unknown_records': len(records) - len(gc_values),
        'mean_record_gc_percent': statistics.mean(gc_values) if gc_values else None,
        'pooled_gc_percent': 100 * (bases['G'] + bases['C']) / known_total if known_total else None,
    }
    notes = []
    if min(lengths) != max(lengths):
        notes.append('Sequence lengths vary. Training and prediction require the selected model’s input length.')
    if duplicates:
        notes.append(f'{duplicates} duplicate record(s) beyond the first occurrence. Remove duplicates before splitting training and validation data.')
    if summary['repeated_identifiers']:
        notes.append('Some FASTA identifiers repeat. Use unique identifiers before model-based batch analysis.')
    if bases['N']:
        notes.append('N bases are unknown. GC percentages exclude N; all-N records have unavailable GC.')
    distributions = [{'title': 'Sequence lengths (bases)', 'bins': histogram(lengths, 'length')},
                     {'title': 'GC content per sequence', 'bins': histogram(gc_values, 'gc')}]
    labels = None
    if kind == 'csv':
        values = [row['label'] for row in records]
        binary = all(value in {0, 1} for value in values)
        grouped = {}
        for row in records:
            grouped.setdefault(row['sequence'], set()).add(row['label'])
        labels = {'count': len(values), 'minimum': min(values), 'maximum': max(values),
                  'mean': statistics.mean(values), 'median': statistics.median(values),
                  'population_sd': statistics.pstdev(values), 'binary': binary,
                  'conflicting_sequence_labels': sum(len(group) > 1 for group in grouped.values()),
                  'class_counts': {'0': values.count(0), '1': values.count(1)} if binary else None}
        if labels['conflicting_sequence_labels']:
            notes.append('Some identical sequences have different labels. Review whether these are intended replicates or inconsistent targets.')
        if min(values) == max(values):
            notes.append('All labels are identical; this dataset cannot demonstrate learning differences between targets.')
        distributions.append({'title': 'Label distribution', 'bins': histogram(values, 'label')})
    metrics = {'Sequences': len(records), 'Total bases': total, 'Unique sequences': len(sequences),
               'Duplicate records': duplicates}
    result = {'format': kind, 'summary': summary, 'base_counts': {base: bases[base] for base in 'ACGTN'},
              'labels': labels, 'records': details, 'metrics': metrics, 'warnings': notes,
              'distributions': distributions,
              'method': 'Descriptive dataset statistics; no model inference or training. GC excludes unknown N bases.'}
    tables = [
        ('Dataset summary', ['Statistic', 'Value'], [(key.replace('_', ' ').title(), value) for key, value in summary.items()]),
        ('Base composition', ['Base', 'Count'], list(result['base_counts'].items())),
    ]
    if labels:
        tables.append(('Numeric labels', ['Statistic', 'Value'],
                       [(key.replace('_', ' ').title(), value) for key, value in labels.items()
                        if key not in {'class_counts', 'binary'}]))
        if labels['binary']:
            tables.append(('Binary label balance', ['Label', 'Records'], list(labels['class_counts'].items())))
    tables.append(('Sequence records (first 200)', ['Record', 'Length', 'GC %', 'N bases', 'Repeated sequence', 'Label'],
                   [(row['id'], row['length'], row['gc_percent'], row['unknown_bases'],
                     'Yes' if row['duplicate_sequence'] else 'No', row['label']) for row in details[:200]]))
    return result, tables
