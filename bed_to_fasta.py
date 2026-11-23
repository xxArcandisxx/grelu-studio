"""Fetch a small, verified BED sample from UCSC (hg38 by default).

    .venv/bin/python bed_to_fasta.py

Creates datasets/gm12878/test_10.bed, test_10.fasta, and provenance JSON.
BED starts are zero-based and ends are exclusive. Sequences use the reference
forward strand. This helper is separate from the website's future BED workflow.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import requests

API = 'https://api.genome.ucsc.edu/getData/sequence'


def read_intervals(path, limit, offset=0, expected_length=2114):
    rows = []
    seen = 0
    with path.open() as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip() or line.startswith(('#', 'track ', 'browser ')):
                continue
            if seen < offset:
                seen += 1
                continue
            fields = line.split()
            if len(fields) < 3 or not re.fullmatch(r'[A-Za-z0-9_.-]+', fields[0]):
                raise ValueError(f'Invalid BED columns at line {line_number}.')
            start, end = int(fields[1]), int(fields[2])
            if start < 0 or end <= start or end - start != expected_length:
                raise ValueError(f'BED line {line_number} must span {expected_length:,} bases with nonnegative coordinates.')
            rows.append({'chrom': fields[0], 'start': start, 'end': end})
            if len(rows) == limit:
                break
    if len(rows) != limit:
        raise ValueError(f'Requested {limit} intervals at offset {offset}, but only found {len(rows)}.')
    return rows


def fetch_sequence(client, interval, genome):
    for attempt in range(3):
        try:
            response = client.get(API, params={'genome': genome, **interval}, timeout=(10, 30))
            response.raise_for_status()
            data = response.json()
            sequence = data.get('dna', '').upper()
            if len(sequence) != interval['end'] - interval['start'] or not re.fullmatch('[ACGTN]+', sequence):
                raise ValueError('UCSC returned missing, invalid, or incorrectly sized DNA.')
            for key, expected in {'genome': genome, **interval}.items():
                if key in data and data[key] != expected:
                    raise ValueError(f'UCSC response does not match requested {key}.')
            return sequence, response.url
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == 2 or (status is not None and status not in {429, 500, 502, 503, 504}):
                raise
            time.sleep(2 ** (attempt + 1))


def convert(bed, output, limit=10, offset=0, genome='hg38', expected_length=2114):
    intervals = read_intervals(bed, limit, offset, expected_length)
    if output.with_suffix('.bed').resolve() == bed.resolve():
        raise ValueError('Choose an output name different from the source BED.')
    sequences, provenance = [], []
    with requests.Session() as client:
        client.headers['User-Agent'] = 'gReLU-Studio-local-sequence-converter/1.0'
        for index, interval in enumerate(intervals):
            if index:
                # UCSC recommends no more than approximately one request/second.
                time.sleep(1)
            sequence, url = fetch_sequence(client, interval, genome)
            identifier = f"{interval['chrom']}:{interval['start']}-{interval['end']}"
            sequences.append('>' + identifier + '\n' + '\n'.join(sequence[i:i+70] for i in range(0, len(sequence), 70)) + '\n')
            provenance.append({**interval, 'id': identifier, 'length': len(sequence),
                               'unknown_bases': sequence.count('N'), 'request_url': url,
                               'sequence_sha256': hashlib.sha256(sequence.encode()).hexdigest()})
            print(f'Fetched {index+1}/{limit}: {identifier} ({len(sequence):,} bases)', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not leave a final FASTA claiming success after a partial network failure.
    temporary = output.with_suffix(output.suffix + '.part')
    temporary.write_text(''.join(sequences))
    temporary.replace(output)
    output.with_suffix('.bed').write_text(''.join(f"{row['chrom']}\t{row['start']}\t{row['end']}\n" for row in intervals))
    with bed.open('rb') as source:
        source_digest = hashlib.file_digest(source, 'sha256').hexdigest()
    metadata = {'assembly': genome, 'strand': 'reference forward', 'coordinate_system': 'BED: zero-based start, exclusive end',
                'source_bed': str(bed), 'source_bed_sha256': source_digest, 'offset': offset,
                'records': len(intervals), 'expected_length': expected_length,
                'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'endpoint': API,
                'fasta_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'intervals': provenance}
    output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Created {output} ({len(intervals)} verified records).', flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bed', type=Path, default=Path('datasets/gm12878/intervals.bed'))
    parser.add_argument('--output', type=Path, default=Path('datasets/gm12878/test_10.fasta'))
    parser.add_argument('--limit', type=int, default=10)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--genome', default='hg38')
    parser.add_argument('--length', type=int, default=2114)
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000 or args.offset < 0 or args.length < 1:
        parser.error('Use 1–1000 records, a nonnegative offset, and a positive length.')
    try:
        convert(args.bed, args.output, args.limit, args.offset, args.genome, args.length)
    except (ValueError, OSError, requests.RequestException) as error:
        parser.exit(1, f'Conversion failed: {error}\n')


if __name__ == '__main__':
    main()
