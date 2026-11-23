"""Strict DNA text parsing shared by website and engine input boundaries.

Policy: ASCII A/C/G/T/N, case-insensitive; ASCII edge whitespace and wrapped
lines normalize explicitly. Internal spaces and all non-ASCII DNA characters
are invalid. FASTA IDs are the nonempty first header token, preserved verbatim.
"""
ASCII_SPACE = ' \t\r\n\v\f'


def normalize_dna(raw, label='input', max_length=200_000):
    if not isinstance(raw, str):
        raise ValueError(f'Provide a {label} DNA string; numbers and missing values are not sequences.')
    if not raw.isascii():
        raise ValueError(f'{label} DNA must use ASCII A/C/G/T/N and ASCII whitespace only.')
    lines = raw.replace('\r\n','\n').replace('\r','\n').split('\n')
    sequence = ''.join(line.strip(ASCII_SPACE) for line in lines).upper()
    if not sequence or len(sequence) > max_length or any(base not in 'ACGTN' for base in sequence):
        raise ValueError(f'Enter nonempty {label} DNA of up to {max_length:,} bases using A, C, G, T or N; internal spaces are not bases.')
    return sequence


def parse_records(text, *, max_records=12, max_length=200_000, allow_raw=True, duplicate_ids=False):
    if not isinstance(text, str):
        raise ValueError('Provide DNA or FASTA text.')
    lines = [line.strip(ASCII_SPACE) for line in text.replace('\r\n','\n').replace('\r','\n').split('\n')]
    lines = [line for line in lines if line]
    if not lines:
        raise ValueError('Enter DNA or a FASTA batch.')
    if not lines[0].startswith('>'):
        if not allow_raw or any(line.startswith('>') for line in lines):
            raise ValueError('FASTA sequence data must begin with a header; preamble DNA is not discarded.')
        return [{'id':'sequence_1','sequence':normalize_dna(text,max_length=max_length)}]
    records, identifier, parts = [], None, []
    def add():
        if identifier is not None:
            records.append({'id':identifier,'sequence':normalize_dna('\n'.join(parts), f'record {identifier}', max_length)})
    for line in lines:
        if line.startswith('>'):
            add()
            header = line[1:].strip(ASCII_SPACE)
            if not header:
                raise ValueError('Each FASTA header needs a nonempty identifier; identifiers are not invented.')
            identifier = header.split(maxsplit=1)[0]
            if len(identifier) > 200:
                raise ValueError('FASTA identifiers must contain at most 200 characters.')
            parts = []
            if len(records) >= max_records:
                raise ValueError(f'Provide up to {max_records} FASTA records; records are not truncated.')
        else:
            parts.append(line)
    add()
    if not duplicate_ids and len({r['id'] for r in records}) != len(records):
        raise ValueError('Each FASTA record needs a distinct identifier.')
    return records


def parse_single(text, label='input'):
    records = parse_records(text, max_records=1)
    return records[0]['sequence']
