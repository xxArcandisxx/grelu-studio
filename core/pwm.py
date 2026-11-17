"""DNA PFM/PSPM parsing and strand-aware log-odds motif scanning."""
import math
import re

import numpy as np

BASES = 'ACGT'
EXAMPLE_MATRIX = '''>DEMO_AT_RICH Synthetic_AT_pattern
A [ 1 18 1 18 18 18 ]
C [ 1 1 1 1 1 1 ]
G [ 1 1 1 1 1 1 ]
T [ 18 1 18 1 1 1 ]
'''


def _matrix(name, values, fmt):
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] != 4 or not 3 <= array.shape[1] <= 50:
        raise ValueError('Each DNA motif needs four A/C/G/T rows and 3–50 positions.')
    if not np.isfinite(array).all() or (array < 0).any() or (array > 1e12).any() or (array.sum(axis=0) <= 0).any():
        raise ValueError('Matrix values must be finite, between zero and 1e12, and nonzero in every column.')
    return {'name':name[:120], 'matrix':array.tolist(), 'format':fmt, 'width':array.shape[1]}


def parse_motifs(text):
    """Read labeled JASPAR count matrices or MEME minimal DNA probability matrices."""
    lines = [line.strip() for line in text.lstrip('\ufeff').splitlines() if line.strip() and not line.lstrip().startswith('#')]
    motifs = []
    if not lines:
        raise ValueError('Paste or upload a JASPAR matrix or MEME motif file.')
    if lines[0].startswith('>'):
        name, rows = None, {}
        for line in lines:
            if line.startswith('>'):
                if name is not None:
                    if set(rows) != set(BASES):
                        raise ValueError('Every JASPAR motif needs one row each for A, C, G and T.')
                    motifs.append(_matrix(name,[rows[base] for base in BASES],'JASPAR PFM'))
                name,rows = line[1:].strip(),{}
                if not name:
                    raise ValueError('Give each motif a nonempty header.')
            else:
                match = re.fullmatch(r'([ACGT])\s*\[?\s*([\d.eE+\-\s]+?)\s*\]?',line)
                if not match or match[1] in rows:
                    raise ValueError('Use JASPAR rows such as A [ 1 2 3 ], with no duplicate nucleotide rows.')
                rows[match[1]] = [float(value) for value in match[2].split()]
        if set(rows) != set(BASES):
            raise ValueError('Every JASPAR motif needs one row each for A, C, G and T.')
        motifs.append(_matrix(name,[rows[base] for base in BASES],'JASPAR PFM'))
    else:
        if not lines[0].startswith('MEME version'):
            raise ValueError('Use a labeled JASPAR PFM or a MEME minimal motif file.')
        alphabet = next((line.split('=',1)[1].replace(' ','') for line in lines if line.startswith('ALPHABET') and '=' in line), 'ACGT')
        if alphabet != 'ACGT':
            raise ValueError('MEME matrices must use the DNA alphabet ACGT in that order.')
        index,name = 0,None
        while index < len(lines):
            line = lines[index]
            if line.startswith('MOTIF '):
                name = line[6:].strip()
            elif line.startswith('letter-probability matrix:'):
                width = re.search(r'\bw\s*=\s*(\d+)',line)
                alength = re.search(r'\balength\s*=\s*(\d+)',line)
                if not name or not width or not alength or int(alength[1]) != 4:
                    raise ValueError('Each MEME motif needs a name, alength=4, and a matrix width w.')
                size = int(width[1])
                if not 3 <= size <= 50 or index + size >= len(lines):
                    raise ValueError('MEME widths must be 3–50, followed by exactly that many matrix rows.')
                values = [[float(value) for value in lines[index+offset].split()] for offset in range(1,size+1)]
                if any(len(row)!=4 or any(not math.isfinite(value) or value<0 for value in row) or not math.isclose(sum(row),1,abs_tol=.002) for row in values):
                    raise ValueError('MEME probability rows need four nonnegative A/C/G/T values summing to one.')
                motifs.append(_matrix(name,np.asarray(values).T,'MEME PSPM'))
                name = None
                index += size
            index += 1
        if name is not None or not motifs:
            raise ValueError('The MEME file contains a missing or incomplete probability matrix.')
    if len(motifs)>32:
        raise ValueError('Use at most 32 matrices per CPU run.')
    names = [motif['name'] for motif in motifs]
    if len(names)!=len(set(names)):
        raise ValueError('Give every motif a unique name.')
    return motifs


def scan_pwm(records, motifs, threshold=.8, background=(.25,.25,.25,.25), pseudocount=.1):
    background = np.asarray(background,dtype=float)
    if background.shape!=(4,) or not np.isfinite(background).all() or (background<=0).any() or not np.isclose(background.sum(),1,atol=1e-6):
        raise ValueError('Background requires four positive A,C,G,T frequencies summing to one.')
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Relative score threshold must be between zero and one.')
    if not math.isfinite(pseudocount) or not .0001 <= pseudocount <= 1:
        raise ValueError('Smoothing weight must be between 0.0001 and 1.')
    hits, summaries = [],[]
    encoded = []
    for record in records:
        sequence = record['sequence']
        indices = np.fromiter((BASES.find(base) for base in sequence),dtype=np.int16)
        encoded.append((record,indices))
    for motif in motifs:
        counts = np.asarray(motif['matrix'])
        probabilities = counts/counts.sum(axis=0)
        probabilities = (probabilities + pseudocount*background[:,None])/(1+pseudocount)
        weights = np.log2(probabilities/background[:,None])
        low,high = float(weights.min(axis=0).sum()),float(weights.max(axis=0).sum())
        if high-low <= 1e-12:
            raise ValueError(f"Motif {motif['name']} has no positional preference under the selected background.")
        width = motif['width']
        count,skipped = 0,0
        for record,indices in encoded:
            if len(indices)<width:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(indices,width)
            valid = np.all(windows>=0,axis=1)
            skipped += int((~valid).sum())
            positions = np.flatnonzero(valid)
            clean = windows[valid]
            # Score the reverse complement with its matching background denominator.
            reverse_weights = np.log2(probabilities[::-1,::-1]/background[:,None])
            strands = [('+',weights,low,high)]
            if not np.allclose(weights,reverse_weights,rtol=0,atol=1e-12):
                strands.append(('-',reverse_weights,float(reverse_weights.min(axis=0).sum()),float(reverse_weights.max(axis=0).sum())))
            for strand,matrix,minimum,maximum in strands:
                scores = matrix[clean,np.arange(width)].sum(axis=1)
                relative = (scores-minimum)/(maximum-minimum)
                for offset in np.flatnonzero(relative >= threshold-1e-12):
                    start = int(positions[offset])
                    from core.sequences import revcomp
                    matched = record['sequence'][start:start+width]
                    hits.append({'record':record['id'],'motif':motif['name'],'start':start+1,'end':start+width,
                                 'strand':strand,'matched_sequence':matched if strand=='+' else revcomp(matched),
                                 'score':float(scores[offset]),'relative_score':float(min(1,max(0,relative[offset])))})
                    count += 1
                    if len(hits)>10000:
                        raise ValueError('More than 10,000 motif hits. Increase the threshold or reduce the input.')
        summaries.append({'name':motif['name'],'width':width,'format':motif['format'],'hits':count,
                          'skipped_unknown_windows':skipped,'information_bits':float((probabilities*np.log2(probabilities/background[:,None])).sum())})
    hits.sort(key=lambda hit:(-hit['relative_score'],hit['record'],hit['start'],hit['motif'],hit['strand']))
    return {'hits':hits,'motifs':summaries,'matrices':motifs,'records':len(records),
        'method':'DNA log2-odds scoring on both strands. Relative scores span each strand matrix’s theoretical minimum and maximum; thresholds are not p-values.',
        'settings':{'relative_threshold':threshold,'background_ACGT':background.tolist(),'smoothing_weight':pseudocount},
        'inputs':records}
