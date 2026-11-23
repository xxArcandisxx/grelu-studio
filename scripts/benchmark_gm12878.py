"""Reproduce small, fixed GM12878 biological benchmarks without changing the app.

Run with the project Python. ``prepare`` downloads public sources (~2.4 GB);
``evaluate`` uses only local data/checkpoints. See datasets/gm12878/biological/README.md.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from zipfile import ZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'datasets/gm12878/biological'
REVISION = '9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e'
DATA_REVISION = '5b0d0f4527d067e3f50d8f56020b119db8070a4e'
XLSX = 'https://static-content.springer.com/esm/art%3A10.1038%2Fng.3331/MediaObjects/41588_2015_BFng3331_MOESM26_ESM.xlsx'
NOTEBOOK = 'https://github.com/Genentech/gReLU-applications/blob/d98346a34e74a88cc7ad47b5596925e8858c211a/VEP_benchmark/'


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def dnase_label(signal):
    """Exact gReLU target transformation on a central 1,000-base signal array."""
    import numpy as np
    signal = np.asarray(signal)
    if signal.shape != (1000,) or np.isinf(signal).any() or (signal[np.isfinite(signal)] < 0).any():
        raise ValueError('Expected 1,000 nonnegative DNase measurements (or missing bases)')
    return float(np.log1p(np.nan_to_num(signal).sum()))


def allele_sequences(genome, chrom, pos, ref, alt):
    """Match gReLU's 1-based SNV centering for an even 2,114-base window."""
    start = pos - 1057
    if start < 0:
        raise ValueError('Variant too close to chromosome start')
    sequence = str(genome[chrom][start:start + 2114]).upper()
    if len(sequence) != 2114 or set(sequence) - set('ACGTN') or sequence[1056] not in [ref, alt]:
        raise ValueError('Variant coordinate/allele mismatch')
    return sequence[:1056] + ref + sequence[1057:], sequence[:1056] + alt + sequence[1057:]


def download(url, path, md5=None, expected_size=None):
    import requests
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {path.name}', flush=True)
        with requests.get(url, stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with part.open('wb') as stream:
                count, last = 0, time.monotonic()
                for chunk in response.iter_content(1024 * 1024):
                    stream.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last > 20:
                        print(f'  {path.name}: {count / 1e6:.0f} MB', flush=True)
                        last = time.monotonic()
        if expected_size and part.stat().st_size != expected_size:
            raise ValueError(f'Unexpected size: {path}')
        if md5 and digest(part, 'md5') != md5:
            raise ValueError(f'MD5 mismatch: {path}')
        part.replace(path)
    if expected_size and path.stat().st_size != expected_size:
        raise ValueError(f'Unexpected size: {path}')
    if md5 and digest(path, 'md5') != md5:
        raise ValueError(f'MD5 mismatch: {path}')
    return {'url': url, 'file': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': digest(path), 'md5': digest(path, 'md5')}


def xlsx_rows(path):
    """Read the original single-sheet supplement using only the standard library."""
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings = [''.join(t.text or '' for t in item.findall('.//s:t', ns))
                   for item in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        rows = ET.fromstring(archive.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row', ns)
        def cells(row):
            result = {}
            for cell in row:
                column = ''.join(c for c in cell.get('r') if c.isalpha())
                value = cell.findtext('s:v', '', ns)
                result[column] = strings[int(value)] if cell.get('t') == 's' else value
            return result
        headers = cells(rows[0])
        return [{name: cells_row.get(column, '') for column, name in headers.items()}
                for cells_row in map(cells, rows[1:])]


def prepare(args):
    import requests
    from Bio import SeqIO
    import numpy as np
    import pyBigWig
    source = DATA / 'source'
    source.mkdir(parents=True, exist_ok=True)
    manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                'seed': args.seed, 'sources': {}, 'protocol_notebooks': NOTEBOOK}
    inputs = {
        'intervals.bed': (f'https://huggingface.co/datasets/Genentech/GM12878_dnase-data/resolve/{DATA_REVISION}/intervals.bed', None, None),
        'hg38.2bit': ('https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.2bit', None, 835393456),
        'hg19.2bit': ('https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit', None, None),
        'ENCFF093VXI.bigWig': ('https://www.encodeproject.org/files/ENCFF093VXI/@@download/ENCFF093VXI.bigWig', '96d6a8647e744d96738531401108bfe2', 779819333),
        'dsqtl.xlsx': (XLSX, None, None),
    }
    # UCSC publishes checksums alongside each assembly file.
    for genome in ['hg38', 'hg19']:
        url = f'https://hgdownload.soe.ucsc.edu/goldenPath/{genome}/bigZips/md5sum.txt'
        response = requests.get(url, timeout=40)
        response.raise_for_status()
        md5 = next(line.split()[0] for line in response.text.splitlines()
                   if line.split()[-1].lstrip('./') == f'{genome}.2bit')
        u, _, size = inputs[f'{genome}.2bit']
        inputs[f'{genome}.2bit'] = (u, md5, size)
    for name, (url, md5, size) in inputs.items():
        manifest['sources'][name] = download(url, source / name, md5, size)
        write_json(DATA / 'sources.json', manifest)
    regions = []
    with (source / 'intervals.bed').open() as stream:
        for line in stream:
            chrom, start, end = line.split()[:3]
            if chrom == 'chr11':
                row = {'chrom': chrom, 'start': int(start), 'end': int(end)}
                if row['end'] - row['start'] != 2114:
                    raise ValueError('Invalid model window')
                regions.append(row)
    regions = sorted(regions, key=lambda r: r['start'])
    manifest['dnase_test_population'] = len(regions)
    rng = random.Random(args.seed)
    selected = sorted(rng.sample(regions, min(args.regions, len(regions))), key=lambda r: r['start'])
    hg38 = SeqIO.parse(str(source / 'hg38.2bit'), 'twobit')
    genome = hg38.sequences
    with pyBigWig.open(str(source / 'ENCFF093VXI.bigWig')) as bw:
        for row in selected:
            start, end = row['start'], row['end']
            row['id'] = f"chr11:{start}-{end}"
            row['sequence'] = str(genome['chr11'][start:end]).upper()
            if len(row['sequence']) != 2114 or set(row['sequence']) - set('ACGTN'):
                raise ValueError('Invalid reference sequence')
            signal = bw.values('chr11', start + 557, end - 557, numpy=True)
            row['missing_signal_bases'] = int(np.isnan(signal).sum())
            row['observed'] = dnase_label(signal)
    write_csv(DATA / 'dnase.csv', ['id', 'chrom', 'start', 'end', 'sequence', 'observed', 'missing_signal_bases'], selected)
    (DATA / 'dnase.fasta').write_text(''.join(f">{r['id']}\n{r['sequence']}\n" for r in selected))
    hg38.stream.close()
    hg19 = SeqIO.parse(str(source / 'hg19.2bit'), 'twobit')
    genome = hg19.sequences
    variants, excluded, mismatches = [], [], []
    for raw in xlsx_rows(source / 'dsqtl.xlsx'):
        chrom, pos = raw['chrom_hg19'], int(float(raw['pos_hg19']))
        ref, alt = raw['allele1'].upper(), raw['allele2'].upper()
        label = int(float(raw['label']))
        if label not in [-1, 1]:
            raise ValueError('Unexpected variant label')
        if chrom not in [f'chr{i}' for i in range(1, 23)] + ['chrX', 'chrY'] or len(ref) != 1 or len(alt) != 1 or ref == alt or set(ref + alt) - set('ACGT'):
            excluded.append({'id': raw['SNPname2'], 'reason': 'unsupported variant'})
            continue
        # gReLU variants use 1-based positions. Match its shared Enformer edge filter.
        if pos < 98304 or pos + 1 > len(genome[chrom]) - 98304:
            excluded.append({'id': raw['SNPname2'], 'reason': 'chromosome edge'})
            continue
        reference = str(genome[chrom][pos - 1:pos]).upper()
        if reference not in [ref, alt]:
            mismatches.append({'id': raw['SNPname2'], 'reason': 'neither allele matches hg19 reference'})
            continue
        variants.append({'id': raw['SNPname2'], 'chrom': chrom, 'pos': pos,
                         'ref': ref, 'alt': alt, 'label': int(label == 1),
                         'allele1_matches_reference': int(reference == ref),
                         'abs_gkm_SVM': float(raw['abs_gkm_SVM'])})
    write_csv(DATA / 'dsqtl_population.csv', list(variants[0]), variants)
    # Uniform sampling retains the natural class prevalence in expectation; no balancing.
    chosen = random.Random(args.seed).sample(variants, min(args.variants, len(variants)))
    chosen.sort(key=lambda r: (r['chrom'], r['pos']))
    for row in chosen:
        row['reference_sequence'], row['alternate_sequence'] = allele_sequences(
            genome, row['chrom'], row['pos'], row['ref'], row['alt'])
    hg19.stream.close()
    write_csv(DATA / 'dsqtl.csv', list(chosen[0]), chosen)
    browser = DATA / 'website-inputs'
    browser.mkdir(exist_ok=True)
    (browser / 'prediction_12.fasta').write_text(''.join(f">{r['id']}\n{r['sequence']}\n" for r in selected[:12]))
    write_csv(browser / 'batch_variants_24.csv', ['variant_id', 'sequence', 'pos', 'ref', 'alt'],
              [{'variant_id': r['id'], 'sequence': r['reference_sequence'], 'pos': 1057,
                'ref': r['ref'], 'alt': r['alt']} for r in chosen[:24]])
    (browser / 'allele_pair.fasta').write_text(
        f">{chosen[0]['id']}_allele1\n{chosen[0]['reference_sequence']}\n"
        f">{chosen[0]['id']}_allele2\n{chosen[0]['alternate_sequence']}\n")
    manifest.update(dnase_sample=len(selected), dsqtl_population=len(variants),
                    dsqtl_sample=len(chosen), dsqtl_positives=sum(r['label'] for r in chosen),
                    dsqtl_excluded=excluded, dsqtl_reference_mismatches=mismatches,
                    dsqtl_allele_convention='Original allele1 vs allele2; both inserted into hg19 background. Not signed reference-genome effects.',
                    sample_files={name: digest(DATA / name) for name in ['dnase.csv', 'dnase.fasta', 'dsqtl.csv', 'dsqtl_population.csv']})
    write_json(DATA / 'sources.json', manifest)
    print(f"Prepared {len(selected)} chr11 regions and {len(chosen)} variants ({manifest['dsqtl_positives']} dsQTLs)", flush=True)


def evaluate(args):
    os.environ.setdefault('MPLBACKEND', 'Agg')
    import numpy as np
    import psutil
    import torch
    from huggingface_hub import hf_hub_download
    from core.models import load_model
    from core.sequences import revcomp
    from core.stats import pearson, spearman
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
    import matplotlib.pyplot as plt
    from importlib.metadata import version
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / 'sources.json').read_text())
    for name, expected in manifest['sample_files'].items():
        if digest(DATA / name) != expected:
            raise ValueError(f'Benchmark input changed: {name}')
    torch.set_num_threads(args.threads)
    checkpoint = Path(hf_hub_download('Genentech/GM12878_dnase-model', 'model.ckpt', revision=REVISION, local_files_only=True))
    adapter, warnings = load_model({'source': 'checkpoint', 'path': str(checkpoint), 'seq_len': 2114}, 'cpu', batch_size=1, backend='builtin')
    if len(adapter.task_names) != 1 or adapter.out_len != 1:
        raise ValueError('Expected a scalar single-task DNase model')
    started = time.monotonic()
    record = {'checkpoint_sha256': digest(checkpoint), 'checkpoint_revision': REVISION,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'data_manifest_sha256': digest(DATA / 'sources.json'),
              'versions': {p: version(p) for p in ['torch', 'grelu', 'numpy', 'pyBigWig', 'biopython']},
              'threads': args.threads, 'batch_size': 1, 'warnings': warnings,
              'scope': 'Fixed samples; DNase is within-experiment held-out chr11 evaluation. dsQTL loci can overlap model training chromosomes. No wet-lab validation of designs or attributions.'}
    def predict(sequence):
        if psutil.virtual_memory().available < 650_000_000:
            raise MemoryError('Available RAM below 650 MB reserve; rerun later to resume')
        value = float(adapter.predict([sequence])[0, 0, 0])
        if not np.isfinite(value):
            raise ValueError('Non-finite prediction')
        return value
    for name in (['dnase', 'dsqtl'] if args.only == 'both' else [args.only]):
        rows = list(csv.DictReader((DATA / f'{name}.csv').open()))
        fields = ['id', 'observed', 'predicted'] if name == 'dnase' else ['id', 'observed', 'predicted', 'signed_effect', 'abs_gkm_SVM']
        predictions_path = output / f'{name}_predictions.csv'
        provenance_path = output / f'{name}_provenance.json'
        current = {'checkpoint_sha256': record['checkpoint_sha256'], 'input_sha256': digest(DATA / f'{name}.csv'),
                   'protocol': 'single-forward' if name == 'dnase' else 'allele2-minus-allele1; forward/RC mean; absolute effect for ranking'}
        completed = []
        if predictions_path.exists():
            if not provenance_path.exists() or json.loads(provenance_path.read_text()) != current:
                raise ValueError('Cannot resume outputs from a different model/input/protocol')
            completed = list(csv.DictReader(predictions_path.open()))
            if [r['id'] for r in completed] != [r['id'] for r in rows[:len(completed)]]:
                raise ValueError('Resume row mismatch')
        write_json(provenance_path, current)
        with predictions_path.open('a', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if not completed:
                writer.writeheader()
            for index, row in enumerate(rows[len(completed):], len(completed)):
                if name == 'dnase':
                    result = {'id': row['id'], 'observed': float(row['observed']), 'predicted': predict(row['sequence'])}
                else:
                    ref, alt = row['reference_sequence'], row['alternate_sequence']
                    effect = (predict(alt) + predict(revcomp(alt)) - predict(ref) - predict(revcomp(ref))) / 2
                    result = {'id': row['id'], 'observed': int(row['label']), 'predicted': abs(effect),
                              'signed_effect': effect, 'abs_gkm_SVM': float(row['abs_gkm_SVM'])}
                writer.writerow(result)
                stream.flush()
                completed.append(result)
                if (index + 1) % 25 == 0:
                    print(f'{name}: {index + 1}/{len(rows)} ({time.monotonic() - started:.0f}s)', flush=True)
        y = np.array([float(r['observed']) for r in completed])
        pred = np.array([float(r['predicted']) for r in completed])
        if name == 'dnase':
            mse = float(np.mean((pred - y) ** 2))
            # Permutation preserves the model score distribution without using test labels to fit a baseline.
            shuffled = np.random.default_rng(args.seed).permutation(pred)
            metrics = {'samples': len(y), 'Pearson r': pearson(y, pred), 'Spearman rho': spearman(y, pred),
                       'MSE': mse, 'RMSE': mse ** .5, 'MAE': float(np.mean(np.abs(pred - y))),
                       'shuffled_prediction_MSE': float(np.mean((shuffled - y) ** 2)),
                       'shuffled_prediction_Pearson': pearson(y, shuffled)}
            plt.scatter(y, pred, s=9, alpha=.45)
            plt.xlabel('Measured log1p DNase sum, central 1000 bp')
            plt.ylabel('Predicted log1p DNase sum')
            plt.title(f'GM12878 held-out chr11 (n={len(y)})')
        else:
            if len(np.unique(y)) != 2:
                raise ValueError('Need both positive and control variants')
            metrics = {'samples': len(y), 'positives': int(y.sum()), 'positive_fraction': float(y.mean()),
                       'average_precision': float(average_precision_score(y, pred)), 'AUROC': float(roc_auc_score(y, pred)),
                       'gkmSVM_average_precision_same_sample': float(average_precision_score(y, [float(r['abs_gkm_SVM']) for r in completed]))}
            precision, recall, _ = precision_recall_curve(y, pred)
            plt.plot(recall, precision, label=f"GM12878 AP={metrics['average_precision']:.3f}")
            plt.axhline(y.mean(), color='gray', linestyle='--', label='Positive fraction')
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.legend()
            plt.title(f'LCL dsQTL classification (n={len(y)})')
        plt.tight_layout(); plt.savefig(output / f'{name}.png', dpi=160); plt.close()
        write_csv(output / f'{name}_website_benchmark.csv', ['observed', 'predicted'], completed if name == 'dnase' else
                  [{'observed': r['observed'], 'predicted': float(r['predicted']) / (1 + float(r['predicted']))} for r in completed])
        record[name] = metrics
        record['elapsed_seconds'] = time.monotonic() - started
        write_json(output / 'metrics.json', record)
        report = ['# GM12878 biological benchmark', '', record['scope'], '',
                  'Model predictions use the website’s existing ModelAdapter on CPU. No weights were trained or changed.', '',
                  'DNase targets: hg38, chr11, log1p(sum central 1000 bp); missing bigWig bases become zero, matching gReLU.',
                  'dsQTL: original hg19 allele1/allele2 convention, reverse-complement averaging, absolute effect ranking.',
                  'Variant website CSV uses score/(1+score), a monotonic ranking transform, NOT calibrated probabilities. Ignore its calibration/Brier metrics.', '']
        for benchmark in ['dnase', 'dsqtl']:
            if benchmark in record:
                report += [f'## {benchmark}', '', '| Metric | Value |', '|---|---|']
                report += [f'| {k} | {v} |' for k, v in record[benchmark].items()]
                report += ['', f'![{benchmark} plot]({benchmark}.png)', '']
        report += ['## Limits', '', 'This is a sampled evaluation, not a full published benchmark reproduction. Nearby DNase windows can overlap; rows are not independent experimental replicates. QTL labels are associations, not proof of causal effects. No dataset here validates sequence design, multi-variant haplotypes, or attribution mechanisms.', '',
                   f'Protocol source: {NOTEBOOK}', f'Checkpoint SHA256: `{record["checkpoint_sha256"]}`', '']
        (output / 'REPORT.md').write_text('\n'.join(report))
        print(name, json.dumps(metrics), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'evaluate'])
    parser.add_argument('--regions', type=int, default=500)
    parser.add_argument('--variants', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--only', choices=['dnase', 'dsqtl', 'both'], default='both')
    parser.add_argument('--output', type=Path, default=ROOT / 'test-results/gm12878-biological')
    args = parser.parse_args()
    if min(args.regions, args.variants, args.threads) < 1:
        parser.error('Sample sizes and thread count must be positive')
    (prepare if args.phase == 'prepare' else evaluate)(args)


if __name__ == '__main__':
    main()
