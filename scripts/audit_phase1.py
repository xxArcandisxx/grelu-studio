"""Audit Phase 1 evidence using cached data; no model inference or downloads."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_gm12878 import DATA, digest, dnase_label, write_csv, write_json
from core.sequences import revcomp
from core.stats import pearson, spearman


def load(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    import numpy as np
    import pyBigWig
    from Bio import SeqIO
    from sklearn.metrics import average_precision_score, roc_auc_score
    out = ROOT / 'test-results/gm12878-biological/concerns-review'
    out.mkdir(exist_ok=True)
    results = ROOT / 'test-results/gm12878-biological'
    manifest = json.loads((DATA / 'sources.json').read_text())
    metrics = json.loads((results / 'metrics.json').read_text())
    assert digest(DATA / 'sources.json') == metrics['data_manifest_sha256']
    for name in ['dnase.csv', 'dsqtl.csv']:
        assert digest(DATA / name) == manifest['sample_files'][name]
        provenance = json.loads((results / f'{name[:-4]}_provenance.json').read_text())
        assert provenance['input_sha256'] == digest(DATA / name)
    regions, variants = load(DATA / 'dnase.csv'), load(DATA / 'dsqtl.csv')
    dp, vp = load(results / 'dnase_predictions.csv'), load(results / 'dsqtl_predictions.csv')
    assert [r['id'] for r in regions] == [r['id'] for r in dp]
    assert [r['id'] for r in variants] == [r['id'] for r in vp]
    assert all(float(a['observed']) == float(b['observed']) for a, b in zip(regions, dp))
    assert all(int(a['label']) == int(b['observed']) and float(a['abs_gkm_SVM']) == float(b['abs_gkm_SVM']) for a, b in zip(variants, vp))
    y = np.array([float(r['observed']) for r in dp])
    p = np.array([float(r['predicted']) for r in dp])
    v = np.array([int(r['observed']) for r in vp])
    s = np.array([float(r['predicted']) for r in vp])
    g = np.array([float(r['abs_gkm_SVM']) for r in vp])
    assert np.isclose(average_precision_score(v, s), metrics['dsqtl']['average_precision'])
    assert np.isclose(average_precision_score(v, g), metrics['dsqtl']['gkmSVM_average_precision_same_sample'])
    with (DATA / 'source/intervals.bed').open() as stream:
        intervals = [(c, int(a), int(b)) for c, a, b in (line.split()[:3] for line in stream)]
    train = [r for r in intervals if r[0] not in ['chr10', 'chr11']]
    val = [r for r in intervals if r[0] == 'chr10']
    test = [r for r in intervals if r[0] == 'chr11']
    test_set = set(test)
    assert all((r['chrom'], int(r['start']), int(r['end'])) in test_set for r in regions)
    assert set(c for c, _, _ in train).isdisjoint(r['chrom'] for r in regions)
    print('Published chromosome split and paired comparison inputs verified.', flush=True)
    # Check all training windows for exact sequence matches, including reverse complements.
    reference = SeqIO.parse(str(DATA / 'source/hg38.2bit'), 'twobit')
    query = {}
    for row in regions:
        query.setdefault(row['sequence'], []).append(row['id'])
        query.setdefault(revcomp(row['sequence']), []).append(row['id'])
    exact_matches = []
    started = time.monotonic()
    for i, (chrom, start, end) in enumerate(train, 1):
        sequence = str(reference.sequences[chrom][start:end]).upper()
        if sequence in query:
            exact_matches.append({'training_interval': f'{chrom}:{start}-{end}', 'test_ids': query[sequence]})
        if i % 100000 == 0:
            print(f'Exact duplicate audit: {i}/{len(train)} ({time.monotonic()-started:.0f}s)', flush=True)
    reference.stream.close()
    # Estimate a deployment-valid constant from training labels only, independent of test outcomes.
    selected = sorted(random.Random(20261122).sample(train, min(5000, len(train))))
    baseline_rows = []
    with pyBigWig.open(str(DATA / 'source/ENCFF093VXI.bigWig')) as bw:
        for chrom, start, end in selected:
            target = dnase_label(bw.values(chrom, start + 557, end - 557, numpy=True))
            baseline_rows.append({'chrom': chrom, 'start': start, 'end': end, 'label': target})
    write_csv(out / 'training_mean_baseline_sample.csv', ['chrom', 'start', 'end', 'label'], baseline_rows)
    mean = float(np.mean([r['label'] for r in baseline_rows]))
    train_mean_mse = float(np.mean((y - mean) ** 2))
    oracle_mse = float(np.mean((y - y.mean()) ** 2))
    observed_mse = float(np.mean((y - p) ** 2))
    def groups(keys):
        return [np.flatnonzero(np.array(keys) == key) for key in sorted(set(keys))]
    # Grouped resampling avoids treating all neighboring genomic rows as independent.
    dna_groups = groups([f"{r['chrom']}:{int(r['start']) // 1000000}" for r in regions])
    var_groups = groups([r['chrom'] for r in variants])
    rng = np.random.default_rng(20261122)
    draws_d, draws_v = [], []
    for _ in range(2000):
        idx = np.concatenate([dna_groups[i] for i in rng.integers(len(dna_groups), size=len(dna_groups))])
        yy, pp = y[idx], p[idx]
        draws_d.append([pearson(yy, pp), spearman(yy, pp), float(np.mean((yy-pp)**2)),
                        float(np.mean((yy-mean)**2) - np.mean((yy-pp)**2))])
        idx = np.concatenate([var_groups[i] for i in rng.integers(len(var_groups), size=len(var_groups))])
        if len(np.unique(v[idx])) != 2:
            continue
        ap = float(average_precision_score(v[idx], s[idx]))
        gap = float(average_precision_score(v[idx], g[idx]))
        draws_v.append([ap, float(roc_auc_score(v[idx], s[idx])), gap, ap-gap])
    write_csv(out / 'dnase_bootstrap.csv', ['pearson', 'spearman', 'mse', 'mean_baseline_mse_minus_model_mse'],
              [dict(zip(['pearson','spearman','mse','mean_baseline_mse_minus_model_mse'], row)) for row in draws_d])
    write_csv(out / 'variant_bootstrap.csv', ['model_ap', 'model_auroc', 'gkmsvm_ap', 'paired_ap_difference'],
              [dict(zip(['model_ap','model_auroc','gkmsvm_ap','paired_ap_difference'], row)) for row in draws_v])
    def ci(rows, names):
        return {name: [float(x) for x in np.quantile(np.array(rows)[:, i], [.025, .975])] for i, name in enumerate(names)}
    # Randomized rankings provide a finite-sample reference; prevalence is the PR no-skill line, not exact finite-sample expected AP.
    random_ap = [float(average_precision_score(v, rng.permutation(s))) for _ in range(2000)]
    order = np.argsort(-s, kind='stable')
    retrieval = {str(k): {'positives': int(v[order[:k]].sum()), 'precision': float(v[order[:k]].mean()),
                          'recall': float(v[order[:k]].sum()/v.sum())} for k in [5, 10, 25, 50, 100]}
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'seed': 20261122,
             'source_metrics_sha256': digest(results/'metrics.json'),
             'leakage': {'training_regions': len(train), 'validation_regions': len(val), 'test_regions': len(test),
                         'all_500_in_original_chr11_test_set': True, 'training_coordinate_overlap': 0,
                         'exact_training_sequence_matches_including_reverse_complements': exact_matches,
                         'near_duplicate_homology_audit': 'NOT performed; exact-match and chromosome checks cannot exclude all homology leakage.',
                         'variant_split': 'External association benchmark; loci can overlap training chromosomes. No hg19-to-hg38 locus overlap/liftover audit performed.',
                         'checkpoint_training_provenance': 'Published chromosome split reconstructed; original training execution was not independently audited.'},
             'fairness': {'same_1000_variant_ids_labels_and_baseline_scores_verified': True,
                          'positives': int(v.sum()), 'controls': int((1-v).sum()), 'positive_fraction': float(v.mean()),
                          'model_ap': float(average_precision_score(v,s)), 'gkmsvm_ap': float(average_precision_score(v,g)),
                          'ap_difference': float(average_precision_score(v,s)-average_precision_score(v,g)),
                          'baseline_limit': 'Supplied published gkmSVM scores; its training/tuning history not re-audited. Same-row scoring fairness does not prove equal training exposure.'},
             'dnase_baselines': {'training_label_sample': len(selected), 'training_only_mean': mean,
                                 'model_mse': observed_mse, 'training_sample_mean_mse': train_mean_mse,
                                 'model_mse_reduction_vs_training_sample_mean': 1-observed_mse/train_mean_mse,
                                 'test_mean_oracle_mse': oracle_mse,
                                 'oracle_limit': 'Test-mean baseline uses test outcomes; descriptive best possible constant, not a deployable fitted baseline.'},
             'uncertainty': {'replicates_requested': 2000, 'method': 'Paired grouped percentile bootstrap, 95% intervals; approximate and conditional on existing samples.',
                             'dnase_groups': '1 Mb genomic blocks by interval start', 'dnase_group_count': len(dna_groups),
                             'variant_groups': 'whole chromosomes', 'variant_group_count': len(var_groups),
                             'variant_valid_replicates': len(draws_v),
                             'dnase_95_percent_intervals': ci(draws_d,['Pearson','Spearman','MSE','mean_baseline_mse_minus_model_mse']),
                             'variant_95_percent_intervals': ci(draws_v,['model_AP','AUROC','gkmSVM_AP','paired_AP_difference']),
                             'limits': 'Few variant positives, only one DNase chromosome, approximate block choice, and limited chromosome groups. Resampling is not a new independent dataset or a retrained-model comparison.'},
             'random_ranking': {'simulations': 2000, 'mean_AP': float(np.mean(random_ap)),
                                'AP_95_percent_reference_range': [float(x) for x in np.quantile(random_ap,[.025,.975])]},
             'variant_top_k': retrieval}
    write_json(out / 'audit.json', audit)
    for name in ['dnase_baselines','fairness','uncertainty','variant_top_k','random_ranking']:
        print(name, json.dumps(audit[name]), flush=True)


if __name__ == '__main__':
    main()
