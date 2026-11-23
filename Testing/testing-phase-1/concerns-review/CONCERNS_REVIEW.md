# Phase 1: review of the five scientific concerns

Date: 2026-11-22 (UTC). This is an additional audit of the saved Phase 1
predictions, not new model training or a new independent experimental dataset.
Original results and the original archive remain unchanged.

## Verdict

DNase prediction is good on the tested chromosome-11 sample and substantially
outperforms a meaningful constant baseline. Variant ranking is promising given
the rare positives. The apparent advantage over gkmSVM is **not yet established
reliably**: the paired uncertainty interval includes zero.

## 1. Data leakage

- Reconstructed the published chromosome split from all 435,055 source intervals:
  390,473 training regions (excluding chr10/chr11), 21,987 validation regions
  (chr10), and 22,595 test regions (chr11).
- All 500 evaluated DNase intervals belong to the original chr11 test set.
  Training chromosomes are disjoint, so genomic coordinate overlap with these
  test windows is zero under the published split.
- Compared all 390,473 training-window sequences against the 500 test-window
  sequences, including reverse complements. Exact matches are recorded in
  `audit.json`; the audit found none.
- **Still unresolved:** near-duplicate sequences and cross-chromosome homology
  were not searched. These checks reconstruct the documented split; they do not
  independently prove the checkpoint's original training execution followed it.
- The dsQTL test is an external association benchmark whose loci can overlap
  training chromosomes. Its hg19 coordinates were not lifted to hg38 for a
  locus-by-locus overlap audit. Do not describe it as a chromosome-held-out
  variant benchmark.

## 2. Positive variant count and practical retrieval

There are **16 positives and 984 controls**, or **1.6% positives**. Labels indicate
experimentally associated dsQTLs, not proven causal effects for every variant.
The PR no-skill horizontal line is 0.016. With only 16 positives, finite-sample
average precision from a random ranking need not equal prevalence exactly:
2,000 randomized rankings had mean AP **0.0227** and a central 95% reference
range of **0.0123–0.0623**. This reference range is not an uncertainty interval
for the model's AP.

Model AP **0.3195** is encouraging in this context. It is not 32% accuracy, and
AUROC 0.8367 is not 83.67% classification accuracy.

| Number of top-ranked variants inspected | Positives retrieved | Precision | Recall of the 16 positives |
|---|---|---|---|
| 5 | 4 | 80% | 25% |
| 10 | 4 | 40% | 25% |
| 25 | 4 | 16% | 25% |
| 50 | 6 | 12% | 37.5% |
| 100 | 10 | 10% | 62.5% |

These are descriptive results on this sample. The cutoffs were inspected after
evaluation; they are not independently validated operating thresholds.

## 3. Baseline fairness

Verified all 1,000 variant IDs, labels, and supplied absolute gkmSVM scores
against the saved input records. Both AP values were recomputed using the same
metric implementation and rows:

- GM12878: **0.319543**.
- Supplied gkmSVM scores: **0.263276**.
- Observed difference: **+0.056267 AP**, about 5.63 percentage points or 21.37%
  relative improvement using unrounded values.

This establishes same-row scoring fairness. The original gkmSVM training and
tuning history was not re-audited, so equal training exposure is not established.
This is a pretrained Genentech model evaluated through the website, not a new
model that learned during this testing phase.

## 4. Uncertainty

Computed 2,000 paired grouped bootstrap replicates with seed 20261122. DNase
resampling used 117 occupied 1-Mb blocks, assigning intervals by their start;
variant resampling used 22 chromosome groups. Measured targets, model scores,
and baseline scores were always resampled together. The following are
approximate central 95% percentile intervals, conditional on the existing data:

| Metric | Point estimate | Approximate 95% interval |
|---|---|---|
| DNase Pearson | 0.785 | 0.701–0.853 |
| DNase Spearman | 0.780 | 0.728–0.825 |
| DNase MSE | 0.571 | 0.395–0.794 |
| Variant AP | 0.320 | 0.110–0.557 |
| Variant AUROC | 0.837 | 0.727–0.935 |
| gkmSVM AP | 0.263 | 0.046–0.552 |
| Paired model AP minus gkmSVM AP | +0.056 | **−0.077 to +0.202** |

The variant intervals are wide. Because the paired difference includes zero,
this sample does not establish reliable superiority over gkmSVM. Resampling is
not independent replication or repeated model retraining. Few positives, only
one DNase test chromosome, the chosen genomic block size, and limited chromosome
groups all limit the inference; block boundaries do not eliminate all dependence.

## 5. DNase mean-value baseline

Sampled 5,000 training intervals only, uniformly with fixed seed 20261122.
Extracted targets with exactly the original central-1,000-base `log1p(sum)`
convention. No chr10/chr11 labels were used to fit this baseline.

| Predictor | MSE on the same 500 test regions |
|---|---|
| GM12878 model | **0.570839** |
| Constant mean estimated from 5,000 training labels | **1.461374** |
| Best possible constant fitted to these test labels (descriptive oracle) | 1.460351 |
| Previously recorded shuffled predictions | 2.190388 |

The training-only mean is 4.219563 on the transformed target scale. The model's
MSE is about **60.94% lower** than this estimated training-mean baseline. The
paired grouped interval for baseline MSE minus model MSE is **0.669–1.137**,
conditional on that fixed estimated mean. Training-mean estimation uncertainty
was not additionally resampled. The test-mean oracle is reported only as the
lowest achievable MSE for any constant on this particular test sample; it uses
test labels and is not a deployable fitted baseline.

## What remains before stronger claims

1. Evaluate the full eligible dsQTL population or another prespecified larger
   sample, and obtain independent experimental replication.
2. Audit sequence similarity/homology beyond exact duplicates and, if needed,
   perform a build-consistent variant-locus overlap audit.
3. Keep claims limited to DNase activity prediction and dsQTL ranking. These
   tests do not validate sequence design, attribution mechanisms, haplotype
   combinations, new cell types, or clinical utility.

## Evidence and reproduction

- `audit.json`: precise results, checks, methods, and limitations.
- `training_mean_baseline_sample.csv`: all 5,000 training-only baseline labels.
- `dnase_bootstrap.csv` and `variant_bootstrap.csv`: all bootstrap draws.
- Original results: `../REPORT.md`, `../metrics.json`, and prediction CSVs.
- Run from the repository root: `.venv/bin/python scripts/audit_phase1.py`.
  The script uses cached files, performs no model inference or training, and
  needs no network access. This narrative reports this audit run; the script
  regenerates the machine-readable audit and supporting CSVs.

Method references:

- [scikit-learn average precision definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [SciPy explanation of paired resampling and percentile bootstrap intervals](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

The grouped bootstrap here is implemented in the audit script; it does not call
SciPy's bootstrap function or claim that the grouping assumptions are guaranteed.
