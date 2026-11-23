# Real-data GM12878 evaluation

These benchmarks evaluate the published checkpoint against measured biological
data. They are separate from `derived-tests/`, whose labels are synthetic.
Nothing here changes website settings, registered models, or trained weights.

## Run

From the repository root, using the existing environment:

```bash
.venv/bin/python scripts/benchmark_gm12878.py prepare --regions 500 --variants 1000 --seed 42
.venv/bin/python scripts/benchmark_gm12878.py evaluate
.venv/bin/python scripts/verify_gm12878_website.py
```

Preparation downloads roughly 2.4 GB of public files into `source/`: hg38 and
hg19 UCSC 2bit references, the ENCODE DNase bigWig, original dsQTL supplement,
and revision-pinned Genentech intervals. Existing downloads are reused. UCSC
reference files and ENCODE signal are checked against source MD5 checksums;
SHA256 hashes, sample selection, and exclusions are recorded in `sources.json`.
The evaluator uses the existing, locally cached revision-pinned checkpoint.
No new packages are required. Preparation requires network access; evaluation
is local apart from any initialization performed by installed gReLU dependencies.

Evaluation uses the website's existing `core.models.ModelAdapter`, batch size
one, and two CPU threads. It checks for a 650 MB free-memory reserve before
each prediction. Predictions are flushed incrementally and can resume after an
interruption if the input and model hashes match. Choose a different `--output`
directory when changing sample size or protocol. Results default to
`test-results/gm12878-biological/`, including `REPORT.md`, `metrics.json`, plots,
per-record predictions, and two-column CSVs for the website Benchmark page.
The final command verifies prediction, published-variant comparison, batch
variants, and both Benchmark modes through isolated Flask requests. It compares
website numerical results with the evaluation outputs and saves rendered HTML.
It does not automate a browser or modify your active account/history.

`website-inputs/` contains a 12-record prediction FASTA, a 24-row Batch Variants
CSV, and a two-allele FASTA, all derived from the measured-data samples and sized
for browser limits. The allele-pair FASTA can exercise allele comparison or
variant tools. These are real published variants, unlike `derived-tests/`.

## DNase activity prediction

- Input: uniformly sampled intervals from **chr11 only**, the original test
  chromosome; validation is chr10. Sampling is fixed by seed 42 and does not
  inspect labels. Nearby windows may overlap and are not independent replicates.
- Reference: **hg38**, 2,114 bases per interval, forward reference sequence.
- Ground truth: ENCODE experiment
  [ENCSR000EJD](https://www.encodeproject.org/experiments/ENCSR000EJD/), signal
  [ENCFF093VXI](https://www.encodeproject.org/files/ENCFF093VXI/).
- Label: `log1p(sum(signal[start+557:end-557]))`, covering the central 1,000
  bases. Missing bigWig bases become zero, matching gReLU's reader.
- Metrics: Pearson, Spearman, MSE, RMSE, MAE, plus a fixed shuffled-prediction
  baseline. The baseline is not fitted to test labels.
- Website: upload `dnase_website_benchmark.csv` and select regression.

This is held-out evaluation within the original experiment, not a new-cell-type
generalization study. A small sample need not reproduce the published full-test
metrics exactly. A Pearson correlation of 0.785 does **not** mean "78.5% accuracy";
it describes co-variation of measured and predicted values. MSE/RMSE here are on
the transformed `log1p` target scale, not raw coverage units. Inspect residuals
and the scatter plot as well as correlation. The generated FASTA can be split into at most 12 records per
upload on Sequence Prediction. Dataset Overview's 200,000-base limit also means
the complete 500-sequence sample cannot be uploaded there in one request.

## Variant effects

- Source: original supplementary table used by Genentech, from
  [Lee et al., 2015](https://doi.org/10.1038/ng.3331), with experimentally
  associated LCL dsQTLs and control variants; the source download URL is in the
  script and manifest.
- Genome: **hg19**, positions treated as **1-based**, matching the original
  gReLU notebook. Do not extract these coordinates from hg38.
- Shared published filtering: single-base standard alleles, autosomes/X/Y,
  chromosome-end exclusion of 98,304 bases (for comparison with Enformer).
- Reference checking: each locus must match one of the two original alleles;
  any mismatches are recorded. The original allele1/allele2 order is preserved
  rather than silently reversing scores. Both are placed in the same hg19
  background, with the variant at zero-based index 1056 of the 2,114-base window.
- `dsqtl_population.csv` retains the eligible full population; `dsqtl.csv` is a
  uniform sample, without balancing, selecting by score, or choosing positives
  preferentially. The sample prevalence can differ from the full population.
- Score: allele2 minus allele1 after averaging forward and reverse-complement
  predictions. Rank by absolute score. Evaluate average precision (the paper's
  AUPRC convention) and AUROC, plus the supplied gkmSVM baseline on the same rows.
- The original table does not supply signed experimental effect sizes; this
  benchmark assesses association ranking, not effect-direction accuracy.
- Website: the binary Benchmark page only accepts scores in [0,1]. Its export
  uses `abs(effect)/(1+abs(effect))`, which preserves ranking but **does not
  produce probabilities**. Only AUROC and average precision are meaningful for
  this export; ignore calibration and Brier results.

The dsQTL experiment is separate from the training assay, but its loci can
overlap chromosomes/regions seen during sequence-model training. This is the
published external association benchmark, not a chromosome-held-out variant
test. Few positives in a small sample make average precision unstable. Full
population evaluation is needed before comparison to published numbers.

## Scope of evidence

These checks establish execution and quantify two specific prediction tasks.
They cannot establish biological accuracy for sequence design, motif mechanisms,
multi-variant haplotypes, or attribution. Those need independent experiments.
Poor accuracy is a valid result and is never relabeled as a passing benchmark.

Protocol sources:

- [gReLU paper and methods](https://www.nature.com/articles/s41592-025-02868-z)
- [Pinned original analysis notebooks](https://github.com/Genentech/gReLU-applications/tree/d98346a34e74a88cc7ad47b5596925e8858c211a/VEP_benchmark)
- [Checkpoint](https://huggingface.co/Genentech/GM12878_dnase-model)
- [Intervals](https://huggingface.co/datasets/Genentech/GM12878_dnase-data)
