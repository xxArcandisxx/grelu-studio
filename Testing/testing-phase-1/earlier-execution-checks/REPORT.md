# GM12878 checks on UCSC hg38 sample

Real model execution through isolated Flask requests; no browser automation. Each feature used a fresh process.
Derived variants and labels are constructed/synthetic as noted; these are execution checks, not biological accuracy results.

| Feature | Status | Seconds | Peak RSS MB |
|---|---|---|---|
| quality | passed | 0.81 | 62.7 |
| prediction | passed | 24.21 | 1299.8 |
| variant-comparison | passed | 11.38 | 1271.6 |
| batch-variants | passed | 13.84 | 1233.3 |
| alleles | passed | 10.15 | 1215.2 |
| prioritization | passed | 10.77 | 1264.6 |
| mutagenesis | passed | 12.19 | 1304.9 |
| attribution | passed | 10.97 | 1328.7 |
| motifs | passed | 0.61 | 63.2 |
| pwm | passed | 0.61 | 64.6 |
| simulation | passed | 9.96 | 1229.0 |
| design | passed | 13.4 | 1313.6 |
| model-zoo | passed | 13.41 | 1401.7 |
| training | passed | 3.66 | 849.8 |
| fine-tuning | passed | 23.97 | 1498.8 |
| model-comparison | passed | 12.18 | 1304.9 |
| evaluation | passed | 1.83 | 177.1 |

## Limits

BED validation is removed from the website. All 435,055 source intervals span 2,114 bases, but only the requested first ten were fetched.
No measured DNase labels, observed variant effects, biological JASPAR motif, or independent second published checkpoint were supplied.
- variant-comparison: Constructed single-base substitution, not an observed variant.
- batch-variants: Three constructed SNVs; shuffled significance testing disabled.
- alleles: Constructed alternate sequences; no observed allele/phasing dataset supplied.
- prioritization: Constructed alternate sequences; no observed allele/phasing dataset supplied.
- motifs: Exact pattern derived from the reference sequence; k-mer enrichment is exploratory.
- pwm: Explicitly synthetic JASPAR-format matrix; no annotated biological motif matrix supplied.
- design: One iteration, two candidate edits; no claim of biological improvement.
- model-zoo: Pinned cache installation tested; remote downloads and other catalogue models not tested.
- training: Labels are synthetic GC fractions, not measured DNase activity. Optimizer/checkpoint execution only.
- training: Compact CNN trained from scratch on center-cropped 2,000-base windows; not training GM12878 from scratch.
- fine-tuning: Labels are synthetic GC fractions, not measured DNase activity. Optimizer/checkpoint execution only.
- model-comparison: Original DNase model versus synthetic-GC fine-tuned checkpoint; different target scales, descriptive comparison only.
- evaluation: Evaluation/track tools checked with bundled synthetic numeric fixtures. BED contains no measured DNase targets; GM12878 accuracy, AUROC, and biological track concordance were not evaluated.

## Recorded run summary

All 17 groups passed. Total sequential phase time: 173.95 seconds. Maximum process-tree RSS: 1498.8 MB. Lowest observed available memory: 1223.8 MB. Isolated persisted snapshots: 27.
