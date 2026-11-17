# Test gReLU Studio

Open the current local preview. If you launch the project yourself with
`bash run_studio.sh`, the default address is http://127.0.0.1:5000/variants.

Choose **Genentech GM12878 DNase** and use files from **trained-model-2114bp**.
For a faster check of the forms and results, choose **Demo CNN ? random weights**
and use the matching names in **quick-demo-200bp**. The FASTA sequences in this
kit are synthetic. Score values depend on the model.

| # | Website page | What to do and expect |
|---|---|---|
| 01 | Predicted Regulatory Effects | Upload **01a** as Reference and **01b** as Alternate. Expect one substitution at position 1058 on the trained model, or 101 on the demo, plus a score difference and diagnostics. |
| 02 | Allele-Specific Regulation | Upload **02** as one FASTA file. Expect one comparison per output track, with two differences. |
| 03 | Variant Prioritization | Upload **03**. Expect two ranked alternates per track, ranked by the absolute score change. On the same page, you can also upload **02** to check that a two-substitution alternate is rejected. |
| 04 | Multi-Allelic & Combinatorial | Upload **04**. Expect three alternates per track, with 1, 1, and 2 differences from the reference. |
| 05 | Saturation Mutagenesis | See **05_Saturation_Mutagenesis_no_upload.md**. Select a bundled locus and expect 24 scores. Uses the trained checkpoint. |
| 06 | Sequence Prediction | Upload **06**. Expect three records per track. Run again with **Compare with five shuffled inputs** to show the baseline columns. |
| 07 | Sequence QC | Upload **07**. Expect one short record, one full-length record with zero unknown bases, and one full-length record with four unknown bases. |
| 08 | Model Inspector | See **08_Model_Inspector_no_upload.md**. Select a model and check the input size and tracks. |
| 09 | Research Utilities ? Benchmark Evaluation | Use **09a** with **Regression** and expect MAE 1, RMSE 1, and both correlations 1. Use **09b** with **Binary classification** and expect AUROC 1 and average precision 1. |
| 10 | Research Utilities ? Track Analysis | Upload **10**. Expect correlation +1 between A and B, and ?1 between A and C. |

Use **Download JSON** on any completed result to check the export.

The former BED-validation fixture (11) is retained for future genome-aware input
work. BED Validation has been removed from the current website and active
acceptance manifest; it is not part of the current demonstration.

## Dataset overview

- FASTA/DNA: `new-workflows/dataset-overview.txt` — four records, 30 bases,
  three unique sequences, one extra duplicate record, six N bases.
- Labeled CSV: `new-workflows/training-gc-regression.csv` — 40 records with
  synthetic GC-fraction labels.

Choose the matching Input format on `/research/dataset`, then upload the file
or select Load example. This tool runs descriptive checks without loading a model.
