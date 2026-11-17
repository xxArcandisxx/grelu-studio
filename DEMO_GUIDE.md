# gReLU Studio demonstration

Start the app with `bash run_studio.sh`, then open http://localhost:5000.
Restart a previously running server to pick up the new routes and Python changes.

The launcher enables the pinned Genentech GM12878 DNase checkpoint when it is
already cached locally. Demo CNN remains available for quick checks. A Model Zoo
catalog entry is not an installed model and cannot be selected for inference.

## A short, repeatable presentation

1. **Sequence Prediction**: choose **Demo CNN · random weights**, upload
   `test-inputs/quick-demo-200bp/06_Sequence_Prediction_three_records.fasta`, and run.
   Show three records across three output tracks. Enable shuffled inputs to show
   baseline comparisons. Explain that the demo model has random weights.
2. **Batch variants**: choose Demo CNN and upload
   `test-inputs/new-workflows/demo-200bp-variants.csv`. Run with zero shuffles for
   speed or five for a small illustrative background comparison. Show the ranked
   variants, expand all tracks, and download JSON and CSV. These are three supplied
   SNVs, not an exhaustive search or clinical ranking.
3. **Sequence design**: choose Demo CNN and upload
   `test-inputs/new-workflows/demo-200bp-seed.txt`. Select an on-target track and
   optionally a different off-target. Start with four iterations and eight
   candidates. Optionally restrict edits to positions 81–120. Show the accepted
   edits, objective history, track scores, and FASTA export. A run can accept no edits when sampled candidates do not improve the
   objective; this is a legitimate result.
4. **Sequence mutagenesis**: upload the same seed, choose a track, and scan
   positions **101–108**. Show 24 mutant deltas, the reference score, and exports.
   This page works on uploaded full-length sequences. The separate **Saturation
   Mutagenesis** page retains its curated hg38 loci and checkpoint requirement.
5. **Gradient attribution**: upload the same seed, choose a track, and display
   positions **51–150**. Show signed contributions and the top-position table.
   Enable the randomized-weight check for a diagnostic correlation. Export CSV
   or JSON to inspect contributions across the full input sequence.
6. **Research tools**: run built-in benchmark, track, and BED examples. Show the
   downloadable results and validation messages.
7. **Accounts**: demonstrate registration, local activation, and sign-in if needed.
   Activation/recovery links are local previews; email delivery is not configured.

## A trained-model demonstration

Select **Genentech GM12878 DNase** under installed models and use
`trained-2114bp-seed.txt` or `trained-2114bp-variants.csv` from
`test-inputs/new-workflows`. For an initial CPU run, use one design iteration with
two candidates, a two-position mutagenesis region (1058–1059), or three batch
variants with zero shuffles. Increase the workload only after timing the machine.
Gradient attribution also supports the trained checkpoint; choose a 100-base display
region such as 1008–1107.
These test sequences are synthetic, including the 2,114-base examples. For
biological analysis, provide aligned sequences from the model's training assembly.

## Input and scoring conventions

- Sequence length must match the selected model. The editor checks uploaded
  FASTA before submission and identifies the matching test-input directory.
- Batch CSV requires exactly `variant_id,sequence,pos,ref,alt`. Positions are
  1-based inside each full reference sequence. IDs must be unique. Only distinct
  single-base A/C/G/T substitutions are accepted. Every row is validated before
  inference; the browser does not silently crop sequences or skip invalid rows.
- Batch effects are alternate minus reference, averaged across output bins.
  Variants are ranked by the absolute effect on their most affected output track.
  Different tracks can have different scales; the rank is descriptive.
- Background p-values are empirical shuffled-context comparisons. With five
  shuffles the smallest possible p-value is 1/6; this does not establish
  statistical significance. The shuffler preserves dinucleotide counts, but does
  not claim uniform sampling. P-values are not adjusted for multiple testing.
- Attribution uses gradient × input with channel-mean correction. Signed values
  describe local model sensitivity, not mutation effects or biological causality.
  The randomized check reports correlation with a randomized-weight model; it is
  a diagnostic rather than a pass/fail certificate.
- Design maximizes on-target score minus optional off-target score. Only improving
  sampled substitutions are accepted. It uses a fixed seed and stops after the
  iteration/patience limit. Optional editable boundaries keep other bases fixed.
  The same model optimizes and scores the sequence.
- Uploaded-sequence scans are limited to 24 positions per CPU run. Unknown N
  positions are skipped explicitly. Coordinate positions refer to the input
  sequence, not genomic coordinates.
- JSON exports contain submitted sequences, settings, model provenance, and
  numerical results. CSV exports contain design history, full variant effects,
  scan deltas, or full-resolution attribution contributions. FASTA export contains the designed sequence.

## Implemented demo scope

Working pages cover sequence prediction, QC, model inspection, aligned variant
comparison, allele-specific regulation, prioritization, multi-allelic comparison,
curated scanning (with its checkpoint), uploaded-sequence scanning, sequence
design, gradient attribution, batch SNV analysis, benchmark evaluation, track
analysis, BED validation, and local account flows.

This is a focused workbench, not a complete implementation of the gReLU library.
Training and fine-tuning pages now support bounded CPU regression experiments.
Motif analysis supports exact pattern scanning and exploratory k-mer enrichment.
Model Zoo management installs official catalog checkpoints at pinned revisions.
Automated genome-context extraction, TF-MoDISco, durable distributed scheduling,
and production email delivery remain outside this demo scope.

## Verification

Run `.venv/bin/python -m unittest discover -s tests -v`.
Tests check account flows, page rendering, example runs, model selection, rejected
inputs, design history reconstruction, dinucleotide preservation, and consistency
of batch/scan effects with the existing pair-comparison endpoint.

Optional offline trained-checkpoint smoke test:
`.venv/bin/python tests/smoke_trained.py`. It requires the pinned checkpoint in
the local Hugging Face cache and exercises the model-backed analysis workflows; add --fine-tuning to check pretrained adaptation. The same
workflows passed against both Demo CNN and the trained checkpoint during this
implementation. JavaScript syntax and export handlers were also checked; a full
browser visual review was unavailable in this environment.


## Model development demonstration

Open **Model training** and run the example for two epochs. The input is synthetic
GC-content regression data. Watch actual epoch losses, download the checkpoint,
and find the registered model in **Model Zoo**. Select it in **Fine-tuning**, run
one example epoch, then choose the resulting model in Sequence Prediction.

For motifs, open **Motifs** and run the example to see exact pattern coordinates
and exploratory enriched words. Discovery here is sequence-level k-mer enrichment,
not attribution-based TF-MoDISco. Model Zoo downloads run only when Install is
selected and can fail visibly when network access, memory, or metadata is missing.

## Motifs, simulations, comparisons, and saved reports

1. **PWM motif scanning**: upload `pwm-regions.txt` and `pwm-motifs.jaspar` from
   `test-inputs/new-workflows`. Keep the relative threshold at 0.8. Show the
   approximate matches, strand, coordinates, and log-odds score. Repeat with
   `pwm-motifs.meme` to demonstrate another matrix format. These are synthetic
   matrices; the score threshold is not a p-value.
2. **Motif simulations**: choose Demo CNN and upload
   `motif-simulation-background.txt`. Use primary `TATAAA`, starts `81,101,121`,
   both orientations, and scrambled controls. Show the score deltas. To explore
   spacing, use one start such as `81`, partner `GATA`, and gaps `0,5,10`; keep
   the total configuration count at or below 12.
3. **Model comparison**: choose Demo CNN and the installed trained GM12878 model,
   upload `trained-2114bp-comparison-records.txt`, and explicitly choose center
   crop / N-pad handling. Show each model's track, scores, transformed inputs,
   and descriptive correlation. Different model scales do not make raw score
   differences biological effect sizes. Comparing a model with itself is also
   supported as a reproducibility check and produces identical scores.
4. **Experiment history**: reopen one of these runs. Show its stored input and
   settings, edit a setting, and repeat it. Both results should appear separately.
   Demonstrate search and workflow filtering.
5. **Reports**: download an HTML report. It contains its own styles and SVG plots,
   so it opens without the running application or an internet connection. Use
   the browser's Print command to save a PDF. Download the JSON for full results;
   long report tables show the first 200 rows.

Pretrained GM12878 fine-tuning has now been exercised on a small synthetic
regression dataset, followed by checkpoint reload and prediction. This validates
the workflow; it is not an evaluation of biological generalization. The optional
smoke command is `.venv/bin/python tests/smoke_trained.py --fine-tuning`.


## Load first, run separately

Use **Load example** on any analysis page to populate its form, including Model
training and Fine-tuning. Review the data and settings, then select the Run/Start
button. Loading never starts a calculation. **Download example files** saves the
same input files; multi-file examples arrive in a ZIP archive.
