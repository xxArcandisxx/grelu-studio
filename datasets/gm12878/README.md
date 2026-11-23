# hg38 reference-sequence sample for GM12878 model checks

For measured-target prediction and published dsQTL evaluation, see
[`biological/README.md`](biological/README.md). That benchmark uses held-out
chr11 regions and real ENCODE DNase measurements; the ten-record sample below
remains an execution fixture.

`intervals.bed` contains 435,055 supplied intervals. Every interval spans 2,114
bases. The converter fetched **only the first ten**, matching the requested
`test_10` sample. The BED file itself does not encode its assembly; hg38 was
selected as requested.

## Files

- `test_10.bed`: the ten selected intervals, unchanged from the source.
- `test_10.fasta`: reference DNA retrieved from UCSC hg38, forward strand.
- `test_10.metadata.json`: coordinates, retrieval time, request URLs, hashes,
  sequence lengths, and unknown-base counts.
- `derived-tests/`: constructed variants, sequence subsets, and synthetic labels
  for feature execution checks; see that folder's README before interpreting them.

The FASTA contains ten distinct 2,114-base records, 21,140 total bases, and no N
bases. It represents hg38 reference sequence at these locations, not sequenced
GM12878-specific haplotypes, measured DNase coverage, or observed variant effects.

## Reproduce conversion

From the repository root, with the existing environment:

```bash
.venv/bin/python bed_to_fasta.py
```

`requests` is already installed in the project environment. The converter validates
BED windows and returned DNA, waits at least one second between requests, retries
transient network failures, and writes the final FASTA after all ten fetches
succeed. BED starts are zero-based and ends are exclusive.

For another small sample, use `--offset`, `--limit`, and `--output`. Large-scale
conversion should use a local reference genome rather than hundreds of thousands
of API requests. UCSC's API documentation recommends approximately one request
per second: https://genome.ucsc.edu/goldenPath/help/api.html

## Feature execution checks

```bash
.venv/bin/python tests/run_gm12878_sample.py --output test-results/gm12878-verified
```

The runner uses fresh sequential processes, isolated accounts/history/model
registrations, a 1,700 MB RSS ceiling, and an 850 MB available-memory reserve. It
does not control the running website or launch browser automation. gReLU needs
write access to its normal genomepy cache as well as the project test-output path.

Results and resource measurements are recorded incrementally in the output
folder, with `REPORT.md`, `summary.json`, per-feature JSON, logs, and standalone
HTML reports. Synthetic GC labels exercise training/fine-tuning only; they do
not evaluate the published DNase prediction task. Benchmark/track utilities use
explicitly synthetic bundled numeric fixtures because measured labels were not
provided. Model comparison uses the original checkpoint and the synthetic-target
fine-tuned checkpoint if that step completes. Cached Model Zoo registration is
tested without fetching other large models.

In the website, select **Genentech GM12878 DNase** and upload `test_10.fasta` in
Sequence Prediction. Model-dependent pages require 2,114 bases per record.

## Completed sample verification

The recorded run in `test-results/gm12878-verified/REPORT.md` passed all 17
feature groups. GM12878 predicted all ten reference sequences. Fine-tuning
performed eight optimizer steps at batch size one and reloaded the checkpoint
for prediction; training likewise performed eight steps using the compact CNN.
Saved history, ownership checks, real background-job status, JSON exports, and
standalone reports were exercised. Peak process-tree RSS was 1498.8 MB.
These are request-level execution checks, not browser or biological-accuracy
validation. The earlier sandbox attempt failed on a genomepy cache-lock
permission and is retained separately in `test-results/gm12878-sandbox-attempt/`.
