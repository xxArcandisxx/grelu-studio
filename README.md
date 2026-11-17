# gReLU Studio

A self-hosted regulatory genomics workbench for sequence prediction, variant analysis, sequence design, and model interpretation.

## Run the review demo

From the checkout on Ubuntu WSL, use the existing environment:

    bash run_studio.sh

Open http://localhost:5000, select **Load example**, and run the comparison. Select an output track and download the JSON run record. The included CNN has random weights. Its output demonstrates the application flow. When the pinned [Genentech GM12878 DNase checkpoint](https://huggingface.co/Genentech/GM12878_dnase-model) is available in the local Hugging Face cache, the launcher also enables it. Select that model before loading an example to run a model-sized input. Synthetic examples verify the workflow. Use aligned hg38 inputs for research runs.

## Host on a lab server

Use the existing Python version and requirements lock and install Flask alongside the engine dependencies. On the lab host, serve the Flask WSGI app with a production server such as Gunicorn behind the lab's HTTPS and access-control proxy. For example, with Gunicorn installed:

    gunicorn --bind 127.0.0.1:8000 --workers 1 --timeout 180 studio_web:app

Set GRELU_HUMAN_CHECKPOINT to an administrator-vetted local checkpoint for a human regulatory model, GRELU_HUMAN_INPUT_LENGTH to its required input length, and GRELU_HUMAN_MODEL_ID to its published name or version. Restart the server whenever the configured checkpoint changes. The JSON export records the model label, checkpoint SHA-256, and engine library versions. The **Load example** button generates a synthetic input sized for the selected model. A checkpoint can execute code during deserialization, so configure only trusted files. The browser submits sequences and a model selection. Prediction outputs use a temporary directory that is removed after inference. Downloaded run records contain the submitted DNA sequences.

The plotted series are model predictions across output bins, optionally averaged into up to 100 displayed bins. The score difference is alternate minus reference, with scores averaged across all model output bins.

The browser includes single-substitution comparison, allele-specific regulation, variant prioritization, multi-allelic comparison, and curated saturation scanning. Each workflow has a dedicated page; saturation scanning requires the pinned local GM12878 checkpoint. Model Zoo entries are disabled catalog references; inference uses only the explicitly selected installed model. The selector shows input lengths and matching test files.

Use Sequence Prediction for one sequence or up to 12 FASTA records, Sequence QC for length and composition checks, and Model Inspector for model metadata. Each completed analysis includes a JSON download.

Research Utilities accepts tabular data for regression or binary evaluation, numerical track summaries and correlations, and basic BED coordinate validation. These pages analyze the provided files without loading a model. Results can be downloaded as JSON.


## Research workspace and local accounts

The responsive workspace includes navigation to every analysis tool, example inputs, result tables, and JSON exports. Run `bash run_studio.sh` and open http://localhost:5000.

Optional researcher accounts start at `/account/register`. Create an account, open the **Local development preview** activation link, confirm activation, and sign in. `/account/forgot` creates a replacement activation link for inactive accounts or a password reset link for active accounts. Links expire after one hour and are single-use. Password resets invalidate existing sign-ins. Analysis tools also remain accessible to guests.

Accounts are stored in the ignored `instance/accounts.sqlite3` database. Passwords are hashed; account forms use CSRF tokens. The ignored `instance/session.key` persists the local session signing key. Set `GRELU_SECRET_KEY` to supply your own stable key and `GRELU_HTTPS=1` when serving over HTTPS.

These account flows are a local development feature: there is no email provider integration. Recovery and activation links are displayed in the browser by default (`GRELU_ACCOUNT_LOCAL_PREVIEW=1`). Do not expose this preview mode on a public server. Setting `GRELU_ACCOUNT_LOCAL_PREVIEW=0` hides links; email delivery must be implemented before using account registration/recovery in a public deployment. Keep the lab's access-control proxy for deployed research data.

Run regression checks with `.venv/bin/python -m unittest discover -s tests -v`.


## Design, batch variants, scans, and attribution

- `/design` optimizes one full-length seed sequence for an on-target output track,
  optionally subtracting an off-target score and restricting editable positions. Results include accepted mutations,
  objective history, before/after track scores, and JSON, CSV, and FASTA exports.
- `/batch-variants` accepts a validated CSV of up to 24 single-base substitutions.
  It reports per-track effects, descriptive ranks, reverse-complement sign
  agreement, optional shuffled-context p-values, and JSON/CSV exports.
- `/mutagenesis` scans up to 24 positions of an uploaded full-length sequence on a
  selected output track, with a substitution heatmap and JSON/CSV exports.
- `/attribution` computes channel-mean-corrected gradient × input, with signed
  contributions, top positions, an optional randomized-weight diagnostic, and
  full-resolution JSON/CSV exports.

All four include model-sized synthetic examples and bounded CPU workloads. The
model selector shows required lengths and the file editor checks FASTA locally.
Existing analysis pages remain available. See [DEMO_GUIDE.md](DEMO_GUIDE.md) for
sample files, a presentation walkthrough, scoring conventions, and scope limits.


## Model development and motifs

- `/training` trains a compact three-layer CNN on `sequence,label` regression CSV
  data. It uses an 80/20 held-out split, Adam, live epoch MSE, and best-checkpoint
  retention. The example learns synthetic GC-content labels.
- `/fine-tuning` adapts an installed model's selected output track to numeric
  labels. Local CPU limits are 256 records, 20 epochs, 4,096-base model windows,
  and 25 million parameters. Trained checkpoints are registered for prediction.
- `/motifs` finds exact DNA motifs on both strands and ranks exploratory k-mer
  enrichment against dinucleotide-shuffled backgrounds. It does not run TF-MoDISco.
- `/model-zoo` installs official catalog checkpoints at pinned Hugging Face
  revisions, inspects their metadata, and adds/removes workspace registrations.
  Downloads require network access and sufficient memory; shared cache files are
  retained when registrations are removed. Large-model installations have not
  been exhaustively verified.

Runs execute in local worker threads and store real statuses, epoch metrics, and
checkpoints under ignored `instance/lab/`. Serve one application worker. Active
jobs do not survive a server restart; complete artifacts and model registrations
persist. These are local research tools, not a distributed training service.

## Saved experiments, PWM scanning, simulations, and comparison

Completed analysis runs are automatically archived under **Experiment history**
(`/experiments`). Each snapshot keeps inputs, settings, model provenance, and
results in ignored `instance/experiments.sqlite3`. Guests see only the history
associated with their browser session. Signing in transfers that guest history
to the account; authenticated history is available after subsequent sign-ins.
Completed training/fine-tuning jobs are archived too. Downloadable JSON snapshots
and standalone HTML reports remain usable after a server restart. Reports embed
SVG plots, result tables, interpretation notes, and the complete numerical record.
Open a report in a browser and use Print / Save as PDF for a printable version.
Deleting a saved experiment does not delete model checkpoints or downloaded files.

- `/pwm-scan` reads labeled JASPAR count matrices or MEME minimal DNA probability
  matrices. It normalizes each column and mixes probabilities with the selected
  positive A/C/G/T background using the smoothing weight. It scans both strands
  using log2 odds, skips N-containing windows, and reports 1-based inclusive
  coordinates. Relative thresholds refer to each strand matrix's theoretical
  score range; they are not statistical p-values. Limits are 32 motifs of 3–50
  positions and 50,000 total sequence bases. Format references:
  [JASPAR documentation](https://jaspar.elixir.no/docs/) and
  [MEME motif specification](https://meme-suite.org/meme/doc/meme-format.html).
- `/motif-simulation` replaces equal-length regions with a primary motif on either
  strand, optionally adds a forward-strand partner at selected gaps, and compares
  predictions with the original background and scrambled-primary controls.
  Sequences keep their length and untouched bases. Up to 12 configurations run on
  a selected output track; full generated sequences are included in the record.
- `/model-comparison` compares up to six sequences on two selected model tracks.
  Strict mode requires exact lengths. Explicit center-crop/N-pad mode records
  every transformed input and offset. Scores, raw differences, and Pearson
  correlation are descriptive because models may use different output scales.
  CPU model windows are limited to 20,000 bases.

All new tools provide JSON/CSV exports and saved HTML reports. Repeat a saved run
with edited settings from its history detail page. The previous snapshot remains
unchanged. New fixtures are under `test-inputs/new-workflows`.

Run the optional cached-checkpoint verification with
`.venv/bin/python tests/smoke_trained.py --fine-tuning`. It checks trained-model
inference, design, attribution, simulations, comparison, pretrained GM12878
fine-tuning, checkpoint reload, and report generation. The fine-tuning run uses
synthetic regression labels to verify execution, not biological model quality.

Saved-run timestamps display in Asia/Kolkata by default. Set
`GRELU_DISPLAY_TIMEZONE` to another IANA timezone name to change report and
history display; stored timestamps remain UTC.


## Loading example files

Every analysis form provides **Load example** and **Download example files**.
Loading fills editable inputs and starter settings without submitting the form,
loading a checkpoint, or creating a worker job. Select the Run/Start button
separately after reviewing the data. Uploaded file selections are cleared when
an example replaces them, so an old upload cannot override the loaded example.

Examples match the selected installed model's required sequence length. Training
provides a synthetic GC-content regression CSV for the chosen input length;
fine-tuning provides eight model-sized labeled records and one starter epoch.
PWM tools supply both DNA and matrix files. Multi-file examples download as ZIP.
Model Inspector, curated scanning, and Model Zoo provide configuration presets
and JSON downloads rather than fabricated input files or model results.

Input-loading-only verification (no inference, training, or browser):
`OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests -p test_example_loading.py -v`.
