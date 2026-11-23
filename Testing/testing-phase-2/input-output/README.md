# Phase 2, Step 1: input and output verification

**Gates 1–5 are implemented separately.** See [Gate 2](gate-2/README.md),
[Gate 3](gate-3/README.md), [Gate 4](gate-4/README.md), and
[Gate 5](gate-5/README.md) for evidence and coverage limitations.
Gate 5 completed with **273 PASS, 0 FAIL, 5 SKIP, 0 BLOCKED**. Browser behavior
remains unverified; prior Gate 3 optional dependency failures are preserved.
Gate 6 and biological output diagnostics have not started. Historical sections
below describe their original run times, not current status.

```bash
.venv/bin/python Testing/testing-phase-2/input-output/verify_gate1.py
```

Read `latest.json` for the most recent timestamped run, then that run's
`REPORT.md` and `results.json`. Every run is retained, including failures;
the command exits nonzero for FAIL/BLOCKED. No expected prediction scores are
hardcoded. The fixture contains model identity expectations only.

## Repository pipeline identified by inspection

| Stage | Actual implementation | Status in Gate 1 |
|---|---|---|
| Startup/checkpoint choice | `run_studio.sh` cache-only pinned Hugging Face lookup; `studio_web.MODELS`, `PROVENANCE`, `resolve_model`, `model_choices`; `model_registry.sync_models` for installed registrations | Pinned launcher-equivalent isolated instance tested; live process not attested |
| Checkpoint load | `core.models.load_model`, `_load_lightning_ckpt`; actual gReLU `LightningModel.load_from_checkpoint` | Real deserialization and per-tensor weight identity tested |
| Inspector/metadata | `workbench.analyze(mode='models')` → `engine.execute_platform_workflow('info')` → `workflows.info.handle_info_workflow` → `model_card`; `/api/model-info` → `studio_services.model_metadata` | Real HTTP requests via Flask client; HTML and embedded records saved |
| Web DNA validation | `studio_web.parse_sequence`, `workbench.parse_batch`, `advanced_views.validate_variants`, `variant_views.validate_pairs` | Identified only; input boundary tests deferred |
| Engine DNA parsing/QC | `core.io._clean`, `read_fasta`, `read_sequences`, `sequence_qc` | Identified only; not verified in Gate 1 |
| Windowing/encoding | `core.sequences.fit_with_offset`, `one_hot`, `revcomp`; gReLU's own `grelu.sequence.format.strings_to_one_hot` in `LightningModel.predict_on_seqs` | No DNA preprocessing executed; independent numerical comparison deferred |
| Inference | `workflows.predict.handle_predict_workflow` → `ModelAdapter.predict` → `_predict_grelu` or `_predict_torch`; adapter can fall back and records a note | No prediction path executed or certified by this identity gate |
| Native model output | `grelu.lightning.LightningModel.forward` → `format_input`, `.model`, `.activation`, `.transform` | Architecture, activation class and transform class recorded; numerical transformations deferred |
| Summary/rounding | `core.models._norm3`; `core.sequences.aggregate`; `workflows.predict._downsample`/track rounding; `workbench.analyze` table formatting (`.6g`); `variant_views.comparisons` deltas | Identified only; raw prediction tensor/rounding equality not tested yet |
| Visualizations | `static/studio.js`, `static/advanced.js`, `static/saturation.js`, templates and numerical records | Not executed in a browser; deferred |
| Exports | `workflows.predict` CSV writer; `static/analysis.js` embedded-record JSON download; `static/studio.js` comparison JSON; `static/advanced.js` and `static/exploration.js` CSV downloads; `experiment_views.record` JSON and `report` HTML, backed by `experiment_store.save_experiment` | Identified only; numerical export correctness deferred |

The run's `source-inventory.json` records source SHA-256 hashes and Python
symbol line numbers. The table above is inspection evidence, not a claim that
unexecuted stages passed.

## Evidence and independence

- `fixtures/model_identity.json`: pinned identity anchor and deliberately
  conflicting length for a model-configuration negative check.
- `raw.json`: safe raw checkpoint metadata and all persisted tensor fingerprints.
- `direct.json`: direct gReLU load without importing website model/sequence code.
- `website.json`: actual website loader observations, model inspector record,
  metadata API bodies and unavailable-selection/configuration checks.
- `model-inspector.html`: real server-rendered inspector response.
- `identity-comparison.json`: direct-versus-website architecture and weight
  identity; **not** prediction-equivalence evidence.
- `environment.json`: runtime versions, resource limits and explicit null
  inference shapes. Actual metadata-route batch sizes are in `website.json`.
- `phase1-before.json` and `phase1-after.json`: existing Phase 1 result hashes.
- Worker logs and peak RSS/elapsed times are preserved in each run.

There are no raw prediction comparisons or DNA fixtures for this gate because
model identity requires neither. These Step 1 deliverables remain explicitly
pending for the later gates, rather than being fabricated or silently skipped.

The direct loader and website loader share only checkpoint deserialization
compatibility/fingerprint utilities in the test runner. No website DNA parsing,
windowing or one-hot encoding feeds an alleged independent numerical test.
Future direct inference must use native gReLU input handling independently;
that has not been implemented here.

CPU workers run sequentially with one thread, a 1.8 GB RSS ceiling and a 650 MB
free-memory reserve; real gReLU imports still need substantial RAM. No network,
training, fine-tuning, weight saving, production registry edits, or Phase 1 edits
are part of these tests. Installed gReLU dependencies may initialize their normal
genomepy cache; a filesystem permission failure is retained in worker logs and
must be resolved explicitly rather than bypassed.

## Initial Gate 1 run

Recorded run: `runs/20261122T121430103785Z`. **FAIL: 18 PASS, 1 FAIL, 2 SKIP.**

- [Detailed verification report](runs/20261122T121430103785Z/REPORT.md)
- [Machine-readable results](runs/20261122T121430103785Z/results.json)
- [Direct/website identity comparison](runs/20261122T121430103785Z/identity-comparison.json)

The correct pinned model and weights passed identity checks. The website
accepted a contradictory declared input length without reporting the mismatch.
No production fix was applied. Live server and browser execution were not
verified. Gate 2 remains unimplemented pending the user's review and next steps.

## Gate 1 configuration remediation — latest completed run

**39 PASS, 0 FAIL, 2 SKIP**, run `runs/20261122T123451791209Z`.

- [Updated report](runs/20261122T123451791209Z/REPORT.md)
- [Machine-readable results](runs/20261122T123451791209Z/results.json)
- [Supplemental startup and existing UI checks](runs/20261122T123451791209Z/supplemental-regressions.json)

The incompatible 200-base declaration now returns an explicit error. Unknown
requirements block compatibility claims. Native workflow windows and
architecture flexibility are recorded separately; there is no global
2,114-base rule. Original failed-run evidence and Phase 1 records remain
unchanged. Gate 2 has not started.
