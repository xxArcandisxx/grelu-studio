# Gate 4: inference boundary and independent reference

This gate checks numerical invocation of a pretrained model. It does not check
biological accuracy, calibrate predictions, train weights, or certify processed
application scores/displays/exports. Earlier Gate 3 optional inverse failures
remain external compatibility findings and are never relabeled by this suite.

## Pinned model and configuration

The model is `Genentech/GM12878_dnase-model`, revision
`9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e`, cached `model.ckpt`, SHA-256
`f266b296f49b4e97e3ac52e94594da0aaae1d467778ea58b421e1ee7c4482bea`.
The checkpoint-defined Studio window is 2,114 bases and task 0 is
`ENCFF093VXI`. The architecture is `grelu.model.models.DilatedConvModel`.
The model's convolutional width flexibility is distinct from this verified
workflow window; no other model is assigned a universal input length.

The selected website key is `human`, with the real cached checkpoint passed
through the normal startup environment. The loader factory and registries are
not substituted with fake adapters or cached test doubles. The isolated app
uses run-specific databases and a run-specific lab registry. No production
database, registry, account or server is changed.

## Exact website call paths

| Stage | Actual source / functions | Captured boundary |
|---|---|---|
| Request | `studio_web.compare` (`/api/compare`); `workbench.analyze` (`/predict`, allele-specific, prioritization, multi-allelic); `advanced_views.run`; `exploration_views.run`; `curated_scan.saturation` | Real Flask test-client requests, parsing, validation, routing and responses |
| DNA validation/preparation | `core.dna_inputs.normalize_dna`, `parse_records`; `core.io.read_sequences`; `workbench.parse_batch`; `core.sequences.fit_with_offset`, `revcomp` | Actual reader IDs and DNA, complete ordered strings supplied to `ModelAdapter.predict` |
| Model load | `core.models.load_model`, `_load_lightning_ckpt` → native `LightningModel.load_from_checkpoint` | Actual selected path, spec, backend, tasks, workflow window and loaded state hash before/after calls |
| Shared native prediction | `ModelAdapter.predict` → `_predict_grelu` → `iter_sequence_batches` → native `predict_on_seqs` | Requested batch size, actual batch dimensions, native-versus-fallback status |
| Native conversion and execution | installed `grelu.sequence.format.strings_to_one_hot`; `LightningModel.predict_on_seqs` → CPU/eval transfer → `LightningModel.forward` → `format_input` → `.model` → `.activation` → `.transform` | Full real forward return, immediately after native model activation/transform and before any Studio `_norm3` |
| Shared builtin prediction | `ModelAdapter._predict_torch` → Studio `one_hot` → `torch.from_numpy(...).to(cpu)` → `LightningModel.__call__`/`forward` | Same native forward return; builtin path uses `torch.inference_mode()` |
| Application normalization | `core.models._norm3`; adapter `np.concatenate` | Copies before/after normalization and complete adapter return, before workflow aggregation |
| Application postprocessing, outside this gate | `core.sequences.aggregate`, `workflows.predict._downsample`, track rounding, variant deltas, selected-track means, tables and download builders | Natural response generation may execute these, but no correctness claim or Gate 5 test is made |

**Raw prediction definition:** the tensor returned by native
`LightningModel.forward`, including that checkpoint's native activation and
native prediction transform. This is not an assertion that the tensor is
pre-activation logits. Architecture logits are internal to the installed model.
Both branches compare exactly the same numerical definition. Native activation
and transform classes and full output shape are saved in the manifests.

The first application boundary is `_norm3`. All raw forward tensors are captured
before it. In addition, each input/output of `_norm3` and the full concatenated
adapter return are preserved to detect dropped records, changed axes or values
before aggregation. Score aggregation and response rendering are left for
later gates. Sequence IDs are application metadata rather than tensor axes;
reader IDs, prepared DNA order, batch offsets and comparison IDs establish the
mapping to actual output rows.

The raw axes are `(sequence, task, output_position)`, with CPU float32 tensors.
Observed native output shape for this checkpoint is **`(batch, 1, 1)`**. Its
`DilatedConvModel` uses the default `final_pool_func='avg'`; `ConvHead` applies
`AdaptiveAvgPool1d(1)` inside the architecture. The complete raw model output
therefore has one task and one pooled output position, not a 1,000-position
profile. No artificial profile is generated for testing. These are native
tensor captures, not application-aggregated scalar scores, even though each
record's native tensor contains only one element.
Input axes are `(sequence, A/C/G/T channel, input_position)`. The native
`predict_on_seqs` method enables evaluation mode but does not itself enter
`no_grad`/`inference_mode`. The direct runner retains that same native behavior.
The builtin path explicitly enters inference mode. Both grad-mode flags and
output `requires_grad` are recorded; evaluation mode, no training modules, no
active dropout and no augmentation calls are required. No backwards pass,
optimizer, attribution gradient computation or weight randomization is run.

## Independent reference

`direct_reference.py` executes in its own process from `/tmp` using Python `-I`.
It imports installed gReLU, NumPy and Torch, loads the checkpoint directly with
`LightningModel.load_from_checkpoint`, calls `.cpu().eval()`, and uses native
`predict_on_seqs` on the deterministic reference queries. No `core`, `workflows`,
`engine`, or `studio_web` modules may be loaded. The editable installation can
still add `src` through site configuration; the actual import path is recorded.

The only separately specified compatibility allowlist concerns safe checkpoint
deserialization of NumPy metadata. It does not process DNA or numerical model
outputs. The reference does not use Studio adapters, encoders, normalization,
aggregation, task selection, response handling or comparisons. Native batch
outputs are saved untouched and concatenated in explicit query order for
storage. Batch-size-1 output is the per-sequence reference; native batch-size-2
output independently characterizes batching effects.

Expected DNA queries come from saved accepted Gate 3 fixtures and independently
specified variants/motif replacement/curated substitutions. Reverse complement
uses a literal complement dictionary in the fixture builder. The reference
inputs are generated before website execution; they are not copied from a
potentially incorrect website output or preprocessing result. The website's
actual prepared strings must equal those independently specified strings.

## Transparent observation, exceptions and limited workflows

`website_worker.py` temporarily wraps the actual methods solely to observe
calls, save copies and return the originals unchanged. `LightningModel.forward`
is observed directly because native gReLU calls `self.forward(...)`, bypassing
a PyTorch hook registered only on the outer Lightning module. The real native
weighted forward always executes for numerical evidence. No expected prediction
is hardcoded and no mock returns a numerical prediction.

Two explicitly labeled exception cases inject `RuntimeError` before the model
body, without inventing output values. A transient native error exercises the
existing builtin fallback, which must produce real full tensors matching the
independent reference and disclose its fallback note. A persistent error must
produce HTTP 500 and no fabricated outputs. The latter alone is not counted as
numerical evidence; its handling check is supported by the real, independently
compared recovery case. Ordinary cases must not silently rely on injection.

Design and attribution stop after the actual baseline output, before
optimization/gradient computation. Curated saturation stops after its actual
reference and first three mutant outputs. These stops raise a labeled
`BaseException` after real numerical values are saved; they do not fabricate a
baseline to advance an inaccessible branch. Those workflows remain PARTIAL.
The uploaded mutagenesis case completes an actual one-position scan.

Model comparison selects `human` twice to execute both real inference branches,
each using the verified 2,114-base workflow window. That certifies a same-model
reproducibility path only. A distinct second identity/window-verified checkpoint
is unavailable in the existing evidence; a random demo or locally trained
fixture is not substituted, and no new model is downloaded.

## Comparison and resources

Before execution, tolerance is fixed at `rtol=1e-5`, `atol=1e-6`. Full outputs
are compared, including every task and output position. Exact equality is
recorded in addition to `np.testing.assert_allclose`. Repeated identical HTTP
requests require exact equality. No larger tolerance is selected after seeing
data. Error calculations use float64 differences and a zero-safe denominator
`max(abs(reference), float32.tiny)` for relative-error reporting; pass/fail uses
the stated additive absolute/relative tolerance.

Website and direct workers run sequentially, each with one Torch/OMP/MKL/BLAS
thread, a 1.8 GB RSS ceiling, a 650 MB available-memory reserve and a 600-second
timeout. Resource failures are BLOCKED, not PASS. Run-specific logs, resources,
raw arrays and failed runs are retained. Phase 1/Gates 1–3 hashes and the
checkpoint are checked before/after. Between website cases, the harness
releases temporary model references, collects garbage and trims unused glibc
allocator pages; it never changes the actual forward settings or results.
No installed package or styling changes
are required to observe these boundaries.
