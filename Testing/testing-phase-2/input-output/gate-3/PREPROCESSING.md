# Gate 3: actual preprocessing trace

This gate checks software input tensors, not biological accuracy or numerical
inference equivalence. It stops before the architecture's forward body.

## Model-specific requirements

The cached, pinned GM12878 checkpoint is `Genentech/GM12878_dnase-model`, revision
`9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e`, SHA-256
`f266b296f49b4e97e3ac52e94594da0aaae1d467778ea58b421e1ee7c4482bea`.
Its architecture is `grelu.model.models.DilatedConvModel`. The checkpoint-defined
Studio workflow window is 2,114 bases; this does not establish that the
convolutional architecture requires every possible input to have that length.
Gate 1 remains authoritative for model identity/length contracts. A separate
untrained demo CNN uses a 200-base window. It provides evidence of its own
preprocessing only, with no biological validity claim. Other models are not
certified by this gate.

## Real functions

| Stage | Source and functions | Behavior |
|---|---|---|
| Validated DNA | `core/dna_inputs.py`: `normalize_dna`, `parse_records`; `core/io.py`: `read_sequences`, `read_fasta`; `workbench.py`: `parse_batch` | ASCII ACGTN; case normalization before native encoding; input IDs retained; invalid characters and duplicate model-input IDs rejected |
| Workflow window | `workflows/predict.py`: `handle_predict_workflow`; `workflows/explore.py`: `prepare_comparison_inputs` | Strict or explicitly requested center handling; original/final lengths, policy, signed offsets and effective DNA recorded |
| Center resize | `core/sequences.py`: `fit_with_offset` | Crop start floor((original-target)/2); padding left floor((target-original)/2), remainder right. Returned shift is negative crop start or positive left padding |
| Native adapter | `core/models.py`: `ModelAdapter.predict`, `_predict_grelu`, `iter_sequence_batches` | Actual slices, center fitting, native `LightningModel.predict_on_seqs` |
| Native encoding | installed `grelu/sequence/format.py`: `strings_to_indices`, `indices_to_one_hot`, `strings_to_one_hot` | Integer NumPy int8: A=0 C=1 G=2 T=3 N=4. Torch one-hot uses five classes, removes N channel; float32 B×4×L, channel order A,C,G,T, N all zero |
| Builtin encoding | `core/sequences.py`: `one_hot`; `core/models.py`: `_predict_torch` | Independent Studio LUT implementation; NumPy float32 B×4×L; `torch.from_numpy(...).to(device)`. New guards reject malformed DNA before allocating an encoded result |
| Native device / boundary | installed `grelu/lightning/__init__.py`: `predict_on_seqs`, `format_input`, `forward` | Encoder result `.to(device)`; `.model.eval().to(device)`; `format_input` retains 3-D tensor. Capture pre-hook is on `.model`, since native inference calls `self.forward` directly |
| Gradient input | `core/models.py`: `grad_attribution` | Studio one-hot, device transfer, `requires_grad_(True)`; captured before model body. No gradient computation occurs |
| Reverse complement | `core/sequences.py`: `revcomp`; prediction / variant workflows | Explicit sequence transformation, tested against independent reversed complement strings and actual model-boundary tensors |
| Variant windows | `workflows/vep.py`: `handle_vep_workflow`, `_fetch_context` | Reference then alternate lists; one-based input positions adjusted with window shift. First/middle/last SNVs tested through actual CSV route; captured prepared alternate batches replayed without numerical scores |
| Motif preparation | `workflows/explore.py`: `simulate_motif` | Replacement preserves length; reference and actual prepared motif mutant tensors tested. This gate does not certify every motif/control/partner combination |
| Curated saturation | `workflows/scan.py`: `handle_scan_workflow`; `curated_loci.json` | Both preset reference tensors captured. Mutants require a preceding reference score and remain unverified |
| Training | `lab_compute.py`: `train` | Shared `one_hot` implementation identified and tested separately; training pipeline never invoked |

The website paths above do **not** use gReLU `SeqDataset`, `VariantDataset`, or
`predict_on_dataset`. No artificial dataset coverage is claimed. Installed
dataset/augmentation sources are fingerprinted for reproducibility, not certified.
Checkpoint training metadata enables reverse-complement/shift augmentation;
`predict_on_seqs` does not consume that metadata or construct an augmenter.
An observed `Augmenter.__call__` wrapper records zero calls and repeated actual
tensor captures match, with unchanged Torch RNG state and evaluation mode on isolated preprocessing
calls. HTTP engine dispatch explicitly sets its requested seed; a request-level
RNG state change is recorded but does not imply stochastic tensor augmentation.
No website inference shift or separate left/right cropping policy was added.

## Evidence and independence

`verify_gate3.py` constructs expected arrays using literal channel vectors and
explicit Python loops. Its crop/pad and complement oracles do not import Studio
or gReLU encoders. DNA fixtures are exact accepted final strings from the Gate 2
run named in each run's `fixtures.json`, with source results hash. Fixtures
include nonperiodic context, N-containing and all-N sequences. Expected arrays
are calculated at runtime, never taken from model predictions.

Actual `actual.npy` arrays are copies of real tensors (or direct short encoding
results); `expected.npy` arrays come from the independent oracle. Comparison
JSON includes IDs, shapes, dtype, finite status, channel values, device,
evaluation/RNG settings, hashes and all exact value mismatches with one-based
position and channel. Shape mismatches are explicitly reported as failures even
when no elementwise comparison is possible. HTTP `/predict` additionally saves
actual reader ID-to-DNA records and checks their mapping to tensor rows.

Real checkpoint adapters are loaded once and reused by redirecting the loader
factory, avoiding repeated heavy deserialization. Actual workflow preparation,
adapter implementations, native encoding, format/device handling and tensor
hooks run. A thin observer on `predict` records its actual input and immediately
calls the original method; it never supplies model outputs. Hooks raise a
`BaseException` sentinel so the adapter's `Exception` fallback cannot hide it.
Conv/Linear safety hooks must remain untouched. No weighted forward body runs.

Stopping at the first tensor cannot advance a loop that needs a prediction
result. The actual shared batch iterator is fully traversed separately, and
each real prepared chunk is sent through a stopped real adapter invocation.
This covers batch sizes 1 and 2, a final incomplete batch, order, duplicate DNA
under distinct IDs, and all prepared reference/alternate tensors without fake
numeric returns. It does not certify numerical concatenation or uninterrupted
multi-batch request completion. The same limitation applies to a second model
in a continuous comparison request; both preflight windows are separately
encoded at their real boundaries.

CPU is verified; CUDA/MPS are not. Worker RSS is limited to 1.8 GB, with a 650 MB
available-memory reserve and 240-second timeout. Earlier evidence content is
hashed before/after. No downloads, optimizers, training, checkpoint rewriting,
biological benchmarks, output transformations, visualizations or exports run.
