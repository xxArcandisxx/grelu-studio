# Gate 5 output-processing pipeline and meaning

This map distinguishes **model-native**, **gReLU-library**, and **Studio**
operations. It describes the implemented operations; it does not infer common
genomics transformations absent from this application. Tests are software and
presentation checks, not biological accuracy diagnostics.

## Authoritative native contract

Gate 4's pinned `Genentech/GM12878_dnase-model` revision
`9f2aaba3e04c5b7324a239c7af53fc4732cb2c3e`, SHA-256
`f266b296f49b4e97e3ac52e94594da0aaae1d467778ea58b421e1ee7c4482bea`,
accepts the verified Studio 2,114-base workflow window and returns float32
`(batch,1,1)`. Task 0 is `ENCFF093VXI`.

The **model-native** `DilatedConvModel` head already performs adaptive average
pooling to one output position. The installed **gReLU-library** activation and
prediction transform observed for this checkpoint are Identity. Its native
return is therefore the score supplied to Studio. No documented unit or
invertible target-scale contract is established by these tests. The website
uses **model prediction score**, native scale, units not established; it does
not assume probability, log1p, count units, exponentiation, sigmoid, softmax,
normalization or clipping. A single score is not a positional coverage profile.

The model-specific window does not imply every genomic model has a fixed
2,114-base length, nor that every model produces a single output position.
The production helpers validate generic `(sequence,task,output_position)`
tensors and preserve supported multi-position model outputs.

## Complete application trace

| Workflow / stage | Actual functions and files | Inputs → operations → outputs | Classification |
|---|---|---|---|
| Native checkpoint execution | installed `grelu.lightning.LightningModel.forward`, `predict_on_seqs`; `grelu.model.models.DilatedConvModel`, `ConvHead`, `AdaptivePool` | Encoding/eval/CPU forward; native pooling; Identity activation/transform → `(N,1,1)` | Model-native + gReLU; already verified by Gate 4 |
| Adapter axis contract | `core.models.ModelAdapter.predict`, `_predict_grelu`, `_predict_torch`, `_norm3` | Native outputs → supported legacy axis normalization → canonical array; wrong batch/empty/nonfinite invalid | Studio; no score rescaling |
| Processing contract | `core.output_processing.prediction_tensor`, `predict_checked` | Require nonempty finite numeric 3-D tensor, matching input count and nonempty unique task names | Studio validation; explicit errors, not NaN-to-zero/null coercion |
| Score selection/mean | `core.sequences.aggregate` | Optional existing center window; L=1 passes through directly; L>1 arithmetic mean, float64 accumulation with scaled overflow fallback | Studio; no extra 1,000-position aggregation |
| Prediction + batch prediction | `workbench.analyze`, `studio_web.compare`, `workflows.predict.handle_predict_workflow` | Validated IDs/DNA → checked raw tensors → score and separately requested RC score per task; optional shuffle baseline → rows and API objects | Studio; multi-record FASTA, upload, ordering and duplicate-DNA distinct IDs tested |
| Prediction chart data | `workflows.predict._downsample` | Keep native values when output fits requested display size; otherwise existing equal-count array splits and means; no fixed 5-decimal truncation | Studio; pooled checkpoint yields one point, never an invented profile |
| Pair API presentation | `static/studio.js`: `showTrack`, `renderPlot`, `format`; `templates/studio.html` | Selected named task → reference / alternate / signed A−R; numeric `toPrecision(5)`; actual singleton plotted as a point with no positional-profile claim | Studio JS; source/data checked, browser NOT TESTED |
| Allele-specific / multi-allelic | `variant_views.validate_pairs`, `comparisons`; `workbench.analyze` | First record baseline, alternate minus baseline per task; RC delta separately; input difference counts | Studio; signed score changes, not proven biological regulation |
| Prioritization | `variant_views.comparisons` | Stable descending `abs(delta)` within task; equal effects retain source order; one-based ranks | Studio; no variant top-k filter; input count bounds are separate |
| Batch SNVs | `workflows.vep.handle_vep_workflow`; `advanced_views.run` | Reference block then alternate block; per-task alternate−reference; RC sign equality averaged across tasks; `argmax(abs(effect))` picks top task; stable decreasing absolute top effect | Studio; known task-0 cases verified, multitask arithmetic additionally uses synthetic fixtures |
| Optional variant backgrounds | `workflows.vep.handle_vep_workflow` | For n shuffles, `(1+count(abs(background)>=abs(effect)))/(n+1)`; n=0 leaves p-values None/Not tested | Studio; real positive-n empirical p-values are NOT VERIFIED here |
| Uploaded mutagenesis | `workflows.scan.handle_scan_workflow` | Original score, three DNA substitutions per known base, each checked mutant−reference; one-based position; A/C/G/T mapping; N marked skipped | Studio; one actual saved-output position numerically tested; full scans remain unverified |
| Curated saturation | `curated_scan.scan_locus` / `saturation` | Preset raw reference then three substitutions; alternate−reference; genomic start0 + input index0 + 1 | Studio; explicit one-position harness subset from Gate 4, not full 24-substitution certification |
| Model comparison | `workflows.explore.prepare_comparison_inputs`, `compare_models`; `exploration_views.run` | Model-specific preflight windows; target selection; individual native scores; B−A only for identical configurations and same target | Studio; different/unverified scales get None + explanation and separate charts, not assumed harmonization |
| Motif simulation | `workflows.explore.simulate_motif`; `exploration_views.run` | Equal-length motif replacement, preserved context → native reference/condition score → condition−reference; optional control score and condition−control | Studio; one saved real placement tested, optional controls/orientation/spacing not fully certified |
| Attribution | `workflows.attribution.handle_attribution_workflow`; `ModelAdapter.grad_attribution` | Native selected-output input derivative → subtract mean gradient over four channels → multiply literal one-hot → sum channels; full positions, display region and stable top-20 abs contribution excluding N | Native derivative + Studio correction. Actual matrix compared to independent literal encoding / native autograd / explicit formulas |
| Sequence design | `workflows.design.handle_design_workflow` | Mean on-task−mean off-task objective; max candidate selected; strict improvement `> current+1e-9`; accepted-history only; untouched bases outside editable interval retained | Studio; one real iteration / two real candidates tested; no weight learning |
| Design baseline percentage | same design workflow | `100*count(shuffled objective < final objective)/n` | Studio; five actual backgrounds tested; descriptive percentage, not probability or calibration |
| Server numeric formatting | `workbench.py`, `templates/{analysis,advanced,saturation,report_content}.html` | Full precision backend values → explicit significant-digit formatting only | Studio HTML; tested rows/metrics and graph data audited separately from browser behavior |
| Server SVG datasets | `report_builder.chart`, `build_report`; live exploration templates | Native numeric series → finite validation → affine SVG coordinates; equal-valued series centered; invalid series cannot silently disappear | Studio; signed/zero/huge synthetic chart fixtures with independent Decimal coordinate oracle |

`predict_checked` wraps every inference-backed calculation entry in prediction,
VEP, uploaded scan, curated scan, design, attribution and exploration. Model
exceptions remain errors or explicitly disclosed fallback notes; the Gate 4
warning-propagation fixes remain intact. Generic `core.utils.to_py` continues
to support unrelated descriptive metadata; invalid model tensors are rejected
before that serializer can turn nonfinite model numbers into null.

## Output meaning and display contract

- Raw pooled values are passed through, with float64 promotion only for safer
  application arithmetic. There is no learned-model modification.
- Effects are signed **alternate−reference**. Absolute values are ranking
  magnitudes, not a replacement for signed stored effects. A positive delta
  means a higher model score; no proven biological increase is inferred.
- RC sign agreement is a software/model consistency description, not an
  accuracy score. Missing empirical tests are None / Not tested, never p=0.
- Single output chart datasets retain full native precision. Display rounding
  is applied in the view only; very small positive/negative scores are not
  replaced with zero in chart data.
- Task names come from the model contract; missing, duplicated or incompatible
  metadata blocks processing rather than inventing a task assignment.
- Different model/task scales have no justified numeric subtraction here.
  Individual scores are retained, their difference is explicitly uncomputed,
  and unverified scales are not forced onto a combined chart axis.
- Server HTML/SVG and data attributes are not browser execution. Actual JS
  pixel placement, event handling and dropdown interactions remain NOT TESTED.

## Independent expected-value calculations

`verify_processing.py` reads untouched Gate 4 `direct/all-batch1.npy` and the
independently specified DNA query list, then replays those recorded tensors at
the inference/postprocessing boundary of real Flask handlers. Expected values
use Python formulas and original raw records, not Studio aggregation/scoring
helpers. Every numerical check records its mode, formula, expected/actual,
errors, native identity/context, IDs, warnings and tolerances. Full backend
records, actual HTML and boundary tensors are saved. Replay is **postprocessing
evidence**, not a claim of fresh or newly independent inference.

Synthetic guards, multitask ranking/objectives, signed/zero/extreme numbers and
chart fixtures are explicitly labeled SYNTHETIC_PROCESSING_ONLY, including in
their replay-adapter metadata and returned test warning. None represents a
pretrained model prediction. The synthetic design objective case has zero
iterations and fabricates no optimized sequence outcome.

`verify_fresh.py` runs a small real CPU design request and full-input attribution
request through the actual website, reusing one real checked checkpoint module
to bound memory. Candidate objectives and acceptance are independently derived
from the captured real raw calls. Attribution's expected tensor uses a literal
independent one-hot encoding, native input autograd, and explicit NumPy channel
mean subtraction/multiplication. It does not call the website's attribution or
aggregation helper as the expected oracle. Every base/channel is compared, not
only a baseline prediction. Native parameter bytes must remain unchanged.

Default scalar tolerances are rtol=1e-6, atol=1e-7 for float32-origin calculations;
strings/IDs/ordering/chart data arrays and rounded cell strings are exact. The
actual float32 attribution correction is compared against explicit float64
arithmetic with predeclared rtol=1e-5, atol=1e-7. All maximum discrepancies are
saved. Chart coordinates use independent high-precision Decimal affine formulas
and exact three-decimal rendered coordinate comparisons.

No trained model is downloaded, retrained or fine-tuned. Input gradients are
not optimizer steps. The one-step design modifies a candidate DNA string,
not model parameters. Persistence/download/history checks remain Gate 6.
