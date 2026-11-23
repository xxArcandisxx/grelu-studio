# Gate 2 input-path coverage map

Scope is parsing, normalization, window preparation and exact DNA strings at the
next-stage boundary. VERIFIED means the recorded backend input cases passed;
it does not certify predictions, browser behavior, or biological validity.
Read the latest run's `coverage.json` for actual classifications/counts.

| Input/workflow | Frontend component and payload | Flask handler | Actual validation/preprocessing | Final representation / boundary |
|---|---|---|---|---|
| Sequence prediction, typed DNA/FASTA or upload | `templates/analysis.html`, `model_select.html`, `static/model-inputs.js`; `model,fasta,file` | `/predict` → `workbench.analyze` | `parse_batch` → `core.dna_inputs.parse_records/normalize_dna`; `required_length`; `core.io.read_sequences`; explicit prediction input policy | Ordered `(id,DNA)` records, then exact string batch before `ModelAdapter.predict` |
| Sequence QC | Same controls | `/qc` → `workbench.analyze` | Same parser; `quality_report` records lengths without resizing | Ordered records, real QC HTTP response; mismatched lengths are descriptive, not compatible model inputs |
| Direct reference/alternate comparison | `templates/studio.html`, `static/studio.js`; JSON `model,reference,alternate`; `/variants` form includes CSRF | `/api/compare`, `/variants`, `/` | `parse_sequence` → `parse_single`; equal lengths; exactly one known-base substitution; native workflow window; unsupported genomic fields rejected | Reference then alternate strings, source-relative 1-based variant position; no genomic assembly implied |
| Supplied haplotypes / prioritization / multiple alleles | `templates/analysis.html`; `model,fasta,file` | `/allele-specific`, `/prioritize`, `/multi-allelic` | Shared FASTA parser, native length check, `variant_views.validate_pairs` | Aligned records in supplied order; prioritization requires one SNV; other two modes allow multiple supplied substitutions, not indels |
| Batch SNV CSV | `templates/advanced.html`, `static/advanced.js`, CSV editor/upload | `/batch-variants` → `advanced_views.run` | `validate_variants`; exact headers; unique IDs; 1-based sequence-relative `pos`; ref match; known unequal single-base alleles; `workflows.vep` text-preserving CSV read and mutation | All reference strings then alternate strings; IDs remain strings such as `001`; all unaffected bases compared against independent fixture expectations |
| Attribution / mutagenesis / design | `templates/advanced.html`; `model,text,file,target,start,end` plus tool settings | `/attribution`, `/mutagenesis`, `/design` | Shared parser, one record, native window and region bounds; design rejects N; real workflow preparation | Initial input/seed string before first prediction; no gradients, edits chosen by scores, or downstream interpretation are tested |
| Motif simulation | `templates/exploration.html`; `text,sequence-file,motif,positions,orientation,gaps,controls` | `/motif-simulation` | Shared parser; native window; `simulate_motif` equal-length replacement and control generation | Reference and generated replacement/control strings captured before prediction; checks preserve unaffected bases; no predicted effects |
| Model comparison | `templates/exploration.html`; `model_a,model_b,text,sequence-file,policy` | `/model-comparison` | Shared parser; `prepare_comparison_inputs` validates BOTH models first; explicit `strict` or `center` | All preflight windows/offsets preserved, first actual handoff captured. PARTIAL: second actual handoff requires execution beyond the first inference boundary and is not run |
| Motif matching | `templates/lab.html`; `text,file,patterns,k` | `/motifs` | Shared parser and real pattern validation | DNA records captured immediately before `motif_analysis`; no motif-result analysis |
| PWM scanning | `templates/exploration.html`; `text,sequence-file,matrices,matrix-file,...` | `/pwm-scan` | Shared parser, real matrix parsing and request validation | DNA records captured before `scan_pwm`; no score or biological interpretation |
| Dataset FASTA/raw DNA/labeled CSV | `templates/research.html`; `kind,text,file` | `/research/dataset` → `core.dataset.dataset_overview` | Shared DNA normalization; explicit limits; numeric labels; duplicates intentionally retained for input QC | Actual normalized local `records` observed via a delegating Counter wrapper; result order/IDs plus unique row indices retained; no model inference |
| Training/fine-tuning labeled CSV | `templates/lab.html`; `model,text,file,length,epochs,...` | `/training`, `/fine-tuning` → `lab_views.page` | `training_rows` parses actual `sequence,label` CSV, exact configured/native window, 8–256 distinct sequences, numeric labels | Parsed `(sequence,label)` rows captured before job submission; no job runs. Training creates a compact model with 50–2,000-base window and does not train GM12878 |
| Curated reference preset | `templates/saturation.html`, `static/saturation.js`; `locus` only | `/saturation` → `curated_scan` | Allows only selected preset and CSRF field; rejects DNA overrides; native checkpoint hash guard | Preset hg38 sequence at first pre-inference boundary. PARTIAL: later saturation mutations are beyond the first prediction and not run. Stored preset bytes independently match cached UCSC hg38 2bit extraction |
| BED upload/extraction | Retired; no active frontend | `/research/bed` GET/POST return 404 | None | NOT TESTED/unsupported. BED interval/assembly/strand extraction checks explicitly SKIP; a BED file uploaded as FASTA is rejected |
| Engine FASTA/sequence CSV/list inputs | No separate website upload endpoint; internal engine/CLI formats | `core.io.read_fasta/read_sequences` | Shared strict DNA parser; exact CSV text; IDs preserved; nonstrings/empty records/duplicate IDs/oversized batches rejected | Helper return recorded separately from website route evidence |
| Standalone genomic conversion/variant CLI | `bed_to_fasta.py`; `workflows.vep._fetch_context` with explicit `genome_fasta` | No Flask route | Separate CLI-only functionality | NOT TESTED as a genomic input pipeline. No new BED/VCF feature or extraction claim is inferred from these helpers |

## Policies established and exercised

- DNA bases: ASCII A/C/G/T/N, case-insensitive. N is intentionally accepted in
  general sequence/context input, rejected by design, and forbidden as a changed
  allele. Other ambiguity codes, punctuation, numbers and non-ASCII DNA are invalid.
- Normalization: ASCII edge whitespace and wrapped lines normalize to uppercase
  DNA; embedded spaces do not represent bases. CRLF/CR/LF are recognized. UTF-8
  upload BOM decoding is explicit; original file bytes remain in evidence.
- FASTA: primary ID is the first header token, with original case/Unicode
  metadata preserved; descriptions remain in the original payload. Empty IDs
  and empty records are rejected. Raw DNA without a FASTA header is a supported
  single record; orphan DNA followed by a FASTA header is invalid.
- Model-bound FASTA IDs must be unique. Descriptive Dataset Overview intentionally
  retains duplicates and exposes unique zero-based `record_index` values.
- Upload and editor supplied together retain the existing rule: upload wins.
  The normal browser preview clears the file field and submits the editable
  textarea; the backend precedence is explicitly tested and both original
  submitted fields remain in evidence. Inputs are never merged.
- GM12878 model-bound routes require its verified 2,114-base workflow window.
  QC, motif matching, PWM scanning and Dataset Overview have their own descriptive
  limits and do not resize to GM12878. The compact trainer has its own declared
  window, not GM12878's. Model Comparison requires explicit resize policy.
- Demo comparison retains its existing short-input convenience through an
  explicitly configured `allow_short_with_n_pad` model policy; effective padded
  strings, offsets, lengths and operation names are recorded. There is no
  unannounced permission to resize arbitrary pretrained inputs.
- Supplied sequence strings have **unknown assembly provenance**. Selecting a
  model trained on hg38 does not make uploaded DNA hg38. Variant positions are
  1-based within the supplied sequence; BED extraction is unavailable.

## Unverified boundaries

No browser or JavaScript runtime, live-server instance, tensor encoding, actual
prediction, output transformation, attribution, biological effect, optimizer or
training job is certified by Gate 2. These are explicit scope boundaries. The
reference extraction checks use real cached data, not a mocked reference result.
