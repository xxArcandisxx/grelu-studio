# Gate 6 — bugs, fixes and regression evidence

The original isolated observations are retained in
[baseline-findings.json](baseline-findings.json), with original engine files and
DB under `baseline-state/`. No genuine experiment database was used.

| Finding | Root cause / original evidence | Resolution | Regression checks |
|---|---|---|---|
| Successful download of corrupt stored NaN | `json.loads` accepted nonstandard constants; Flask emitted NaN in HTTP 200 | Strict JSON object reading and finite serialization; generic visible error, retain original bytes | `corruption_visible`, `corruption_retained`, `reject_nonfinite_*` |
| JSON exponent overflow in legacy experiment/job | `parse_constant` does not catch legal JSON `1e309`, which overflows to Infinity; standalone actual endpoints both returned 200 | Validate all parsed nested values for finiteness too, even without a legacy checksum | `legacy_corruption_*`, `overflow_job_*`; original [reproduction evidence](baseline-overflow/observations.json) and [minimal script](probe_overflow.py) retained |
| Uploaded path retained in form | Existing filter excluded CSRF/password but not `file` and internal path keys | Filter secret/path form keys case-insensitively; preserve ordinary replay inputs | `form_safe_fields`; normal snapshot path checks |
| Missing job lookup created folders | `job_path` unconditionally mkdir'd on read | Explicit allocation only on submission/create | `missing_job_404`, `missing_job_no_directory` |
| Prediction CSV formula-like IDs active; GC rounded | `csv.writer` quoting alone did not protect text; GC used `round(...,4)` | Prefix risky text, quote CSV fields, retain full numeric precision and exact original IDs in JSON | Real generated file independently parsed: `csv_quoted_ids`, `csv_gc_precision`, `csv_score_precision`, `engine_json_original_ids` |
| NumPy direct storage unsupported | Standard JSON encoder rejected arrays/scalars | Explicit finite conversion for tested scalar/array types; unsupported values still reject | `numpy_json_roundtrip`, `reject_nonfinite_*` |
| Missing snapshot reproducibility/status information | Original row had no metadata column | Additive schema migration; status, payload/input hashes, save-time versions and available model provenance | Per-workflow identity/hash/status/version checks; `legacy_*`; fresh checkpoint/task checks |
| Job two-store partial completion and non-atomic status updates | SQLite archive and directly written status file had independent completion paths | Pending archive, atomic status replacement, promotion and compensating deletion; explicit pending cleanup if compensation fails | `job_*`, `partial_*`, `double_fault_*` |
| Unsafe metadata in attachment filename | Raw workflow string interpolated into header | Sanitize filename only; preserve original Unicode/workflow in JSON | `filename_safe_*`, `workflow_unicode_preserved` |

The VEP writer and the two existing browser CSV generators use the text-escaping
policy too. Engine prediction/VEP files were **executed and parsed**. Browser
generator changes received source audit only; their dynamic behavior remains
SKIP. This is not a claim of browser-export verification.

## CSV policy and limits

Numbers retain numeric signs and full Python float precision. Text beginning
with tab/CR/LF or, after whitespace, `=`, `+`, `-`, `@` (including fullwidth
variants) is apostrophe-prefixed and quoted; embedded quotes are doubled.
Prediction IDs with commas, quotes, newline, Unicode and formula prefixes were
tested in actual generated files using Python's independent CSV parser. Exact
standalone strand `+`/`-` are exempt in the Python/exploration policy. The advanced
client generator conservatively escapes all matching text; browser coverage is
still pending. Do **not** automatically remove apostrophes from imported CSV:
that cannot distinguish an original apostrophe from an escape. Canonical IDs
remain unchanged in JSON.

This mitigates a defined set of spreadsheet formula entry points; it does not
guarantee safety across every spreadsheet/import mode or after re-saving a
file. Spreadsheet behavior and loss of escaping on re-save are discussed in
[OWASP CSV Injection](https://community.owasp.org/attacks/CSV_Injection).

## Preserved failed test run and test-harness correction

`runs/20261123T095528332154Z/` retains **431 PASS, 2 FAIL, 5 SKIP**. The two FAIL
assertions expected the observer to see one save for motif simulation/model
comparison. Both pages actually completed and stored rows: their module had
already imported `save_experiment` by name, while the observer patched only the
source module. The observer therefore counted zero invocations. The corrected
test observes both actual call bindings and delegates to the same real saver;
it also checks the committed row count, downloaded original snapshot and real
restart. No model results or SQLite operations are fabricated to make it pass.
The next run retained **537 PASS, 0 FAIL, 5 SKIP** before the final additions.
These historical counts are distinct from the final run linked in REPORT.md.

The original Gate 3 optional inverse-conversion **five FAIL findings** remain
unchanged. Gate 6 neither relabels nor fixes that external library behavior.

## Production files changed for Gate 6

- `core/export_values.py`: finite JSON normalization and CSV text policy.
- `experiment_store.py`: strict snapshot reading/writing, additive metadata,
  corruption detection, safe form keys, visible history status, owned job
  archive status/compensation support.
- `experiment_views.py`: visible storage errors, sanitized attachment name,
  metadata export and nosniff.
- `lab_jobs.py`, `lab_views.py`: allocation/read separation, strict atomic
  status, collision handling, partial archive behavior and safe missing-file
  download handling.
- `workflows/predict.py`, `workflows/vep.py`: actual CSV quoting/text policy;
  prediction GC precision preserved.
- `static/advanced.js`, `static/exploration.js`: broader existing CSV text
  escaping policy; **not dynamically browser-tested**.
- `templates/experiment.html`, `templates/experiments.html`,
  `templates/advanced.html`, `templates/exploration.html`: concise status,
  snapshot-retention and export-policy explanations; no layout/CSS redesign.

Other working-tree edits predate Gate 6. The protected hash manifests cover
Phase 1, Gates 1–5 evidence and the checkpoint; tests do not mutate these files.
No installed package or pretrained state was changed.
