# Gate 2 bugs, fixes and documented behavior changes

See preserved baseline and follow-up run results. Bugs are grouped below;
multiple failed cases can demonstrate the same bug. A failed expectation in the
test harness is not counted as a website bug.

| ID | Observed issue | Change | Regression cases |
|---|---|---|---|
| B01 | Unicode whitespace/separators could disappear through strip/splitlines and produce accepted DNA | Shared ASCII DNA policy rejects non-ASCII DNA before normalization; ordinary ASCII wrapping and lowercase remain supported | `dna_unicode_separator`, `dna_unicode_edge`, `dataset_unicode_dna` |
| B02 | Blank FASTA identifiers were replaced with invented names | Explicit rejection of missing identifiers; valid primary IDs and record order preserved | `fasta_missing_identifier_text`, `fasta_missing_identifier_upload` |
| B03 | Engine parsing accepted invalid/nonstring DNA, dropped empty records/preamble, allowed duplicate IDs, or truncated oversized batches | `core.dna_inputs` centralizes strict validation; engine input returns explicit errors rather than changing/drop/coerce/truncate records | `engine_empty_records`, `engine_nonstring`, `engine_excess_records`, `engine_invalid_base`, `engine_duplicate_ids`, `engine_fasta_preamble`, `engine_fasta_duplicate` |
| B04 | Pandas empty-cell inference could become string `NAN`; numeric-looking IDs lost leading zeros | Standard-library sequence CSV reader preserves text and rejects missing DNA; VEP CSV uses string dtype with NA coercion disabled | `engine_csv_empty_cell`, `engine_csv_leading_zero_id`, six `variant_csv_*` boundary cases |
| B05 | Sequence-only comparison ignored submitted genomic fields | Reject unsupported fields with a clear sequence-only error; no assembly or genomic position is silently honored | `json_genomic_fields_not_supported` |
| B06 | Strict model comparison could reach the first inference boundary before discovering that the second model's length was incompatible | Preflight BOTH model windows before any prediction; exact center crop/padding operations and effective strings retained | `comparison_validate_second_model_before_inference`, `comparison_center_*` |
| B07 | Demo short-input padding was not an explicit model policy, and prediction results lacked a complete effective-input record | Declare demo padding policy in its model spec; ordinary prediction defaults strict; save effective sequences, offsets, lengths and operations in prediction records; update the existing frontend hint | `demo_short_policy`, strict/center comparisons |

Dataset Overview continues to retain duplicate identifiers for descriptive QC,
and now includes a unique `record_index` for each row. Duplicate model-input
identifiers still fail. No design/layout/CSS changes were introduced.

## Test-harness corrections, retained transparently

1. An initial fixture-generation attempt contained Python float NaN, which cannot
   be serialized as strict JSON. It was blocked before website checks, retained
   as `controller-error.json`, and replaced with a JSON numeric input. Actual
   empty-CSV-to-NAN behavior was still tested independently.
2. The first motif-simulation expectation assumed only a reference string, then
   assumed a control was selected even though the test omitted the checkbox.
   The corrected payload explicitly selects controls and verifies the reference,
   equal-length replacement, control base composition and all unchanged bases.
   These were test-expectation errors, not evidence of a motif-processing defect.
3. Descriptive input preservation was initially marked SKIP rather than inferred
   from summary statistics. Delegating observation of the actual normalized
   `records` now supplies direct preservation evidence.

The expanded upload tests initially used `file` for three exploration pages whose actual frontend/backend schema is `sequence-file`. Their payloads were corrected from the real form schema. An invalid-upload clone also inherited a valid-resize expectation; that assertion was removed for rejection cases. No production upload handler was changed for those test-harness mistakes.

All earlier failed/blocked runs are retained. Fixture expectations are explicit,
not numeric model predictions. No production repair was made to retired BED or
CLI-only genomic workflows; their lack of website coverage remains visible.

## Changed production files

- `core/dna_inputs.py`: shared parser and normalization policy.
- `studio_web.py`: common parser, explicit demo policy and strict comparison schema.
- `workbench.py`: shared FASTA parsing.
- `core/io.py`: strict engine sources and lossless text CSV handling.
- `core/dataset.py`: shared validation and explicit row identity.
- `workflows/vep.py`: preserve literal variant text/IDs during CSV reading.
- `workflows/explore.py`: atomic preparation of both comparison windows.
- `workflows/predict.py`: explicit window policy and effective input records.
- `static/model-inputs.js`: align hints with parsing and configured demo padding.

Existing Gate 1 length-authority logic and pretrained checkpoint contents are
preserved. No further gates or biological analysis are implemented here.
