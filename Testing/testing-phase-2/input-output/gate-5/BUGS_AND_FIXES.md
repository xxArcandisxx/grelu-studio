# Gate 5 bugs and fixes

## Preserved pre-fix evidence

[baseline-failures.json](baseline-failures.json) contains six original FAIL
observations from actual processing boundaries/static labels before fixes:

1. `_norm3` accepted a batch dimension inconsistent with the record count.
2. `aggregate` returned NaN rather than rejecting an invalid prediction tensor.
3. A finite float32 mean overflowed its intermediate sum to Infinity.
4. `report_builder.chart` silently discarded an invalid series while plotting
   the valid one.
5. Fixed five-decimal chart-data rounding turned a score near 1e-8 into zero.
6. The GM12878 comparison page called its pooled score a signal by output bin.

[scale-policy-baseline.json](scale-policy-baseline.json) adds an explicitly
synthetic FAIL: two task scores on unverified scales were subtracted without a
justified comparison contract. Those values are arithmetic fixtures, not model
inference. All baseline evidence is retained, not overwritten with passing data.

[curated-warning-baseline.json](curated-warning-baseline.json) retains an additional
saved-real-output replay failure: curated scan dropped actual fallback warning
metadata. It now propagates loader/adapter notes into its result and HTML,
with regression checks.

[downsampling-baseline.json](downsampling-baseline.json) records the same finite-mean
overflow in the separate multi-position chart downsampler. It now uses the
validated stable mean, with odd-group and large finite synthetic regressions.
This branch does not invent a positional output for pooled GM12878.

## Production changes

- `core/output_processing.py` adds finite nonempty tensor, record-count and task
  metadata validation, checked inference entry points, and neutral output
  semantics. None, empty/duplicate/mismatched task names are explicit errors.
- `core/models.py:_norm3` rejects empty/nonfinite/wrong-count results after
  supported legacy shape conversion. No native score transformation is added.
- `core/sequences.py:aggregate` passes L=1 through directly and safely averages
  L>1 in float64, with a scaled fallback for a finite mean whose sum would
  overflow. Raw positive/negative/zero values are neither clipped nor rescaled.
- Prediction, VEP, scan, design, attribution, exploration and curated scan call
  the shared checked boundary. All weights and native forward computation are
  unchanged.
- `workflows/predict.py` retains full precision in chart datasets and records
  native output semantics. Existing `tracks` keys remain for compatibility;
  their singleton value is not a spatial profile.
- `workflows/explore.py` permits a raw difference only for identical model
  configurations and the same selected task. Otherwise it returns None with an
  explicit comparability policy and warning; individual native scores remain.
  `exploration_views.py` gives unverified scales separate plots.
- `report_builder.py` rejects invalid chart series rather than silently
  removing them and avoids coordinate overflow for finite large values.
- Comparison JS rejects missing/nonfinite scores, retains signs, uses robust
  affine scaling, and labels a singleton as one model score without a profile.
  Relevant templates use neutral score/task labels and unknown-unit notes;
  mutagenesis legends describe lower/higher model scores, not proven biology.
  No stylesheet, navigation redesign, download algorithm or persistence
  implementation was changed.

The original Gate 4 fallback warning behavior is preserved and tested through
the current API using its actual saved warning metadata. Earlier Gate 3 optional
inverse failures remain unchanged as a separate external dependency limitation.

## Test-harness corrections retained

The run `20261122T172437060885Z` reported 254 PASS, 3 FAIL, 5 SKIP. Three expanded
HTML assertions expected a positive delta without the application's explicit
plus sign. The site correctly rendered its documented signed `+.6g` convention;
the expected formatting contract was corrected without changing any effect
formula, raw value or site output. The failed HTML and assertions remain saved.

Fresh design and attribution capture files now have separate phase prefixes,
so resetting a per-workflow call counter cannot reuse a raw filename. The two
baseline numbers happened to be identical in the inspected fixture, so there
was no observed numerical mismatch; the ambiguous capture namespace was still
corrected. [Recording note](raw-capture-harness-finding.json) preserves that
distinction. No prediction value was manually corrected.

One renderer-only rerun reused completed fresh evidence after checking that
production sources and the fresh oracle were unchanged; its resource record
explicitly says `reused_saved_evidence`. The default verification command runs
both workers. The latest final run's resource record identifies whether its
fresh evidence was executed or reused; previous full runs remain available.

## Remaining limitations

No browser runtime/Playwright/Selenium/Node was installed, and none was downloaded.
JS checks are static; interactive charts and event behavior are NOT TESTED.
Same-model comparison is verified; distinct checkpoint scale equivalence is
not. Saturation uses an explicit one-position harness subset. Full combinatorial
motif simulations, larger design searches, optional prediction shuffle controls,
and positive-n variant empirical p-value calculations are not fully certified.
No persistence/history/download-file or biological-accuracy claim is made.
