# Gate 3: bugs, fixes, and remaining failures

## Fixed: malformed low-level input could become zero channels

`core/sequences.one_hot` previously used -1 LUT entries as a zero-column mask
without validating characters. Invalid `X`, punctuation, digits, non-ASCII
fullwidth A and empty sequences were accepted as encoded input. Website Gate 2
validation already rejected those inputs; the remaining defect affected callers
that bypassed that boundary. [Original observations](encoder-baseline.json)
retain five failing pre-fix checks and actual values.

The encoder now rejects malformed/empty/non-ASCII DNA and invalid window/batch
containers with clear ValueErrors before encoding. Valid lowercase support,
ACGTN mapping, resize behavior and float32 values remain intact. Gate 3 negative
guards and exact valid tensor comparisons are automated regression evidence.

## Fixed: invalid explicit batch settings

`ModelAdapter.__init__` previously used `batch_size or default`. Zero silently
selected a default; a negative size could produce no chunks; booleans and
noninteger values had no consistent validation. Explicit settings must now be
positive integers; `None` retains the existing CPU/device defaults. Regression
checks cover 0, -1, True and 1.5. No prediction is needed to demonstrate rejection.

`ModelAdapter.iter_sequence_batches` extracts the original identical slicing
logic for native, builtin and gradient paths. This enables complete inspection
of batching without fabricating numeric returns to get past a stopped forward
call. It introduces no new preprocessing policy. No pretrained weights or
website design files are changed by Gate 3.

## Remaining FAIL: optional upstream tensor-to-DNA conversion

Installed gReLU `one_hot_to_indices` obtains a NumPy int8 argmax, then indexes it
using a Torch boolean tensor to assign N. In this installed Torch/NumPy
combination, single-position A/C/G/T tensors decode as N and a single-position
all-zero N tensor raises IndexError. The direct forward encodings and integer
values are correct; the inverse check is separate. Longer mixed fixtures used
by this suite round-trip correctly, which does not erase the short failures.

The five failing inverse checks are retained as **FAIL**, with actual values or
traceback under each run's `integer-roundtrip/`. This function is not called by
the website prediction/encoding paths. No installed package patch or dependency
upgrade was applied: such an unpinned environment change would require separate
validation and is unnecessary to verify actual forward input tensors. Gate 3
therefore remains FAIL overall and awaits review; it is not presented as a
fully passing gate.

## Test-harness corrections retained in earlier runs

Initial runs stopped on the upstream inverse exception. The runner now saves
the failure and continues all other checks. Another run expected a two-record
tensor at routes that explicitly choose batch size 1. That was a harness error:
the first capture correctly contained one record. Current tests verify the
complete prepared list/order, capture the first batch, and replay remaining
actual chunks through the real adapter. Earlier logs, failing results and
comparisons remain on disk; their failures are not evidence of a website
two-record tensor defect.

## Files changed for Gate 3

- `core/sequences.py`: encoder input guards.
- `core/models.py`: explicit batch-size validation and shared batch iterator.
- `Testing/testing-phase-2/input-output/gate-3/`: runner, independent oracle,
  accepted DNA fixtures, expected/actual arrays, reports and evidence.
- `Testing/testing-phase-2/input-output/README.md`: current gate status links.
- New timestamped Gate 1 and Gate 2 regression runs and latest pointers only;
  their historical reports/results and Phase 1 content remain unchanged.

Other modifications already present in the working tree belong to prior work.
This gate does not claim them as its own changes.

A later harness revision initially required an unchanged Torch RNG state across
a whole HTTP request. `engine.execute_platform_workflow` explicitly seeds Torch
at dispatch (`engine.py`), so that assertion was too broad: the tensor was
identical, with zero mismatches. Final tests retain request-level RNG
observations and require unchanged RNG state for isolated repeated preprocessing
instead. The earlier failed run is retained.

## Closure investigation

See [the full investigation](investigation/REPORT.md) and
[complete website usage audit](investigation/USAGE_AUDIT.md). The original
failures remain unchanged. The direct isolated reproduction confirms an
upstream mixed NumPy/Torch indexing defect for one-element masks under the
installed versions, including both batched and unbatched valid representations.
No current website workflow invokes the affected inverse directly or indirectly.
Recommendation: SAFE TO PROCEED WITH DOCUMENTED LIMITATIONS after review;
Gate 4 has not started.
