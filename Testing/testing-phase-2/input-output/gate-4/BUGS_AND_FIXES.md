# Gate 4 bugs and fixes

## Production fix: fallback warnings were omitted

The actual fault-injection API request in run `20261122T163447347143Z` raised a
native-model RuntimeError and recovered with real builtin pretrained inference.
Its full raw outputs exactly matched the independent reference, but the API
returned only the checkpoint loader warning. The adapter's `notes` contained
the native failure and fallback notice, and the workflow discarded those notes.

`workflows/predict.py`, `workflows/vep.py` and `workflows/design.py` now append
`adapter.notes` to returned loader warnings, matching the convention already
used by attribution, scan and exploration workflows. These are the only
production edits in Gate 4. No encoding, model configuration, weights, output
values, aggregation, styling or frontend controls are changed.

Regression cases use real pretrained raw-output comparisons plus warning
assertions: transient-error API prediction, transient-error CSV variant
analysis, and zero-iteration/no-background engine design. The design case runs
no optimization; its actual initial and final seed forward calls execute and
the workflow's warning return is checked. Completed browser design behavior
remains outside this gate. Persistent-error API handling must give HTTP 500
without fabricated values and is paired with numerically verified recovery.

[Pre-fix findings](pre-fix-findings.json) retain the failing warning check,
actual adapter notes, actual API warning list, and original result hash.
The full affected Gate 4 suite is rerun after this fix.

## Harness correction: equivalent checkpoint paths

The same run originally reported 1 PASS, 25 FAIL and 7 SKIP. All 118 individual
raw/return/determinism numerical comparisons passed with maximum absolute error
zero. The broad case failures came from checking path-string equality:
the reference used the Hugging Face revision snapshot symlink, while Studio
resolved it to the content-addressed blob. Both name the same file and weights.

The test now requires equal resolved paths, the pinned file SHA-256 and
identical loaded state hashes. No inference reference, encoding, predicted
value, numerical tolerance or independent input was changed. The original
failed report, raw arrays and comparisons are preserved.

## Resource limit: retained BLOCKED run

The first run, `20261122T163138335995Z`, stopped at 1,812,254,720 bytes RSS, just
over the 1.8 GB ceiling: 0 PASS, 0 FAIL, 0 SKIP, 1 BLOCKED. No numerical PASS
was claimed. The harness reloads real checkpoints for distinct cases and was
retaining unused allocator pages between cases.

The worker now releases its temporary adapter reference, collects unreachable
objects, and calls glibc `malloc_trim(0)` between cases. The next website worker
completed at a peak of 1,446,555,648 bytes RSS. The 1.8 GB limit, reserve,
case sizes, numeric settings and tolerances remain unchanged. This cleanup is
Linux-specific test-process housekeeping, not a production inference change.

## Preservation and external findings

The suite uses actual loaders, model forward, adapters and Flask handlers.
No pretrained weight, installed package, website styling or existing Phase 1 /
Gates 1–3 result is changed. Every failed/blocked run remains in its timestamped
directory. Supplemental input-only regressions pass all 13 existing tests.

The Gate 3 optional inverse-conversion issue is an independently reproduced
external dependency defect. Its five FAIL results remain unchanged; Gate 4
does not use that inverse to compare model outputs.
