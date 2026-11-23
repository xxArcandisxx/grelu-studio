# Gate 3 — preprocessing and tensor encoding

**Completed, overall FAIL: 150 PASS, 5 FAIL, 8 SKIP, 0 BLOCKED.**

All tested forward-input tensor comparisons pass. Five optional installed gReLU
single-base inverse-conversion checks fail; they are preserved as failures.
The website does not use that inverse conversion for prediction.

- [Detailed report](runs/20261122T142625611276Z/REPORT.md)
- [Machine-readable results](runs/20261122T142625611276Z/results.json)
- [Workflow coverage matrix](runs/20261122T142625611276Z/WORKFLOW_COVERAGE.md)
- [Actual functions, independent oracle and limits](PREPROCESSING.md)
- [Bugs, fixes and remaining failures](BUGS_AND_FIXES.md)
- [Deterministic accepted DNA fixtures](runs/20261122T142625611276Z/fixtures.json)
- [Actual / expected tensor arrays](runs/20261122T142625611276Z/captures/)
- [Regression results](runs/20261122T142625611276Z/regressions.json)

Run from the repository root:

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-3/verify_gate3.py
```

Each run saves new evidence and returns nonzero for FAIL/BLOCKED. Preloaded real
GM12878 model-input hooks stop before its architecture forward body. No numeric
model outputs, mocked prediction values, training or biological analysis are
used. The independent literal encoding oracle is in the test runner.

Complete regression reruns: Gate 1 **39 PASS / 0 FAIL / 2 SKIP**; Gate 2
**265 PASS / 0 FAIL / 12 SKIP / 0 BLOCKED**. Existing Phase 1 and prior gate
evidence and checkpoint weights are unchanged. Gate 4 has not started.

## Failure investigation and closure

[Closure report](investigation/REPORT.md): **SAFE TO PROCEED WITH DOCUMENTED
LIMITATIONS**, pending user review. Original combined results above remain
unchanged and FAIL. The five failures reproduce directly in installed gReLU;
no current website workflow reaches the affected inverse. Required recorded
website subset: **148 PASS, 0 FAIL, 8 SKIP**. Separate optional dependency suite:
**2 PASS, 5 FAIL**. No failures were relabeled. Gate 4 has not started.
