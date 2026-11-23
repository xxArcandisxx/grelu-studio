# Gate 4 — numerical inference equivalence

**28 PASS, 0 FAIL, 7 SKIP, 0 BLOCKED.**

All 93 captured real website forward tensors exactly match the independent
direct-gReLU reference. All 136 full-output comparisons have zero error.
This is software correctness evidence, not biological accuracy.

- [Detailed report](runs/20261122T163955672475Z/REPORT.md)
- [Machine-readable results](runs/20261122T163955672475Z/results.json)
- [Workflow coverage](runs/20261122T163955672475Z/WORKFLOW_COVERAGE.md)
- [Actual / independent raw tensors](runs/20261122T163955672475Z/website/manifest.json)
- [Independent reference manifest](runs/20261122T163955672475Z/direct/manifest.json)
- [Model/settings manifest](runs/20261122T163955672475Z/model-settings-manifest.json)
- [Pipeline and independence](PIPELINE.md)
- [Bugs and fixes](BUGS_AND_FIXES.md)

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-4/verify_gate4.py
.venv/bin/python Testing/testing-phase-2/input-output/gate-4/audit_evidence.py
```

The real native output is (batch,1,1), with one pooled position. Full arrays
are captured before application aggregation. Earlier failed/blocked runs are
retained. HTML files are actual server-response evidence, not website source
copies; numerical checks rely on raw arrays, not displayed summary values.

Fixed omitted fallback-warning propagation in prediction, variant and design
workflows; actual model calculations and weights are unchanged. Partial paths,
distinct second-model unavailability, other devices and later output checks
remain documented. The separate Gate 3 optional failures are preserved.

**Stopped after Gate 4. Gate 5/6 and biological output analysis have not started.**
