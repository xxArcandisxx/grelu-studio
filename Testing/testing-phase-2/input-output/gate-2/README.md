# Gate 2 — DNA Input Integrity

**Completed: 265 PASS, 0 FAIL, 12 SKIP, 0 BLOCKED.**

- [Detailed verification report](runs/20261122T135128551126Z/REPORT.md)
- [Machine-readable results](runs/20261122T135128551126Z/results.json)
- [Coverage matrix](INPUT_PATHS.md)
- [Bugs and behavior changes](BUGS_AND_CHANGES.md)
- [Input fixtures and expected outcomes](fixtures/cases.json)

Run from the repository root:

```bash
.venv/bin/python Testing/testing-phase-2/input-output/gate-2/verify_gate2.py
```

Every run gets a new directory. Earlier failed/blocked evidence is retained.
Input-only probes exercise real Flask handlers and parsers, capture strings
before encoding/inference and return no numeric predictions. HTML files are
saved responses/form evidence, not copies of the application implementation.

BED upload is retired and unverified. Browser execution is unverified. Model
comparison's second actual handoff and curated saturation's post-inference
mutants remain partial. The cached real hg38 preset reference was independently
checked without downloads. Gate 1 remains 39 PASS, 0 FAIL, 2 SKIP; 13 existing
input-only regression checks also pass. No Gate 3 work is authorized or started.
