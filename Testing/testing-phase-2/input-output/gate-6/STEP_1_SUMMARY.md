# Phase 2, Step 1 — final verification summary

**Gate 6 is complete for its tested backend scope, with documented limitations;
Step 1 awaits user review/closure. Step 2 has not started.**

These are assertion counts, not a count of biologically validated samples. The
previous gates are retained results, not newly rerun full suites in Gate 6.

| Gate | PASS | FAIL | SKIP | BLOCKED | Evidence / conclusion |
|---|---:|---:|---:|---:|---|
| 1 — model identity | 39 | 0 | 2 | 0 | [Report](../runs/20261122T141545575344Z/REPORT.md): pinned checkpoint/state identity; native 2114 workflow window rejects conflicting 200 config; variable architecture distinguished from workflow constraint |
| 2 — DNA input integrity | 265 | 0 | 12 | 0 | [Report](../gate-2/runs/20261122T141643941401Z/REPORT.md): tested backend input validation/record integrity; genome/browser paths and full curated workflows limited |
| 3 — required website preprocessing | 148 | 0 | 8 | 0 | [Original report](../gate-3/runs/20261122T142625611276Z/REPORT.md), [closure](../gate-3/investigation/REPORT.md): actual forward-boundary tensors match independent encoding oracle |
| 3 — optional dependency compatibility | 2 | **5** | 0 | 0 | Same closure: reproducible gReLU single-base inverse defect; audited affected path unused by website; original five FAIL results retained |
| 4 — numerical inference equivalence | 28 | 0 | 7 | 0 | [Report](../gate-4/runs/20261122T163955672475Z/REPORT.md): independent native gReLU vs actual website raw output, exact equality in tested CPU fixtures |
| 5 — output processing and presentation | 273 | 0 | 5 | 0 | [Report](../gate-5/runs/20261122T180450337008Z/REPORT.md): actual processing/Flask HTML and limited fresh attribution/design; browser unverified |
| 6 — persistence, exports and history | 586 | 0 | 5 | 0 | [Report](runs/20261123T110400114845Z/REPORT.md): isolated real storage/download/restart tests; approved-output replay, fresh inference and synthetic fixtures separately labeled |

## What the evidence supports

The pinned GM12878 workflow's tested input parsing, encoding, raw numerical
inference, processing and stored/downloaded values agree with the independent
oracles described in each gate. It produces one task (`ENCFF093VXI`) and one
pooled output position: `(batch,1,1)`, not a fabricated spatial profile or
calibrated probability. Gate 6 keeps exact finite JSON values and original IDs,
preserves attribution matrices/design histories, and exercises actual owned
JSON/HTML downloads, engine CSV files and fresh-process history retrieval.

## Remaining limitations and closure decision

- **Browser execution remains unverified** across all six gates. Actual Flask
  HTML/static JS checks do not certify interactive charts, client number parsing,
  submission or Blob CSV/JSON/FASTA downloads. [Browser follow-up](BROWSER_SMOKE_PLAN.md).
- Full curated saturation, large design searches, motif/control combinations,
  positive-n variant-background tests and a distinct second pretrained model
  remain limited as recorded in earlier workflow matrices. Saving those partial
  results does not close their numerical/scientific gaps.
- Genome coordinate work without required local resources, GPU/other models,
  live-server identity and deployment behavior remain unverified where skipped.
- Gate 3's optional inverse defect remains an external compatibility limitation;
  no failure has been silently converted into PASS or SKIP.
- Gate 6 covers specific real archive workflows; PWM/research track/benchmark
  lifecycles are audited but not executed. Training/install tasks were not run.
  Generic job failure protocol uses synthetic tasks, not model-training evidence.
- JSON snapshots are processed results/replay settings, not native tensor or
  checkpoint archives. Genome/seed may be unknown, original older snapshots lack
  new metadata, and save-time versions do not establish original replay versions.
- Filesystem status and SQLite are not one atomic store. Pending archive status
  and explicit cleanup flags expose partial failures; durable task recovery and
  arbitrary repeated-disk-failure recovery are not certified.
- CSV escaping mitigates tested formula prefixes but is not a universal guarantee
  for every spreadsheet/import/re-save behavior. Canonical JSON IDs stay exact.

**Recommendation:** accept Step 1 only for the documented tested scope after
review; retain these limitations and the optional five external FAIL findings.
Do not describe the entire site as end-to-end browser verified or biologically
accurate. No Gate 6 failure currently blocks that scoped review if the run is
PASS. Biological usefulness, generalization, calibration and disease/variant
validity require separate authorized analysis; no Step 2 work was performed.
