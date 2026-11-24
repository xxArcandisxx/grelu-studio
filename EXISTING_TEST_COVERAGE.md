# Existing test coverage — evidence audit

Audit date: 2026-11-23 UTC. **No tests or browser actions were performed during preparation.** Historical test results below are not new passes. Source implementation and completed result artifacts were inspected separately.

## Evidence standard

- **Backend verified:** actual executed backend checks for named inputs/invariants; may be numerical oracle, integration, storage or unit regression. Always read the scope.
- **Browser verified (historical):** retained completed Chromium scenario, screenshot/download artifacts and runner provenance. Does not certify current source after later changes.
- **Partially verified:** only bounded branches, generic protocol, historical subset or non-browser layer covered.
- **Not tested:** no completed result for that capability/layer; a runner/test function alone is not execution evidence.
- **Blocked:** required resource/capability unavailable or previously blocked run; not a substitute for partial coverage.

Flask test-client responses, static JavaScript, server-rendered HTML, metadata-only adapters and saved-result replay **are not real browser execution**. A screenshot proves a rendered state, not every action or numerical correctness. The earlier shared-tab inspection in this conversation established access/rendering only and did not run inference.

## Gates 1–6, latest retained runs

| Gate | Latest UTC run / result | Actual execution and important exclusions |
|---|---|---|
| 1: identity | [20261122T141545575344Z](Testing/testing-phase-2/input-output/runs/20261122T141545575344Z/REPORT.md): **39 PASS, 0 FAIL, 2 SKIP** | Real checkpoint/native/website weight identity and metadata/inspector HTTP checks, conflicting length rejected. No DNA prediction, live-server attestation or browser. Workers recorded ~4/11/14 seconds for raw/direct/website. |
| 2: DNA integrity | [20261122T141643941401Z](Testing/testing-phase-2/input-output/gate-2/runs/20261122T141643941401Z/REPORT.md): **265 PASS, 0 FAIL, 12 SKIP** | Actual Flask validation/parsing with metadata-only probe adapters stopping before encoding/inference. Eleven genome/BED skips and one browser skip. [verify_gate2.py](Testing/testing-phase-2/input-output/gate-2/verify_gate2.py) `ProbeAdapter.predict` deliberately stops execution. |
| 3: preprocessing | [20261122T142625611276Z](Testing/testing-phase-2/input-output/gate-3/runs/20261122T142625611276Z/REPORT.md): **150 PASS, 5 FAIL, 8 SKIP** | Real forward-boundary input tensors matched independent literal encoding, stopped before weighted computation. Five installed-gReLU singleton inverse failures remain. [Closure](Testing/testing-phase-2/input-output/gate-3/investigation/REPORT.md) separates required website subset **148 PASS/0 FAIL/8 SKIP** from optional dependency **2 PASS/5 FAIL**, not a replacement all-pass run. ~15.6 sec. |
| 4: numerical inference | [20261122T163955672475Z](Testing/testing-phase-2/input-output/gate-4/runs/20261122T163955672475Z/REPORT.md): **28 PASS, 0 FAIL, 7 SKIP** | Genuine pretrained CPU outputs: 19 distinct queries, 93 website forwards, 136 full-output comparisons against independent native gReLU, exact equality for tested cases. Native `(batch,1,1)`. Website ~118 sec/direct ~36 sec. Full saturation, distinct second checkpoint, gradients/design optimization, training/browser/GPU not certified. |
| 5: processing/presentation | [20261122T180450337008Z](Testing/testing-phase-2/input-output/gate-5/runs/20261122T180450337008Z/REPORT.md): **273 PASS, 0 FAIL, 5 SKIP** | Approved Gate 4 raw-tensor replay through actual processing, synthetic edge cases, plus fresh real bounded design/attribution. Actual Flask HTML/chart datasets and static JS, **not dynamic browser rendering**. Processing ~10 sec/fresh ~33 sec. [Workflow matrix](Testing/testing-phase-2/input-output/gate-5/runs/20261122T180450337008Z/WORKFLOW_COVERAGE.md). |
| 6: persistence/exports | [20261123T110400114845Z](Testing/testing-phase-2/input-output/gate-6/runs/20261123T110400114845Z/REPORT.md): **586 PASS, 0 FAIL, 5 SKIP** | Real isolated SQLite, endpoint downloads, generated engine CSV, separate-process retrieval; 470 replay + 53 synthetic serialization + 32 synthetic job + 10 fresh inference/storage + 20 regression + 1 preservation passes. Storage ~33 sec/fresh ~49 sec/regressions ~8 sec. No actual training/install tasks, browser-triggered exports or universal workflow lifecycle. [Coverage matrix](Testing/testing-phase-2/input-output/gate-6/WORKFLOW_COVERAGE.md). |

Assertion counts are not sample counts or biologically validated workflows; do not sum them as independent evidence. Earlier gates were not all rerun in Gate 6. The parent [input-output README](Testing/testing-phase-2/input-output/README.md) still says Gate 6 has not started; the newer [Gate 6 summary](Testing/testing-phase-2/input-output/gate-6/STEP_1_SUMMARY.md) and latest pointers supersede that stale status.

### Gate provenance and independence

- [verify_gate1.py](Testing/testing-phase-2/input-output/verify_gate1.py) `worker`: actual Flask client, identity only.
- [verify_gate3.py](Testing/testing-phase-2/input-output/gate-3/verify_gate3.py) `oracle`/input hook: independent literal encoding, not an output prediction oracle.
- [direct_reference.py](Testing/testing-phase-2/input-output/gate-4/direct_reference.py) `main`: native gReLU predictions; [website_worker.py](Testing/testing-phase-2/input-output/gate-4/website_worker.py) `observed_forward`: captures delegated real forwards.
- [verify_processing.py](Testing/testing-phase-2/input-output/gate-5/verify_processing.py) `Replay` and [verify_fresh.py](Testing/testing-phase-2/input-output/gate-5/verify_fresh.py) separate replay arithmetic from new gradients/design work.
- [verify_storage.py](Testing/testing-phase-2/input-output/gate-6/verify_storage.py) `Replay`, `observe_save`, `restart` delegate to real saver/storage; [fresh worker](Testing/testing-phase-2/input-output/gate-6/verify_fresh.py) separately executes fresh compare/download. Synthetic job protocol and literal checkpoint attachment fixture do not establish optimizer/weight behavior.

## Historical real-browser evidence

| Bundle | Completed results | What the artifacts support |
|---|---|---|
| [browser pilot](test-results/browser-pilot/results.json) | Nov 17 03:50:29–03:50:42 UTC, **3 passed, 0 failed**, 4 browser download events | Chromium headless variant-demo, predict-demo, PWM JASPAR; three screenshots, seven saved download/report files. |
| [whole website](test-results/whole-website/results.json) | Nov 17 03:51:24–03:53:46 UTC last recorded progress, **36 passed, 0 failed**, 53 browser download events | Thirty-eight screenshots and 91 download/report files. Includes result JSON/CSV/FASTA/checkpoint actions for completed scenarios. HTML reports are fetched via request client in `capture`, not necessarily opened/rendered in browser. |
| [phase-2 browser-e2e plan](Testing/testing-phase-2/input-output/browser-e2e/SETUP_PLAN.md) | Setup plan only, untracked before audit | No phase-2 completed browser results, screenshots or trace in this directory. Installation intent is not successful execution. Preserved unchanged. |

The [runner](tests/run_website_suite.py), `WebsiteSuite.upload`, `native_submit`, `download`, `capture` and Chromium-launch code establish actual browser provenance for the recorded scenarios. The [server helper](tests/serve_website_suite.py) uses a real isolated HTTP server and cached H checkpoint. The runner was inspected, never invoked.

The **36-case run was interrupted**. The [current manifest](test-inputs/website-suite/manifest.json) has **43 cases**, eight without a completed old result: dataset FASTA, dataset CSV, H fine-tuning, accounts, Model Zoo cache registration, history/reports, mobile, negative-input flows. The old 36 include retired BED validation, so only 35 map to active scenarios. The browser summary's generic cached-registration claim is not supported by a completed browser case. `write_results` saves after every scenario and the server log continues after its `finished` value; that timestamp is not suite closure. See [README section 15.2](README.md) and [server.log](test-results/whole-website/server.log).

No exact historical browser-version/source-hash/fixture-hash attestation is retained in that result JSON. It predates November 22–23 fixes. Current local [browser tooling](.test-tools/) exists, but installed files do not prove successful current launch. No tooling was installed or launched here.

## Every feature: backend and browser classification

`Historical PASS` below refers only to named November 17 scenarios. **Current full manual/browser verification is outstanding for all 30 groups.** `No browser result` is not negated by backend evidence. See [source inventory](CODEBASE_FEATURE_INVENTORY.md) for feature definitions and functions.

| ID / feature | Backend status / evidence scope | Real-browser status | Overall preparation classification |
|---|---|---|---|
| F01 Prediction/batch/shuffles | G4 raw tested, G5 processing partly verified, G6 persistence; R real predictions | Historical PASS D/H + shuffles | **Partially verified** current workflow; positive-shuffle independent statistics limited. |
| F02 SNV comparison | G4 first/middle/last API, G5 arithmetic, G6 save/download | Historical PASS D/H | **Backend verified** bounded SNVs; current singleton plot/UI pending. |
| F03 Allele-specific | G4–6 effect mapping/RC/storage | Historical PASS D/H | **Backend verified** supplied pairs; biological allele mechanism not established. |
| F04 Prioritization | G4–6 deltas/ranks/storage | Historical PASS D/H | **Backend verified** supplied SNVs. |
| F05 Multi-allelic | G4–6 supplied alternate mapping | Historical PASS D/H | **Backend verified** supplied comparisons; broader combinatorial function partial. |
| F06 Batch variants | G4 inference, G5 ranks/format, G6 engine CSV | Historical PASS D/H | **Partially verified**; positive-n background path unverified independently. |
| F07 Curated saturation | G4–6 one-position subset; current full end-to-end backend incomplete | Historical PASS ACTB full small grid | **Partially verified**, exact model availability conditional. |
| F08 Mutagenesis | G4–6 one-position/three substitutions | Historical PASS D/H | **Partially verified** wider/N branches. |
| F09 Attribution | G5 fresh full real 4×2114 matrix/formula; G6 snapshot | Historical PASS D/H | **Backend verified** bounded fresh matrix; revised browser chart/export pending. |
| F10 Design | G5 one iteration/two candidates/five backgrounds; G6 sequence/history | Historical PASS D/H | **Partially verified**, larger search not certified. |
| F11 Motifs/k-mers | R actual exact/enrichment computation; G6 limited stored hit | Historical PASS | **Partially verified** scope, no biological motif annotation. |
| F12 PWM | R actual scan, historical formats | Historical PASS JASPAR/MEME | **Partially verified**; G6 specific persistence lifecycle not tested. |
| F13 Motif simulation | G4 real single condition; G5/6 limited replay | Historical PASS D/H | **Partially verified** combinations/positive controls. |
| F14 Model comparison | G4–6 same H/task; R H vs synthetic-GC adapted model | Historical PASS D/H | **Partially verified**; no distinct second published-model benchmark. |
| F15 QC | G2 parser, R descriptive results, G6 storage | Historical PASS D/H | **Backend verified** named descriptive fixtures. |
| F16 Dataset overview | Six executed numerical/integration tests in G6 regression log, limited G6 save/restart | **No browser result** | **Backend verified** tested fixtures; browser **not tested**. |
| F17 Benchmark | Phase 1 five HTTP checks include metrics; R utilities | Historical PASS regression/binary | **Backend verified** named arithmetic; G6 specific lifecycle untested. |
| F18 Track analysis | R synthetic numerical utility | Historical PASS | **Backend verified** small fixture; G6 specific lifecycle untested. |
| F19 Training | R actual steps/save/reload; not trained in Gates | Historical PASS compact training/prediction | **Partially verified** current lifecycle, minimum 50bp case not tested. |
| F20 Fine-tuning | R and low-memory actual H steps/reload | Historical PASS local compact only; **H browser not tested** | **Partially verified** pretrained acceptance; biological quality unsupported. |
| F21 Inspector | G1 real checkpoint identity and metadata/HTTP | Historical PASS D/H | **Backend verified** pinned model, current UI pending. |
| F22 Model management | R cache-only H registration/predict/remove | **No completed browser result** | **Partially verified**; remote/catalog coverage not tested. |
| F23 Examples | Seven executed G6 regression tests; load does not call model loader/jobs; download bytes checked | **No browser result** for current shared controls | **Backend verified**, frontend races/input reset not tested. |
| F24 History/filter/pages | G6 real SQLite ownership/filter/pagination/restart | **No completed browser result** | **Backend verified**, UI not tested. |
| F25 Detail/reports/exports | G6 real owned JSON/HTML + engine CSV; W actual browser downloads | Historical browser download PASS; report rendering mostly HTTP, not browser | **Partially verified** revised CSV/FASTA/standalone detail interactions. |
| F26 Editable replay | G6 whitelist/retained fields/new-run routes; original snapshots preserved | **No completed browser rerun result** | **Partially verified** actual edited replay UI. |
| F27 Delete record | G6 owned deletion/missing record/CSRF/storage | **No completed browser deletion result** | **Backend verified**, UI not tested. |
| F28 Accounts | Executed credential/session regression; G6 ownership/guest migration; registration/token tests also exist in source | **No completed browser result** | **Partially verified** local flow; SMTP **blocked/not implemented**. |
| F29 Settings/avatar | Seven executed account-settings tests in G6 regression log | **No completed browser result** | **Backend verified**, UI not tested and small image fixture needed. |
| F30 Responsive/navigation feedback | Source only for dynamic navigation; form validators partly checked backend/static | **No completed mobile/keyboard result** | **Not tested** as interactive navigation; rendering alone not behavioral coverage. |

Summary by testing-layer definition: **9 feature groups have no completed historical end-to-end browser flow** (F16, F22–24, F26–30); F25 has only partial browser support by export type and F20 lacks pretrained-browser execution. **One group (F30 interactive navigation) has no retained behavioral execution evidence at any layer.** This is not a claim that only one feature needs new manual testing: every current group needs freshness checks.

## Other retained evidence, scientific limits and failures

- [Fresh-pass report](test-results/gm12878-fresh-pass/REPORT.md): 17 sequential feature groups passed, real reference contexts and real H weights through Flask clients; derived variants/GC labels synthetic. [Earlier sandbox attempt](test-results/gm12878-sandbox-attempt/REPORT.md): 5 pass/11 fail/1 not tested, including genomepy cache-lock permissions. Later successes do not erase that evidence.
- [Low-memory fine-tune](test-results/low-memory/fine-tuning-result.json): eight records, six updates, checkpoint reload, explicitly no browser. [Resource record](test-results/low-memory/fine-tuning-resources.json): ~1,563 MB peak tree RSS/~51 sec.
- [Phase 1 report](Testing/testing-phase-1/REPORT.md): real pretrained DNase/dsQTL benchmark, five Flask website checks, not browser. 500 chr11 DNase windows, 1,000 hg19 dsQTL variants with only 16 positives. Do not infer causal variant/design/attribution validity or calibrated probabilities.
- [Concerns review](Testing/testing-phase-1/concerns-review/CONCERNS_REVIEW.md): no exact train/test sequence or reverse-complement matches in audited DNase split, but homology unresolved; dsQTL chromosome-held-out claim unsupported; paired AP advantage interval includes zero. Correlation and AUROC are not percentage accuracy.
- Gate 3 remaining singleton inverse defect: native A/C/G/T singleton conversion yields N; singleton N raises IndexError. Closure audited affected inverse as unused by website. Retain **FAIL**, do not silently relabel.
- Prior gate failed/budget-blocked runs and harness fixes remain under timestamped runs and `BUGS_AND_*` reports: identity mismatch handling, parser fixes, Gate 4 fallback warning, Gate 5 signed HTML expectations, Gate 6 save-observer binding. These are distinguishable from outstanding production failures.

## Evidence integrity and source drift

Read-only delegated hash comparison found latest recorded artifact manifests for Gates 3–6 internally consistent: 324/677/374/732 entries respectively matched; Gate 1/2 cited evidence exists; 87 Phase 1 convenience-copy/original entries matched [COPY_MANIFEST.json](Testing/testing-phase-1/COPY_MANIFEST.json). This is internal integrity, not independent authenticity or fresh workflow execution.

Current source differs from recorded source inventories for G1 (12/27 entries), G2 (7/17), G3 (6/8), G4 (8/25), G5 (8/63). G6's selected 13-file [source inventory](Testing/testing-phase-2/input-output/gate-6/runs/20261123T110400114845Z/source-inventory.json) matches, but does not cover the entire app. Older results must remain dated and scoped. Browser outputs/tooling can be ignored by [.gitignore](.gitignore), so tracked cleanliness alone is not evidence retention.

No existing log, report, screenshot, checkpoint, fixture, database or setup plan was rewritten to reconcile contradictory/stale status language. New audit documents state those discrepancies explicitly.
