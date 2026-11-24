# Gaps and recommendations

Audit date: 2026-11-23 UTC. Read-only source/resource/evidence inspection; **testing has not started**. Recommendations below are future work, not code changes, downloads or fixture generation performed here.

## Summary

| Measure | Finding |
|---|---|
| Website features | **30 groups: 22 tool workflows +8 supporting capabilities.** [Inventory/counting](CODEBASE_FEATURE_INVENTORY.md). Batch/shuffle prediction and regression/binary evaluation are subfeatures, not duplicate counts. |
| Fully implemented | **27**, within their explicitly bounded local scope; not a claim of complete browser or scientific validation. |
| Partially implemented | **3:** F05 supplied multi-allelic scoring lacks full combination generation/epistasis; F22 catalog-wide model compatibility is partial; F28 activation/recovery has no production email transport. |
| Not tested | **1 group** lacks retained behavioral execution evidence: F30 interactive responsive/navigation behavior. **9 groups lack a completed historical browser flow:** F16, F22–24, F26–30. All30 need current-browser freshness checks; pretrained F20 and revised F25 exports are additionally partial. |
| Historical browser run | **36 completed scenarios**, zero recorded failures among completed cases, interrupted old suite including retired BED. Current manifest43 cases; eight lack a completed old result. |
| Backend gates | Gates1/2/4/5/6 recorded PASS. Gate3 remains FAIL:150 PASS/5 FAIL/8 SKIP; required website subset148 PASS is separated from optional dependency failures, not relabeled. [Coverage](EXISTING_TEST_COVERAGE.md). |
| Existing testing data | Ten real hg38 windows;500 chr11 DNase records with measured labels;1,000 sampled hg19 dsQTL variants,16 positives; local hg38/hg19 2bit, DNase bigWig, supplement; small200/2114bp synthetic FASTA, CSV, motif and negative fixtures; saved score tables; GC-target training files. |
| Main resource blocker | Original pinned GM12878 snapshot/blob absent in visible filesystem; current live process/resource namespace not attested. Old fine-tuned files retain base-model dependencies. |
| Missing inputs | Dedicated8×50 training fixture, small biological batch subset, purpose-built avatar and extended encoding/size/ID-export fixtures; biological motif database, independent second pretrained model, measured-target externally partitioned training/validation/test CSVs. None generated. |
| Training readiness | Actual Adam/MSE/validation/checkpoint/registration/reload/prediction. Historical compact-browser and H-Flask evidence. No freeze/unfreeze, optimizer resume, architecture builder or validated biological adaptation. |
| Effort | Smoke20–30 min; comprehensive core about4–6.5 hands-on hours with resources ready; another1–2 hours for missing fixtures/isolated restart/edges. Resolving model availability is additional and unknown. |

## P0 — presentation readiness

### Establish current pretrained identity and resource visibility

The old snapshot and blob in [checkpoint metadata](Testing/testing-phase-2/input-output/runs/20261122T141545575344Z/checkpoint.json) return ENOENT in this audit's filesystem; the default registry file is absent. [run_studio.sh](run_studio.sh) uses cache-only lookup. An earlier shared-page H label does not prove the current process can reopen its file or shares this filesystem namespace.

In a later authorized testing session, record actual server model key/hash/window/task and file access. If unavailable, mark H workflows BLOCKED and resolve the environment/resource in a separate task. Do not silently download weights, substitute D, or register archived artifacts. No endpoint, app import or deserialization was used to probe this during the audit.

### Verify current browser behavior, not old passes alone

[Whole-website results](test-results/whole-website/results.json) support historical real Chromium execution, not current full acceptance. [README](README.md) documents the interrupted36/42 old run; the current [manifest](test-inputs/website-suite/manifest.json) has43 cases after BED removal/overview additions. Generic cached-install summary text is not a completed browser result. The [phase-2 setup plan](Testing/testing-phase-2/input-output/browser-e2e/SETUP_PLAN.md) has no completed execution bundle.

Use the [manual plan](MANUAL_BROWSER_TEST_PLAN.md) smoke first. Prioritize pooled H output, model/length selection, non-submitting Load example/race handling, upload preview reset, changed CSV/FASTA exports, standalone reports, keyboard and mobile behavior. Preserve old bundles.

### Keep biology and coordinates accurate

- D computes with random weights: pipeline evidence only. [core/models.py](core/models.py), _build_demo_model/load_model.
- H has one pooled task output, not a dense genomic track. [Gate6 summary](Testing/testing-phase-2/input-output/gate-6/STEP_1_SUMMARY.md).
- dsQTL uses hg19 and published allele1→allele2. In817/1000 sampled rows allele1 is not genomic reference. Its context uses displayed position1057; CLI extraction and many synthetic fixtures use1058. [Preparation](scripts/benchmark_gm12878.py), allele_sequences/prepare/evaluate; [VEP](workflows/vep.py), _fetch_context. Never move a variant without changing its corresponding context.
- Constructed variants, synthetic GC targets and artificial PWMs validate execution, not biological effects. The measured DNase/dsQTL evaluation does not validate designs, attribution mechanisms or causal alleles. [Phase1 report](Testing/testing-phase-1/REPORT.md), [concerns review](Testing/testing-phase-1/concerns-review/CONCERNS_REVIEW.md).
- Transformed dsQTL rankings are not calibrated probabilities; Brier/calibration is not biological evidence. Only16 positives and a paired AP difference interval spanning zero limit claims versus gkmSVM. Correlation/AUROC are not percentage accuracy.

## P1 — inputs and computation

| Gap | Current evidence | Recommended later action — not performed |
|---|---|---|
| Biological24-row file exceeds H budget | [advanced_views.py](advanced_views.py), run: rows×(4+2×shuffles)≤48 non-D; [batch24](datasets/gm12878/biological/website-inputs/batch_variants_24.csv) has24 rows | Preserve original; prepare a separate1–3-row biological fixture or paste≤12 rows with zero shuffles. Full-file rejection is a useful invalid case. |
| Full biological files exceed page limits | [core/dataset.py](core/dataset.py):200k total bases/550KB; standard parser12 records; DNase FASTA500 records/1,057,000 bases | Prefer [12-record FASTA](datasets/gm12878/biological/website-inputs/prediction_12.fasta), [three contexts](datasets/gm12878/derived-tests/comparison.fasta), [tiny overview](test-inputs/new-workflows/dataset-overview.txt). No genome/bigWig upload is implemented. |
| Absolute minimum training file absent | [lab_views.py](lab_views.py): scratch50–2000bp; [lab_compute.py](lab_compute.py), training_rows: minimum8 distinct | Later prepare8×50bp with truthful varied GC labels and check uniqueness after cropping. Existing40×200 fixture is ready without preparation. |
| Avatar/boundary inputs absent | [account_views.py](account_views.py), profile_picture; no dedicated test-input image | Later create64×64 still PNG plus invalid/oversized/metadata-bearing variants. Do not use a giant page screenshot as the primary avatar input. |
| Biological motif resource absent | [core/pwm.py](core/pwm.py); local AT-rich matrix is synthetic | Synthetic format/score tests ready; biological TF annotation needs a curated database and independent truth set. No download now. |
| Independent second published model absent | Catalog strings in [studio_web.py](studio_web.py); tests use H/H or H/GC-adapted output | Same-model reproduction/window handling ready; an independent model requires compatible weights and meaningful task/scale selection. |
| No prepared measured adaptation split | [dnase.csv](datasets/gm12878/biological/dnase.csv) has measured values but wrong training schema and is held-out chr11; current training labels are GC | Separately design nonoverlapping train/validation/independent test data and supported CSVs. Do not train on the benchmark test set and call the result held-out accuracy. |
| RAM/CPU exposure | H imports~1.2–1.5GB historically; gradients/Adam/best-state clones; parameter cap after load;600KB body cap | Keep ranges/searches small and sequential; batch1 for H jobs where server is already configured appropriately. A31.9MB checkpoint does not imply31.9MB runtime RAM. |

### Branches needing particular attention

- Positive-n batch statistics: empirical paired-shuffle p values exist, but independent numeric/browser coverage is limited. One shuffle yields p=.5 or1, not meaningful .05 significance. Current H evaluation budgets restrict large background counts. [workflows/vep.py](workflows/vep.py), handle_vep_workflow; [Gate5 matrix](Testing/testing-phase-2/input-output/gate-5/runs/20261122T180450337008Z/WORKFLOW_COVERAGE.md).
- Full curated scan, wider/N-mutagenesis, motif partners/orientations/controls, randomized attribution copy and larger design searches have narrower latest-gate coverage than their full capability. Do not fill untested positions with invented values.
- CSV text escaping/precision changed after the old browser run. Recheck actual Blob downloads for numeric minus, strand+/−, Unicode/quotes/formula-like IDs and blank nulls; JSON IDs must remain canonical. [static/advanced.js](static/advanced.js), [static/exploration.js](static/exploration.js), [core/export_values.py](core/export_values.py), [Gate6 fixes](Testing/testing-phase-2/input-output/gate-6/BUGS_AND_FIXES.md).
- Malformed-encoding upload errors need checks: [workbench.py](workbench.py) upload decode lacks the explicit UnicodeError catch used by research/lab routes. Source inspection suggests a500 rather than friendly400 is possible. This is a **hypothesis for later testing**, not a reproduced bug.
- Current Load example prepares fields without submitting. Legacy POST example branches execute workflows; the legacy fine-tune branch generates40 rows while the current loader supplies8. Test the actual current button. [example_views.py](example_views.py), build_example; [lab_views.py](lab_views.py), page; [static/examples.js](static/examples.js).

## P1 — training and persistence

Full details: [TRAINING_FINETUNING_STATUS.md](TRAINING_FINETUNING_STATUS.md).

1. All-parameter Adam/MSE exists; freezing, head replacement, classification/multiple targets, optimizer resume, scheduler and early stopping do not. Upstream library methods are not wired website controls.
2. Eight rows split6/2, not exactly80/20. Validation selects the checkpoint; it is not an independent test set. Exact duplicates are rejected, but related loci/RC/homology are not grouped.
3. Best checkpoint may be initial epoch0 after positive optimizer steps. Test update count/selection/reload, not mandatory improvement. Recorded GC runs lose to a constant baseline.
4. Fine-tuned payloads recursively need base files and are not necessarily portable standalone bundles. No arbitrary checkpoint-upload UI; registry hashes are recorded but not enforced on reload. Base task labels/performance metadata can survive target changes.
5. Jobs are session-owned, registry workspace-shared. Login does not create a private model collection. Removing a registration retains weights/cache; only remove a new test registration.
6. Daemon threads and an in-process lock are not a durable queue, cancellation or resume. Job may appear running while waiting for lock; poll failure needs reload. Avoid multiple concurrent jobs.
7. Checkpoint/registration can precede archive failure. Filesystem and SQLite are not one transaction; Gate6's partial-state/cleanup tests are not universal disk-failure recovery. Inject faults only in new isolated state, later.
8. Saved JSON contains processed outputs/settings, not raw inference tensors or weights. Save-time versions need not be original computation versions; missing assembly/seed metadata must remain unknown rather than invented.

## P2 — unsupported or future capabilities

Not implemented as current website controls: BED/VCF/assembly extraction or liftover, automatic combinatorial variant generation/epistasis, bigWig/genomic track upload, TF-MoDISco/SHAP/DeepLIFT, custom training architectures/freezing/optimizer resume, universal saved CSV/FASTA regeneration, server PDF export, SMTP activation/recovery and universal catalog architecture compatibility.

[bed_to_fasta.py](bed_to_fasta.py) is a separate network-enabled CLI; it and the local2bit resources do not create a website BED page. Keep the old BED pass as retired-feature evidence. The [inventory](CODEBASE_FEATURE_INVENTORY.md) names the implemented alternatives.

## Evidence maintenance

- Do not rewrite old reports/manifests/screenshots to reconcile stale prose. Parent README's “Gate6 not started” is superseded by newer completed Gate6 evidence.
- Preserve Gate3's five optional inverse-conversion failures. An audited unused path is a scope qualification, not a historical PASS.
- Preserve old sandbox failures, blocked runs, baselines and harness corrections. Internal hash consistency is not independent authenticity.
- Future manual evidence should include source/version/fixture IDs, actions, screenshots, outputs, resource observations and explicit blocked/untested branches. A test function or installed Chromium is not an executed check.
- This audit did not authorize or perform production fixes, deployment, downloads, training/inference, browser execution or an automated test run.

## Deliverables

1. [CODEBASE_FEATURE_INVENTORY.md](CODEBASE_FEATURE_INVENTORY.md)
2. [DATASET_RESOURCE_INVENTORY.md](DATASET_RESOURCE_INVENTORY.md)
3. [FEATURE_DATASET_MAPPING.md](FEATURE_DATASET_MAPPING.md)
4. [TRAINING_FINETUNING_STATUS.md](TRAINING_FINETUNING_STATUS.md)
5. [EXISTING_TEST_COVERAGE.md](EXISTING_TEST_COVERAGE.md)
6. [MANUAL_BROWSER_TEST_PLAN.md](MANUAL_BROWSER_TEST_PLAN.md)
7. [GAPS_AND_RECOMMENDATIONS.md](GAPS_AND_RECOMMENDATIONS.md)

Only these new reports are intended writes. Existing untracked browser setup evidence and all production/data/checkpoint/prior-evidence files remain untouched. **Stop after audit delivery; testing has not started.**
