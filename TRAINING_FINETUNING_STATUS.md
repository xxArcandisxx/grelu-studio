# Training and fine-tuning status

Audit date: 2026-11-23 UTC. Source and historical artifacts only; no training, inference, model deserialization, or tensor inspection was executed. Feature IDs refer to [the inventory](CODEBASE_FEATURE_INVENTORY.md).

## Verdict

**F19 scratch training and F20 all-layer fine-tuning are genuinely implemented, not simulated.** Historical evidence covers compact-CNN browser training/local fine-tuning and real GM12878 fine-tuning through isolated Flask integration. **Freezing/unfreezing, resumable optimizer checkpoints, biological training readiness, and current-browser pretrained fine-tuning remain unsupported or unverified.**

| Capability | Actual implementation | Historical evidence / qualification |
|---|---|---|
| New model creation | `/training` constructs one-output compact CNN; 50–2,000 bp. [lab_views.py](lab_views.py) `page`, training branch; [core/models.py](core/models.py) `_build_demo_model` | [Browser results](test-results/whole-website/results.json) `training`; [real-reference training](test-results/gm12878-fresh-pass/training-result.json). Architecture dropdown is a single choice, not an architecture builder. |
| Optimizer steps / weight updates | [lab_compute.py](lab_compute.py) `train`: `Adam(module.parameters())`, zero_grad, MSE backward, gradient clipping 1.0, step, counter | Saved jobs record nonzero updates (6 or 8 for small batch-1 runs). Positive updates do not guarantee the **exported best state** changed: initial state is eligible. Tensor identity was not rechecked in this audit. |
| Training/validation datasets | `training_rows` requires 8–256 distinct exact-length sequences and finite labels; deterministic seeded row split | Existing 40-record synthetic, 8-record fine-tune, and ten-record real-reference/synthetic-label CSVs. No separate validation-file control or chromosome/group split. |
| Loss / metrics | Selected output task averaged over output dimensions; MSE against scalar label. Initial/best validation MSE, weighted training loss, mean-training-label baseline, history, steps, best epoch | Real metric histories in saved job JSONs. No trainer Pearson, AUROC, classification accuracy or independent-test metrics. Separate benchmark page consumes supplied prediction CSVs. |
| Checkpoint selection | Best state initialized to epoch 0, replaced only when validation MSE strictly improves; restored before saving | `best_epoch=0` and unchanged exported state are legitimate even after all requested epochs/steps. No early stopping: epochs still run. |
| Fine-tuning pretrained model | `/fine-tuning` uses installed spec, CPU/built-in forward, <=4,096 bp and <=25M parameters; same Adam/MSE loop | [Eight-record H result](test-results/low-memory/fine-tuning-result.json), [ten-record H result](test-results/gm12878-fresh-pass/fine-tuning-result.json). Historical **browser** fine-tuning used newly trained compact model, not H. |
| Layer freezing/unfreezing | **Not implemented in website.** All module parameters enter optimizer; existing `requires_grad=False` flags are not explicitly reversed | No controls or branches in `train` or [training form](templates/lab.html). Upstream library capability does not imply website support. |
| Checkpoint registration | `train` saves `model.pt`, then [model_registry.py](model_registry.py) `save_registration`; key `local-<12 job-id chars>`, SHA, length, label, Torch version | Saved [registry](test-results/gm12878-fresh-pass/state/lab/models.json). Both scratch/fine-tune provenance use origin `training`, which is not a biological target label. |
| Reload | [core/models.py](core/models.py) `load_model`, `studio_trained` branch: recursive base-model reconstruction, state dict restore, eval | Historical reloaded predictions in browser and Flask artifacts. Not an arbitrary uploaded checkpoint UI. |
| Predict using trained model | `/predict` resolves registered key and executes normal prediction workflow | [Browser training-prediction export](test-results/whole-website/downloads/training-prediction-051-grelu-analysis.json), [fine-tune prediction export](test-results/whole-website/downloads/fine-tuning-local-prediction-053-grelu-analysis.json); real-reference result JSONs include subsequent prediction. |
| Training checkpoint download | GET `/jobs/<id>/checkpoint`, fixed `jobs/<id>/model.pt`, completed owned jobs only | Real artifacts in [fresh-pass](test-results/gm12878-fresh-pass/); Gate 6 attachment fixture is a controlled literal file, **not trained weights**. |
| Resume/cancel/durable queue | **Not implemented** | Daemon thread, no optimizer resume, cancellation or checkpointed job continuation. Dead-PID status detection is not recovery. |

## Exact source and endpoint map

- [lab_views.py](lab_views.py) `check_csrf` (line 25), `page` (65), `owned_job` (152), `job` (165), `job_status` (171), `checkpoint` (180).
- [lab_compute.py](lab_compute.py) `training_example` (11), `training_rows` (23), `train` (49); setup/split at 59–74, optimizer at 90, update loop at 93, best selection at 107, save/register at 116, baseline/result at 126. These are navigation references, not executed code in this audit.
- [lab_jobs.py](lab_jobs.py) `submit_job` (79): queued state, worker thread, running/complete/failed states, archive handling. `read_job` can write a stale-worker failure, which is why no app endpoint was called during this read-only audit.
- [model_registry.py](model_registry.py) `lab_root`, `registrations`, `save_registration`, `remove_registration`, `sync_models`.
- [templates/lab.html](templates/lab.html), [templates/job.html](templates/job.html), [static/lab-job.js](static/lab-job.js): config, polling every 1.5 seconds, actual epoch curves, metrics, saved-report/model-collection/checkpoint/run-JSON actions. A polling error displays a message and does not schedule another attempt until reload.

Backend flow: POST `/training` or `/fine-tuning` validates data/settings, submits job and redirects to `/jobs/<id>`. That page polls `/api/jobs/<id>`. Completed job can expose `/jobs/<id>/checkpoint`, saved snapshot, and model key. Failure may occur asynchronously after redirect, e.g. invalid target, parameter cap, model load or storage failure.

## Data and configuration contract

| Setting | Supported values | Implication |
|---|---|---|
| CSV schema | Exactly `sequence,label`, in that order/case | No IDs, chromosome/group/split columns, multiple targets or classification task declaration. |
| Rows | 8–256, exact normalized DNA sequences distinct | Exact duplicates rejected; reverse complements/overlap/homology not grouped. |
| DNA | Full native length, A/C/G/T/N after normalization | N encodes zero; no training-specific N threshold. |
| Label | Finite scalar, absolute value <=1,000,000 | Accepted range is not a numerically well-conditioned target recommendation. |
| Scratch model | 3 convolution layers: 4→32 (k9), 32→32 (k9), 32→1 (k1); ReLU between first layers | Not training the published GM12878 architecture from scratch. |
| Scratch length | 50–2,000 bp | Existing 2,114 bp biological windows need externally prepared matching data, not silent crop. |
| Fine-tune length | Required metadata-defined window <=4,096 bp | No arbitrary configured-length override. |
| Parameters | <=25,000,000, checked **after load** | Not a preallocation memory guard. |
| Epochs / LR | 1–20; finite 0.00001–0.01 inclusive | Default form 5 epochs/.001; current example preparation 2 epochs scratch or 1 fine-tune. |
| Batch size | Server config `TRAIN_BATCH_SIZE`, env `GRELU_TRAIN_BATCH_SIZE`; 1–32, default 8 | No per-job UI field; batch-1 config requires the appropriate test server configuration, not a form edit. |
| Request size | Entire request <=600,000 bytes | Max rows and max lengths may not fit simultaneously; 256×4,096 bases already exceeds cap. |
| Target | One valid model track; scratch defaults 0 | Form permits a large integer range; worker checks actual task count. Selecting one task does not freeze unrelated parameters/outputs. |

For N accepted records: `V=max(2, round(.2*N))`, `T=N−V`, updates=`epochs*ceil(T/batch_size)`. Split permutation seed is 0, train order repeats every epoch. Dataset encoding allocates all records in memory; minibatching does not stream CSV or eliminate best-state cloning.

| N | Train / validation | Example |
|---:|---|---|
| 8 | 6 / 2 | Minimum accepted; 75/25 despite UI's simplified “80/20”. |
| 10 | 8 / 2 | Derived hg38 fixtures. |
| 40 | 32 / 8 | Existing synthetic training sample. |
| 256 | 205 / 51 | Parser maximum, body limit still applies. |

Validation selects the checkpoint and is not an independent held-out test set. The website does not protect against locus overlap, reverse complements, donor leakage or cross-chromosome homology.

## Smallest supported and existing ready-to-use recipes

No fixture below was generated or training executed during this audit.

| Recipe | Exact existing data / minimal missing input | Configuration | Expected mechanics / evidence |
|---|---|---|---|
| Absolute smallest scratch smoke | **Missing as a dedicated file:** eight distinct 50-bp known-base sequences with `sequence,label` header and varied actual GC labels. Could later crop bases 76–125 of the first eight [200-bp rows](test-inputs/website-suite/training-200bp.csv), recheck uniqueness and recompute labels | Length 50, epochs 1, LR .001, track 0; batch 1 preferred for observable six updates | Split 6/2; six steps at batch 1, one step at default batch 8. Code-supported, not historically demonstrated exact minimum. |
| Smallest ready existing scratch-length data without preparing a file | [training-200bp.csv](test-inputs/website-suite/training-200bp.csv), all 40 rows; identical training-GC resource also present | Length 200, one epoch .001 for small future check; historical browser config used two epochs | 32/8; 32 steps per epoch at batch 1 or 4 at batch 8. Synthetic learning execution only. |
| Fewest existing labeled rows at 200 bp without new file | [invalid-training-duplicates.csv](test-inputs/website-suite/invalid-training-duplicates.csv) is **invalid**, not a minimum valid dataset | Do not substitute this for a valid eight-row fixture | Must reject exact duplicates before job creation. |
| Ready minimum-count pretrained H fine-tune | [fine-tuning-2114bp.csv](test-inputs/website-suite/fine-tuning-2114bp.csv), 8 rows | H task 0, one epoch, .0001, batch 1 | 6/2, six updates; actual historical reload-for-prediction evidence. Requires base cache/registration availability first. |
| Real hg38-reference scratch execution | [derived training.csv](datasets/gm12878/derived-tests/training.csv), ten 2,000-bp windows | Length 2,000, one epoch, .0001, batch 1 | 8/2, eight updates; labels synthetic GC, windows center-cropped from 2,114 with 57-base offset. |
| Real hg38-reference fine-tune execution | [derived fine-tuning.csv](datasets/gm12878/derived-tests/fine-tuning.csv), ten 2,114-bp windows | H task 0, one epoch, .0001, batch 1 | 8/2, eight updates; synthetic GC labels, not DNase adaptation accuracy. |

Smallest **code-supported** does not mean smallest **existing** file or smallest **measured safe** run. No promise of validation improvement is appropriate for these tiny samples.

## Historical numerical evidence

| Run / evidence | Initial validation MSE | Best validation MSE | Training-mean-label baseline MSE | Update / reload evidence |
|---|---:|---:|---:|---|
| [8-record H](test-results/low-memory/fine-tuning-result.json) | 28.060861 | 1.229744 | **0.023679** | Six steps, batch 1, one epoch, `checkpoint_reloaded_for_prediction=true`; Flask integration, not browser. |
| [10-record compact CNN](test-results/gm12878-fresh-pass/training-result.json) | 0.112856 | 0.084082 | **0.007830** | Eight steps, batch 1, one epoch, prediction payload after registration/reload. |
| [10-record H](test-results/gm12878-fresh-pass/fine-tuning-result.json) | 13.214995 | 0.536247 | **0.007426** | Eight steps, batch 1, one epoch, reloaded prediction. |

These runs improve versus initialization but all lose to the constant baseline on synthetic GC targets. They demonstrate workflow execution, not useful biological learning. The [Phase 1 biological benchmark](Testing/testing-phase-1/REPORT.md) explicitly kept pretrained weights unchanged, so it cannot validate fine-tuned artifacts.

The [older browser bundle](test-results/whole-website/results.json) records scratch and local compact fine-tune passes with curves, checkpoint download and reloaded prediction artifacts. The current runner's `case_fine_tuning_human` function is **not** evidence of execution: no corresponding passing scenario is in that bundle. Gates 1–6 did not train; Gate 6 uses synthetic tasks for generic job protocol only.

## Persistence, portability, permissions, and RAM limitations

- Saved tensor payload contains only `state_dict`, `base_model`, `task_names`; no optimizer state, dataset, split indices, RNG state or history. Run history lives separately in job/snapshot storage. There is no resume control.
- Reload recursively reconstructs the original base specification. A fine-tuned H artifact still needs its original pretrained checkpoint path; chained fine-tunes retain earlier dependencies. Downloaded Studio payload is not automatically a standalone Lightning checkpoint or importable through a browser upload form.
- Registry records SHA-256 but reload does not revalidate against it. Registration is workspace-shared (no owner column), while job access uses the browser session's `lab_owner`; signing in is not equivalent to a private per-account model registry.
- Base task names/training metadata can survive GC-target fine-tuning. An inherited `ENCFF093VXI` label is not evidence that the new model still predicts measured DNase activity.
- Lock serializes within one process, not across every possible server process. Daemon threads can accumulate; no durable queue, timeouts or cancel. Job can appear running while waiting for the lock. Worker interruption is detected, not resumed.
- Saving/registering checkpoint precedes experiment archive; a later archive failure can leave a model artifact/registration even if job reporting fails. Filesystem status and SQLite are not one transaction. Retain explicit failure/cleanup flags.
- Historical [fine-tune resource wrapper](test-results/low-memory/fine-tuning-resources.json) records about 1,563 MB process-tree RSS and ~51 seconds; [fresh-pass report](test-results/gm12878-fresh-pass/REPORT.md) records 1,494 MB maximum across its suite. These are measured old runs, not current guarantees. Full encoding, gradients/Adam buffers and best-state clone add RAM; model loads precede parameter cap.
- Old [sandbox failure](test-results/gm12878-sandbox-attempt/fine-tuning.log) reflects genomepy cache-lock write permissions. Offline model use can still need writable dependency caches. Do not repair caches or restart anything as part of this audit.

Readiness: **ready for a carefully isolated compact regression execution check once manual testing is authorized; pretrained fine-tuning is conditional on visible base checkpoint availability, memory and server configuration. Biological adaptation requires an externally curated independent split and matching measured labels.**
