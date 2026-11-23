# Gate 6 — persistence and export architecture

This map describes actual repository paths. Gate 6 does not add a history CSV
endpoint, browser test framework, training task, or model download.

## Completed results and history

1. `studio_web.compare` handles `/api/compare`; `workbench.run` handles
   prediction, pair/allele/prioritization/multi-allelic analysis and QC;
   `advanced_views.run` handles attribution, design, batch variants and
   mutagenesis; `exploration_views.run` handles PWM scanning, motif simulation
   and model comparison; `curated_scan` handles curated saturation;
   `research_views.run` handles supplied dataset/track/benchmark summaries;
   `lab_views.page` handles exact motif search and job submission.
2. These handlers construct processed results, inputs/settings, warnings and
   provenance. Public inference handlers remove private checkpoint paths and
   transient CSV/FASTA paths. CLI results can intentionally contain local
   artifact paths. A public snapshot is not a raw inference-tensor archive.
3. `experiment_store.save_experiment` validates completed/explicit partial
   results, normalizes supported NumPy values using
   `core.export_values.json_value`, and serializes finite standard JSON.
   It prepares the complete snapshot before inserting one SQLite row and
   commits before returning the ID. Close on failure rolls back an uncommitted
   insert. The returned `saved_run_id` is added to the response **after** save;
   therefore it is not recursively embedded in the stored payload. The export's
   top-level `id` identifies that snapshot.
4. SQLite defaults to `instance/experiments.sqlite3` with WAL and 20-second
   connection timeout; `HISTORY_DATABASE` overrides it. Tests exclusively use
   run-local `state/experiments.db`. Schema: `experiments(id PRIMARY KEY,
   owner,title,workflow,created,payload,path,form,presentation,metadata)`;
   JSON-object columns are text. `owner_aliases(guest PRIMARY KEY,account)`
   supports claiming guest history on sign-in. An owner/date index supports
   listing. Adding the metadata column preserves legacy columns and leaves
   legacy metadata `{}`: no retrospective provenance is invented.
5. `experiment_store.experiment` validates the ID, ownership, JSON object
   columns and (for new records) the canonical payload SHA-256. Corruption is
   visible and never repaired automatically. The hash detects accidental
   changes; it is not a signature against someone who can alter the DB/hash.
6. `experiment_views` exposes owned list/detail, JSON download, HTML report,
   and CSRF-protected delete routes. `report_builder.build_report` renders the
   stored payload/presentation. JSON includes id/title/workflow/created,
   payload, replay form and metadata; presentation is rendered in the HTML
   report, not a separate JSON field. History cards contain title, workflow,
   timestamp, short ID, status and links; **there are no numerical preview or
   model/task fields on the cards**. Model/task and result details belong to
   the detail/report and JSON where supplied.

## Supported exports

| Implementation | Actual output | Lifetime / scope |
|---|---|---|
| `GET /experiments/<id>/record.json` | Owned JSON attachment | Stored processed snapshot and replay settings, strict finite JSON |
| `GET /experiments/<id>/report.html` | Owned standalone HTML attachment | Stored result/presentation, restrictive CSP, no scripts |
| `static/studio.js`, `static/analysis.js` | Client Blob JSON download | Current page state; browser execution remains NOT TESTED |
| `static/advanced.js` | Client Blob CSV; design FASTA | Current design/variant/attribution/mutagenesis results; NOT TESTED in browser |
| `static/exploration.js` | Client Blob CSV | Current PWM/motif/comparison results; NOT TESTED in browser |
| `workflows.predict.handle_predict_workflow` | `predictions.csv` | CLI/output directory; temporary for website handlers |
| `workflows.vep.handle_vep_workflow` | `vep_results.csv` | CLI/output directory; temporary for website handlers |
| `engine.execute_platform_workflow` | `<workflow>_result.json` | Caller-selected output directory; internal local paths may be present |
| `workflows.design` | Designed FASTA text in processed result | Client button uses that text; no new server FASTA route |
| `GET /jobs/<id>/checkpoint` | Fixed owned `model.pt` attachment | Only a complete job advertising a checkpoint; missing file returns 404 |

No `/experiments/<id>/record.csv` route exists. Testing an engine CSV does not
establish that clicking a browser CSV button works. JSON retains original IDs;
CSV text escaping is an explicitly different representation (see BUGS_AND_FIXES).

## Background jobs and partial failures

`lab_jobs.submit_job`, `update_job`, `read_job`, `_write_state` use
`LAB_ROOT/jobs/<hex-id>/status.json`. States are queued/running/complete/failed;
they include UTC epoch timestamps, worker PID, result, message and history.
`model_registry.lab_root` creates the configured workspace root. Read-only
missing-job lookup no longer creates individual job folders. Model registrations
are separate `LAB_ROOT/models.json` entries with spec/provenance. Registration,
installation, training and optimizer behavior are outside this test execution.

Status updates write strict JSON to a sibling temp file then atomically replace
the old status. Temp-write/rename failure preserves prior bytes; failed temp
files are cleaned without deleting other outputs. IDs are allocated in distinct
directories with bounded collision retries. The process-local RLock serializes
status writes. It is not a cross-process distributed lock or durable queue.

After a task succeeds, its archive is initially **pending**. The worker writes
complete status including the archive ID, then promotes archive metadata to
complete. An exception causes compensating owned-archive deletion and a failed
job with no result. If deletion also fails, the archive remains pending and
`archive_cleanup_pending=true` is visible. There is no atomic transaction across
SQLite and JSON files. A crash can leave pending metadata for review; repeated
disk failure can prevent even a failed-state write. Those conditions are not
guaranteed recoverable, and no automatic resumption/reconciliation is claimed.
Reading a queued/running job with a demonstrably dead PID marks it failed.
The generic protocol is exercised with synthetic tasks, not trained models.

## Ownership, retention and reproducibility

Guest experiments belong to a signed Flask session's random lab token; active
accounts use user IDs. Account login may claim guest history through aliases.
Job access still uses the session lab token. This is the existing local security
model; Gate 6 does not certify authentication, email delivery or deployment.
Different guests and seeded active accounts are tested against isolated records.

History/detail/JSON/report and job-status responses use no-store. Titles are
escaped by Jinja. JSON download filenames use a sanitized basename, and
checkpoint download ignores result-supplied path hints. Public errors omit
diagnostic paths/input; full diagnostics remain in server logs.

New snapshot metadata records schema/status, UTC save time, installed Python
and NumPy/torch/gReLU/Flask/pandas versions, canonical payload/input hashes,
available model IDs/checkpoint hashes/tasks, known genome assembly and seed,
plus retention/raw-tensor policies. **Versions describe save time**; they do
not assert when a replayed result was computed. Original computation provenance,
settings, transformations, warnings and fallback notes remain in the payload.
Model comparison keeps separate identity/task entries. Unknown genome/seed
remain JSON null; descriptive analyses have no invented model identity.

Existing full inputs and replay forms are retained locally until explicit
snapshot deletion. Hash metadata adds no duplicate DNA strings. Keeping full
inputs supports existing repeat-run functionality; Gate 6 does not replace it
with hashes. Secrets, uploaded/internal path form keys and executable replay
URLs are filtered. False/null HTML form controls are intentionally omitted;
false/null/zero/missing **payload** values remain distinct.

There is no automatic history expiry, job garbage collector or user-export
cleanup. Deleting a snapshot does not delete downloaded files, model weights
or separate job artifacts. Website inference temp directories use context-managed
`TemporaryDirectory`; CLI directories are retained. Tests retain their isolated
DBs, artifacts and intentional corruption fixtures as evidence.
