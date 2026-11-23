# Gate 6 — workflow coverage

VERIFIED means the stated persistence/export capability was exercised for the
listed fixtures. It does not certify the numerical method beyond earlier gates.
PARTIAL denotes limited cases or generic protocol only. UNSUPPORTED means no
such app capability exists; NOT TESTED means it exists but was not executed.
Actual Flask HTML is server integration evidence, not browser execution.

| Workflow / fixture | Persistence | Retrieval / detail | CSV download | JSON download | History display | Restart durability |
|---|---|---|---|---|---|---|
| Prediction: FASTA five records, N, duplicate DNA/distinct IDs, upload | VERIFIED | VERIFIED | NOT TESTED client / UNSUPPORTED history; generated engine CSV VERIFIED | VERIFIED owned endpoint | VERIFIED cards/links | VERIFIED fresh process |
| Pair variant: first-base ref/alt + one fresh real API request | VERIFIED | VERIFIED | UNSUPPORTED history; engine prediction CSV VERIFIED separately | VERIFIED | VERIFIED | VERIFIED replay process; fresh new connection |
| Allele-specific | VERIFIED | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| Prioritization | VERIFIED including signed effects/stable absolute ranks | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| Multi-allelic | VERIFIED reference/alternate mapping | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| Batch variants: positions 1/1057/2114 | VERIFIED | VERIFIED | NOT TESTED client; engine VEP CSV VERIFIED | VERIFIED | VERIFIED | VERIFIED |
| Mutagenesis: one position / three substitutions | VERIFIED limited snapshot | VERIFIED | NOT TESTED client | VERIFIED | VERIFIED | VERIFIED |
| Curated saturation: one position only | VERIFIED limited snapshot; full scientific workflow PARTIAL | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| Attribution: accepted Gate 5 full processed matrix | VERIFIED matrix/positions/track/score | VERIFIED | NOT TESTED client | VERIFIED | VERIFIED | VERIFIED |
| Design: one iteration / two candidates / one editable base | VERIFIED sequence/history/scores | VERIFIED | NOT TESTED client; FASTA button NOT TESTED | VERIFIED | VERIFIED | VERIFIED |
| Motif simulation: one replacement, no controls | VERIFIED limited snapshot | VERIFIED | NOT TESTED client | VERIFIED | VERIFIED | VERIFIED |
| Model comparison: same checkpoint/task in both slots | VERIFIED two identities/tasks and zero differences | VERIFIED | NOT TESTED client | VERIFIED | VERIFIED | VERIFIED |
| Descriptive QC | VERIFIED duplicate inputs/Unicode IDs | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| Dataset overview | VERIFIED two records / one duplicate | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| Exact motif search | VERIFIED specified hit coordinates/ID | VERIFIED | UNSUPPORTED history | VERIFIED | VERIFIED | VERIFIED |
| PWM scanning | NOT TESTED specific lifecycle (shared saver audited) | NOT TESTED | NOT TESTED client | NOT TESTED specific workflow | NOT TESTED | NOT TESTED |
| Research track / benchmark pages | NOT TESTED specific lifecycle | NOT TESTED | UNSUPPORTED history | NOT TESTED specific workflow | NOT TESTED | NOT TESTED |
| Training / fine-tuning / model installation | PARTIAL generic synthetic job protocol only; no tasks executed | PARTIAL synthetic owned job/result routes | UNSUPPORTED history | PARTIAL generic job archive only | PARTIAL generic archive | PARTIAL dead-PID detection; no task resumption |
| Job checkpoint attachment | PARTIAL controlled literal routing file, **not trained weights** | VERIFIED ownership/fixed path/missing file | UNSUPPORTED | UNSUPPORTED checkpoint format | PARTIAL generic job | NOT TESTED actual trained artifact |

Numerical result previews are not a history-card feature. Titles/types/dates/
status/detail links and 30-row pagination/search/workflow filters are tested.
Actual result values, model/tasks and warnings are checked in saved payloads,
downloads and detail/report HTTP responses. Browser clicks, chart interaction,
client-side number parsing/precision and Blob downloads remain NOT TESTED.

The source mapping is in [ARCHITECTURE.md](ARCHITECTURE.md). Machine-readable
case IDs, all downloaded artifacts and new-process responses are in the latest
run's `storage/` directory. See [REPORT.md](REPORT.md) for the current run link.
