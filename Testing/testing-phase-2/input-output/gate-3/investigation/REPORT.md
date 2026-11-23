# Gate 3 — failure investigation and closure

**Recommendation: SAFE TO PROCEED WITH DOCUMENTED LIMITATIONS.**

The five failures are a reproducible external gReLU/NumPy/Torch interoperability
defect, not incorrect Gate 3 tensor dimensions or expected DNA. No current
website workflow reaches the affected inverse conversion. The original Gate 3
result remains **FAIL: 150 PASS, 5 FAIL, 8 SKIP, 0 BLOCKED**. No failure has been
changed into PASS/SKIP, and this recommendation does not authorize Gate 4 to run.

## 1. Direct reproduction and supported inputs

Both [minimal_repro.py](minimal_repro.py) and the separate automated optional
[verify_optional_inverse.py](verify_optional_inverse.py) were executed from
`/tmp` using the project's Python with `-I`. Neither imports Studio, loads a
model, runs inference, nor invokes the Studio preprocessing/oracle. Each tensor
is constructed directly from literal nucleotide channel vectors. The optional
suite also confirms exact equality to gReLU's own forward encoder.

Environment: **gReLU 1.1.0, Torch 2.15.0, NumPy 2.5.4, Python 3.14.4**, CPU
float32. See [environment](standalone-direct/environment.json),
[minimal stdout](minimal-repro.stdout.txt), [execution record](execution.json),
[test results](standalone-direct/results.json), and saved per-base tensors.

| Input | Shape | Expected inverse | Actual inverse | Optional test |
|---|---|---|---|---|
| A | (1,4,1) | `['A']` | `['N']` | FAIL |
| C | (1,4,1) | `['C']` | `['N']` | FAIL |
| G | (1,4,1) | `['G']` | `['N']` | FAIL |
| T | (1,4,1) | `['T']` | `['N']` | FAIL |
| N | (1,4,1) | `['N']` | IndexError: index 1 is out of bounds for axis 0 with size 1 | FAIL |
| ACGTN | (1,4,5) | `['ACGTN']` | `['ACGTN']` | PASS |
| GANTCTNNAG | (1,4,10) | `['GANTCTNNAG']` | `['GANTCTNNAG']` | PASS |

All seven inputs pass installed `check_one_hot`, are identified as `one_hot`,
and exactly match installed `strings_to_one_hot([dna])`.

The [official gReLU format documentation](https://genentech.github.io/gReLU/autoapi/grelu/sequence/format/index.html)
defines float32 tensors shaped `(4,length)` or `(N,4,length)` and gives no minimum
length for the inverse. Therefore length 1 meets the documented representation;
there is no documented exclusion that would invalidate these tests. No explicit
single-base example is claimed. The installed 1.1.0 module docstring gives the
same format contract and its validator accepts these cases. Installed source
is authoritative for this environment; the online docs are not version-pinned.
See [contract assessment](official-contract.json) and
[installed source with hashes/line numbers](standalone-direct/installed-source.json).

The run used isolated Python, no `core`, `workflows` or `studio_web` modules
were loaded, and cwd was `/tmp`. The editable installation still adds the
project's `src` directory through site configuration; its presence in `sys.path`
is recorded, not hidden. No Studio code is used by either reproduction.

An initial sandbox import stopped because importing gReLU also imports genomepy,
which needs its existing cache lock. That environmental blocker is recorded in
`execution.json`; the authorized cache-access rerun completed. It is not one of
the five inverse failures and is not an outstanding BLOCKED check.

## 2. Exact source-level cause

Installed `grelu/sequence/format.py:323–339`, `one_hot_to_indices`, first computes
a NumPy int8 argmax. Its N correction is:

```python
indices = one_hot.argmax(axis=-2).numpy().astype(np.int8)
indices[one_hot.max(axis=-2).values == 0] = 4
```

The indexing mask in the second line is a **Torch boolean tensor**, while
`indices` is a **NumPy array**. With a one-element mask in this environment,
NumPy indexing uses its scalar integer-index behavior: False becomes 0 and True
becomes 1. [Index-protocol evidence](index-protocol.json) explicitly records
`operator.index(mask)` as 0/1, whereas NumPy conversion retains boolean dtype.

For A/C/G/T, the all-zero test is False. Index 0 selects the existing row or
element and overwrites the correct base index with 4 (N). For N, the mask is
True, so index 1 is requested from an axis of size 1 and raises IndexError.
`one_hot_to_strings` delegates to this function, so it inherits both behaviors.
The error occurs before `indices_to_strings` can restore the DNA string.

The [dimension/mask diagnostics](standalone-direct/dimension-and-mask-diagnostics.json)
isolate this operation without patching gReLU:

- Both valid `(1,4,1)` and `(4,1)` A/N inputs show the defect. Removing the batch
  dimension does not fix it. Batched original expectations are correctly lists;
  diagnostic unbatched expectations are strings.
- Substituting a NumPy boolean mask **only in a standalone diagnostic copy of
  this indexing operation** preserves the A/C/G/T index and correctly assigns N.
  This establishes the mask interoperability cause; it is not a production fix
  or a replacement for the failing native compatibility test.
- A five-record single-base batch succeeds, as do the sampled length-2 and
  longer inputs. Thus the demonstrated trigger is a one-element mask, not an
  assertion that all single-base batches or all versions of gReLU fail.

The bug is in the installed library's mixed NumPy/Torch indexing under this
specific dependency combination. These results do not establish how another
NumPy/Torch/gReLU combination behaves. No installed package, weight or website
code was changed, and no dependency upgrade was attempted.

## 3. Complete website impact audit

[USAGE_AUDIT.md](USAGE_AUDIT.md) provides the complete feature matrix and manual
indirect-path review. [usage-audit.json](usage-audit.json) saves the 101-file
source inventory, file hashes, Python import/alias inventory, dynamic call
inventory, zero parse errors, zero direct inverse/dispatcher references, and
whole-library inverse/dispatcher callsites for comparison.
[Indirect-source review](indirect-source-review.json) also saves the relevant
installed callable bodies, line numbers and file hashes.

Prediction, every variant workflow, attribution, mutagenesis/saturation,
sequence design, motif analysis, model comparison, exports, training and
descriptive/metadata workflows were reviewed. None reaches the faulty inverse.
Native `predict_on_seqs` encodes strings forward and never decodes a tensor.
The website design implementation mutates strings itself; it does not call
native gReLU design routines. Exports retain original/designed DNA strings.

External gReLU design/dataset/interpretation capabilities can reach the affected
conversion through `convert_input_type`. They are not current website paths.
Any future adoption requires a fresh audit and compatibility validation.
Static absence does not certify unrelated unexecuted branches or output
correctness; previous PARTIAL and NOT TESTED coverage remains in force.

## 4. Original and updated evidence, explicitly distinguished

| Evidence view | PASS | FAIL | SKIP | BLOCKED | Meaning |
|---|---:|---:|---:|---:|---|
| Original combined Gate 3, preserved | 150 | 5 | 8 | 0 | Historical suite and report remain FAIL, byte-for-byte unchanged |
| Required website preprocessing subset | 148 | 0 | 8 | 0 | Explicit classification of the original saved checks; not a newly rerun tensor suite |
| Original optional inverse subset | 2 | 5 | 0 | 0 | Seven original inverse tests, retaining every status |
| Newly executed standalone optional dependency suite | 2 | 5 | 0 | 0 | Independently reproduces all seven original inverse statuses outside Studio |

The two passing longer inverse tests belong to the optional subset too; they
were not retained in the required subset to inflate its pass count. No checks
were deleted and no fixture expected result was changed. Additional dimension
experiments are diagnostic evidence, not additional pass-count inflation.

[results.json](results.json) includes the original result hash, both complete
classified original check lists, the fresh optional result hash and explicit
Gate 4 status. The separate compatibility runner exits **1** for the real
failures. The original combined Gate 3 runner remains unchanged and also exits
nonzero while those failures remain. Closure classification is an impact
decision; it does not claim that the external defect is repaired.

## 5. Preservation and reproducible commands

[preservation.json](preservation.json) compares 7,407 protected files and the
101-file source inventory. No protected file or source file changed. Protected
content includes Phase 1 archives/original results, Gate 1 runs, Gate 2
tests/fixtures/runs, all earlier Gate 3 runs, the original Gate 3 runner/latest
pointer, the installed format module and the pinned checkpoint. Full before
and after SHA-256 maps are retained. Gates 1 and 2 retain their prior recorded
39 PASS/0 FAIL/2 SKIP and 265 PASS/0 FAIL/12 SKIP/0 BLOCKED respectively; they
were not rerun during this read-only production investigation.

From `/tmp`, the exact optional reproduction command is:

```bash
/home/akshadha/Desktop/GRELU-Studio/.venv/bin/python -I /home/akshadha/Desktop/GRELU-Studio/Testing/testing-phase-2/input-output/gate-3/investigation/verify_optional_inverse.py --output /tmp/grelu-inverse-fresh-run
```

Use a new output directory for every run; the runner refuses to overwrite an
existing directory. Exit 1 is expected for the five saved compatibility failures.
The minimal print-only reproduction (exit 0 despite printing faulty behavior):

```bash
/home/akshadha/Desktop/GRELU-Studio/.venv/bin/python -I /home/akshadha/Desktop/GRELU-Studio/Testing/testing-phase-2/input-output/gate-3/investigation/minimal_repro.py
```

From repository root, repeat the audit and classified-results preservation check:

```bash
python3 Testing/testing-phase-2/input-output/gate-3/investigation/audit_usage.py
python3 Testing/testing-phase-2/input-output/gate-3/investigation/summarize_closure.py
```

The audit and closure summarizer have no model/library imports. The standalone
test imports only installed dependencies and standard library modules. Tests
load no checkpoints, run no numerical model forward, optimizer, training,
biological benchmark, inference-equivalence check or Gate 4 code.

## 6. Final recommendation

**SAFE TO PROCEED WITH DOCUMENTED LIMITATIONS**, because the required tested
website preprocessing checks have zero failures and the reproduced optional
dependency defect has no reachable production use in the audited source.
Keep the optional failures visible as a known external limitation. Carry forward
all eight original skips and workflow PARTIAL/NOT TESTED restrictions; this
investigation does not close them or establish biological accuracy.

**Stopped after the investigation. Gate 4 has not started and awaits review.**
