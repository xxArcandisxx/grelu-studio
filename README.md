# GRELU Studio

> **Work in progress** — this project is actively being developed and is not ready for production use.

GRELU Studio is an experimental command-line toolkit for exploring sequence-model workflows with [gReLU](https://github.com/Genentech/gReLU). It is intended to provide a small, reproducible foundation for variant-effect prediction, sequence design, model inspection, and evidence-based experiment outputs.

## Current Status

The project is still in an early development stage. The workflow engine and core utilities are being built, tested, and refined. APIs, input formats, model-loading behavior, and output structures may change without notice.

Current workflow prototypes include:

- `info` — inspect the available device, PyTorch, gReLU, and model metadata
- `vep` — estimate variant effects from sequence or genomic-context inputs
- `design` — iteratively design sequences for selected target and off-target tasks

The included demo model uses random weights and is intended only for testing the pipeline, not for biological conclusions.

## Planned Development

- Improve model-loading support and validation
- Add clearer example parameter files and input datasets
- Expand workflow output summaries and evidence checks
- Add automated tests and continuous integration
- Improve documentation for gReLU models, variants, and sequence design
- Add safer, more convenient configuration for local and batch runs

## Tech Stack

- Python 3.14+
- NumPy
- PyTorch
- gReLU-compatible model adapters
- CSV and JSON input/output
- `uv` for project and dependency management

## Getting Started

### Prerequisites

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A compatible PyTorch and gReLU environment for real model workflows

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/xxArcandisxx/grelu-studio.git
   cd grelu-studio
   ```

2. Install the project environment:

   ```bash
   uv sync
   ```

3. Run the built-in self-test:

   ```bash
   uv run python engine.py --selftest
   ```

The self-test creates example inputs and writes demonstration outputs under `runs/selftest/`.

## Running a Workflow

The engine accepts a workflow name and a JSON parameter file:

```bash
uv run python engine.py --workflow info --params params.json
```

Every workflow returns one JSON object. Successful responses include `"ok": true`; failures include `"ok": false` and an error message.

Results can be written to a directory by adding `output_dir` to the parameter file:

```json
{
  "model": {
    "source": "demo",
    "seq_len": 200,
    "n_tasks": 3
  },
  "output_dir": "runs/example"
}
```

## Project Structure

```text
grelu-studio/
├── engine.py              # CLI entry point and workflow router
├── core/
│   ├── device.py          # Device detection
│   ├── models.py          # Model loading and prediction adapter
│   ├── sequences.py       # Sequence encoding and transformations
│   └── utils.py           # Output conversion and evidence helpers
├── workflows/
│   ├── info.py            # Environment and model inspection
│   ├── vep.py             # Variant-effect prediction
│   └── design.py          # Sequence design
├── src/grelu_studio/      # Installable package entry point
├── pyproject.toml         # Project metadata and dependencies
└── uv.lock                # Locked environment dependencies
```

## Important Notes

- This software is experimental and should not be used as the sole basis for biological or clinical decisions.
- Passing an evidence check does not prove that a model prediction or designed sequence is biologically correct.
- Model checkpoints may contain executable pickle data; load only checkpoints from sources you trust.
- Real workflows may require additional model, genome FASTA, and gReLU-specific dependencies.

## Contributing

Contributions, suggestions, and issue reports are welcome while the project is being developed. Please keep changes focused and include reproducible examples when reporting workflow or model-loading problems.

## License

No license has been specified for this project yet.
