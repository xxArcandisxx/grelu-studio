# Browser verification setup plan — recorded before installation

No installed Chromium/Chrome/Firefox, Node, Playwright or Selenium was found.
Existing repository script: `scripts/setup_browser_qa.py`.

Planned project-local installation:

- Separate Python virtual environment: `.test-tools/venv` (not `.venv`).
- Playwright Python `1.65.2`, its driver and dependencies.
- Playwright-matched Chromium **headless shell**, an actual browser engine.
- Download/extract Ubuntu `libnspr4`, `libnss3`, `libasound2t64` packages to
  `.test-tools/libraries`; do not install them into the system.
- Browser artifacts under `.test-tools/browsers`; downloads under
  `.test-tools/runtime-downloads`.

Estimated network download **150–250 MB**, approximate until the package sizes
are available. Record actual downloads, versions and disk footprint afterward.
No sudo, apt install, system configuration change, application dependency change,
model download, training or weight update is planned. Launch at actual localhost
with isolated databases/downloads and the already cached pinned GM12878 model.

Command:

```bash
.venv/bin/python scripts/setup_browser_qa.py
```

If environment dependencies cannot be resolved safely, retain the logs and mark
real browser execution BLOCKED. Static HTML is not substitute browser evidence.
