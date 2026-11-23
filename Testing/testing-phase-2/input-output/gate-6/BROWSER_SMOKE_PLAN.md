# Follow-up browser smoke test — NOT EXECUTED

Gate 6 checks installed runtimes/frameworks in `environment.json`. It installs
nothing. Source/static HTML checks do not substitute for these browser steps.

1. Start the app with a new temporary HISTORY_DATABASE, ACCOUNT_DATABASE and
   LAB_ROOT, fixed test secret, pinned cached GM12878 checkpoint and CPU mode.
   Use one Gate 4 FASTA fixture and the independent expected raw-score manifest.
2. In an available real browser/framework, submit prediction and verify visible
   input count/IDs, model/task, warning, score rounding and history save link.
   Open history, filter by workflow/title, open the exact new record and reload.
3. Trigger the JSON and HTML report buttons; capture the actual download event,
   filename, content type and bytes. Independently parse JSON against the
   pre-submit manifest. Confirm HTML escaping and displayed values.
4. Submit the three-variant fixture; click CSV download and parse the **downloaded
   file**, checking defined headers/order, signed effects, blank optional fields,
   quote/newline/Unicode preservation and documented text escaping. Compare
   numerical values with the manifest, not rounded DOM text.
5. Repeat small attribution/design and motif/model-comparison cases using
   explicitly approved replay fixtures for storage-only browser testing. Check
   client CSV, JSON and design FASTA buttons separately. Label replay clearly;
   do not claim fresh inference from it.
6. Start a distinct browser context and verify result/job/file IDs cannot be
   opened across guest sessions. Seed isolated accounts if testing account
   behavior. Confirm failure messages on missing files and controlled DB faults.
7. Stop/restart app using the same isolated DB/secret, then reopen the record and
   repeat downloads. Check filters/pagination with 34 isolated history fixtures.
8. Save screenshots, console/network errors, browser/version, commands, manifests
   and exact bytes. Explicitly test JS integer precision if displaying integers
   above 2^53: Python JSON round-trip PASS does not guarantee browser precision.

All resulting classifications stay NOT TESTED until these steps actually run.
This follow-up is software verification, not Step 2 biological interpretation.
