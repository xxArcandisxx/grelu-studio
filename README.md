# gReLU Studio

A self-hosted browser interface for comparing the predicted signal of two aligned DNA windows with one nucleotide substitution.

## Run the review demo

From the checkout on Ubuntu WSL, use the existing environment:

    bash run_studio.sh

Open http://localhost:5000, select **Load example**, and run the comparison. Select an output track and download the JSON run record. The included CNN has random weights. Its output demonstrates the application flow. When the pinned [Genentech GM12878 DNase checkpoint](https://huggingface.co/Genentech/GM12878_dnase-model) is available in the local Hugging Face cache, the launcher also enables it. Select that model before loading an example to run a model-sized input. Synthetic examples verify the workflow. Use aligned hg38 inputs for research runs.

## Host on a lab server

Use the existing Python version and requirements lock and install Flask alongside the engine dependencies. On the lab host, serve the Flask WSGI app with a production server such as Gunicorn behind the lab's HTTPS and access-control proxy. For example, with Gunicorn installed:

    gunicorn --bind 127.0.0.1:8000 --workers 1 --timeout 180 studio_web:app

Set GRELU_HUMAN_CHECKPOINT to an administrator-vetted local checkpoint for a human regulatory model, GRELU_HUMAN_INPUT_LENGTH to its required input length, and GRELU_HUMAN_MODEL_ID to its published name or version. Restart the server whenever the configured checkpoint changes. The JSON export records the model label, checkpoint SHA-256, and engine library versions. The **Load example** button generates a synthetic input sized for the selected model. A checkpoint can execute code during deserialization, so configure only trusted files. The browser submits sequences and a model selection. Prediction outputs use a temporary directory that is removed after inference. Downloaded run records contain the submitted DNA sequences.

The plotted series are model predictions across output bins, optionally averaged into up to 100 displayed bins. The score difference is alternate minus reference, with scores averaged across all model output bins.

The browser includes single-substitution comparison, allele-specific regulation, variant prioritization, multi-allelic comparison, and curated saturation scanning. Each workflow has a dedicated page; saturation scanning requires the pinned local GM12878 checkpoint. Model Zoo entries are catalog selections and use the configured local backend, as explained beside the selector.

Use Sequence Prediction for one sequence or up to 12 FASTA records, Sequence QC for length and composition checks, and Model Inspector for model metadata. Each completed analysis includes a JSON download.

Research Utilities accepts tabular data for regression or binary evaluation, numerical track summaries and correlations, and basic BED coordinate validation. These pages analyze the provided files without loading a model. Results can be downloaded as JSON.


## Research workspace and local accounts

The responsive workspace includes navigation to every analysis tool, example inputs, result tables, and JSON exports. Run `bash run_studio.sh` and open http://localhost:5000.

Optional researcher accounts start at `/account/register`. Create an account, open the **Local development preview** activation link, confirm activation, and sign in. `/account/forgot` creates a replacement activation link for inactive accounts or a password reset link for active accounts. Links expire after one hour and are single-use. Password resets invalidate existing sign-ins. Analysis tools also remain accessible to guests.

Accounts are stored in the ignored `instance/accounts.sqlite3` database. Passwords are hashed; account forms use CSRF tokens. The ignored `instance/session.key` persists the local session signing key. Set `GRELU_SECRET_KEY` to supply your own stable key and `GRELU_HTTPS=1` when serving over HTTPS.

These account flows are a local development feature: there is no email provider integration. Recovery and activation links are displayed in the browser by default (`GRELU_ACCOUNT_LOCAL_PREVIEW=1`). Do not expose this preview mode on a public server. Setting `GRELU_ACCOUNT_LOCAL_PREVIEW=0` hides links; email delivery must be implemented before using account registration/recovery in a public deployment. Keep the lab's access-control proxy for deployed research data.

Run regression checks with `.venv/bin/python -m unittest discover -s tests -v`.
