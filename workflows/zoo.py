from core.zoo import list_zoo_models, zoo_model_info


def handle_zoo_workflow(params: dict) -> dict:
    """params: action 'list' | 'info' ; repo_id (for 'info')"""
    action = params.get("action", "list")
    note = ["The model list comes live from the gReLU Hugging Face collection, so it needs internet."]
    if action == "list":
        ids = list_zoo_models()
        return {"device": None, "results": {"models": ids, "n_models": len(ids)}, "evidence": [], "warnings": note}
    if action == "info":
        if not params.get("repo_id"):
            raise ValueError("'repo_id' is required for action 'info'.")
        return {"device": None, "results": zoo_model_info(params["repo_id"]), "evidence": [], "warnings": note}
    raise ValueError("action must be 'list' or 'info'")