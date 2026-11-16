from core.grelu_api import require_grelu


def list_zoo_models():
    """Repo ids in the gReLU model zoo (live from Hugging Face)."""
    require_grelu()
    import grelu.resources

    return list(grelu.resources.list_models())


def zoo_model_info(repo_id):
    """Model card, linked datasets and base models for one zoo model."""
    require_grelu()
    import grelu.resources as r

    info, errors = {"repo_id": repo_id}, {}
    for key, fn in (("model_card", r.get_model_info), ("datasets", r.get_datasets_by_model),
                    ("base_models", r.get_base_models)):
        try:
            info[key] = fn(repo_id)
        except Exception as e:  # noqa: BLE001
            info[key], errors[key] = None, str(e)
    if errors:
        info["errors"] = errors
    return info