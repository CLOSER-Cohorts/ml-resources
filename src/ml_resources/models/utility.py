def create_model_package(model, 
    X,
    target_variable,
    accuracy_report={},
    preprocessing=[],
    framework="sklearn",
    notes="",
    model_version="NA",
    training_data_version="NA"):
    model_package = {
    "model": model,
    "metadata": {
        "model_type": str(model),
        "framework": framework,
        "input_features": list(X.columns),
        "feature_types": X.dtypes.astype(str).to_json(),
        "preprocessing": preprocessing,
        "target_variable": target_variable,
        "accuracy": accuracy_report,
        "training_data_version": training_data_version,
        "notes": notes,
        "model_version": model_version
        }
    }
    return model_package
