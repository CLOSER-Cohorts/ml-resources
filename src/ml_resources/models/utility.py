def create_model_package(model, 
    training_data,
    target_variable,
    accuracy_report={},
    preprocessing=[],
    framework="sklearn",
    notes="",
    model_version="NA",
    training_data_version="NA",
    training_item_ids=[]):
    model_package = {
    "model": model,
    "metadata": {
        "model_type": str(model),
        "framework": framework,
        "input_features": [x for x in list(training_data.columns) if x != target_variable],
        "feature_types": training_data.dtypes.astype(str).to_json(),
        "preprocessing": preprocessing,
        "target_variable": target_variable,
        "accuracy": accuracy_report,
        "training_data_version": training_data_version,
        "training_item_ids": training_item_ids,
        "notes": notes,
        "model_version": model_version
        },
     "data": training_data

    }
    return model_package
