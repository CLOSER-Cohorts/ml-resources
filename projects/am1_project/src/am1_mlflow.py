import mlflow
import mlflow.sklearn
import json

with open("./config/config.json") as f:
    general_config = json.load(f)

def register_model_and_metrics(model, 
    model_class,
    model_name,
    report,
    notes,
    input_example,
    training_data_file,
    prediction_results):
    mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
    with mlflow.start_run():
        # Log parameters and metrics using the MLflow APIs
        mlflow.log_param("model_class", model_class.__name__)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", report['accuracy'])
        mlflow.log_metric("macro average precision", report['macro avg']['precision'])
        mlflow.log_metric("macro average recall", report['macro avg']['recall'])
        mlflow.log_metric("macro average f1-score", report['macro avg']['f1-score'])
        mlflow.log_metric("macro average support", report['macro avg']['support'])
        mlflow.log_metric("weighted average precision", report['weighted avg']['precision'])
        mlflow.log_metric("weighted average recall", report['weighted avg']['recall'])
        mlflow.log_metric("weighted average f1-score", report['weighted avg']['f1-score'])
        mlflow.log_metric("weighted average support", report['weighted avg']['support'])
        mlflow.log_metric(f"top {prediction_results['N']} accuracy", prediction_results['TopNAccuracy'])
        
        mlflow.set_tag(
            "training_data_url",
            training_data_file
            )
        mlflow.set_tag(
            "mlflow.note.content", training_data_file 
            )
        mlflow.log_artifact("training_item_ids.parquet")
        # Log the sklearn model and register it
        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        input_example=input_example,
        registered_model_name=model_name,
        serialization_format="skops"
        )
