import mlflow
import json

def record_model(model, report, model_name="", input_example=None, notes=""):
    with open("./config/config.json") as f:
            general_config = json.load(f)
    mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
    with mlflow.start_run():
        # Log parameters and metrics using the MLflow APIs
        mlflow.log_param("model_class", model.__class__.__name__)
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
        mlflow.set_tag(
                            "training_data_url",
                            "https://s3.amazonaws.com/bucket/training-data.csv"
                        )
        mlflow.set_tag(
                            "mlflow.note.content", (notes + " https://s3.amazonaws.com/bucket/training-data.csv") 
                        )
        # Log the sklearn model and register it
        model_info = mlflow.sklearn.log_model(
                sk_model=model,
                name=model_name,
                input_example=input_example,
                registered_model_name=model_name,
                serialization_format="skops"
                )

