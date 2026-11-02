import mlflow
import mlflow.sklearn
import json
import numpy as np
from cryptography.hazmat.primitives import serialization
from sklearn.linear_model import LogisticRegression
import base64
from src.cryptography import hash_directory

with open("./config/config.json") as f:
    general_config = json.load(f)

def register_model_and_metrics(model, 
    model_class,
    model_name,
    report,
    notes,
    input_example,
    data_filename,
    prediction_results=None,
    roc_auc_score=None
    ):
    mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
    with mlflow.start_run():
        # Log parameters and metrics using the MLflow APIs
        mlflow.log_param("model_class", model_class)
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
        if prediction_results is not None:
            mlflow.log_metric(f"top {prediction_results['N']} accuracy", prediction_results['TopNAccuracy'])
            mlflow.log_metric(f"L1 accuracy", prediction_results['L1Correct'])
            mlflow.log_metric(f"Top {prediction_results['N']} accuracy L1", prediction_results['TopNAccuracyL1'])
        #if roc_auc_score is not None:
        #    mlflow.log_metric(f"ROC AUC score", roc_auc_score)
        mlflow.set_tag(
            "training_data_url",
            data_filename
            )
        mlflow.set_tag(
            "mlflow.note.content", notes
            )
        # Log the sklearn model and register it
        # Just log a small model for now, to avoid uploading large files.
        # I just want to save metadata
        lr=LogisticRegression(max_iter=1000)
        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        input_example=input_example,
        #registered_model_name=model_name,
        serialization_format="skops",
        skops_trusted_types=['xgboost.core.Booster', 'xgboost.sklearn.XGBClassifier']
        )
        model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_info.model_uri
        )
        print(model_path)
        with open("./keys/ed25519_private_key.pem", "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None
            )
        model_hash = hash_directory(model_path)
        print(model_hash)
        signature = private_key.sign(
            bytes.fromhex(model_hash)
        )
        signature_b64 = base64.b64encode(signature).decode("ascii")
        registered_model = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=model_name,
        )
        model_version = registered_model.version
        client = mlflow.MlflowClient()
        # Get the newly created version
        latest = client.get_latest_versions(
            model_name,
            stages=None
            )[-1]
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="signature_algorithm",
            value="Ed25519",
        )
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="hash_algorithm",
            value="SHA-256",
        )
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="model_sha256",
            value=model_hash,
        )
        client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key="ed25519_signature",
            value=signature_b64,
        )
        client.set_model_version_tag(
            name=model_name,
            version=latest.version,
            key="notes",
            value=notes
        )



def register_model_and_cross_validation_metrics(model, 
    model_class,
    model_name,
    cross_validation_results,
    notes,
    input_example,
    data_filename):
    mlflow.set_tracking_uri(f"{general_config["MLFlowServerHost"]}:{general_config["MLFlowServerPort"]}")
    with mlflow.start_run():
        # Log parameters and metrics using the MLflow APIs
        mlflow.log_param("model_class", model_class.__name__)
        mlflow.log_params(model.get_params())
        mlflow.log_metric("Number of folds in cross validation", 
            len(cross_validation_results['test_accuracy']))         
        mlflow.log_metric("Mean cross validation accuracy", 
            np.mean(cross_validation_results['test_accuracy']))
        mlflow.log_metric("Mean cross validation macro precision", 
            np.mean(cross_validation_results["test_precision_macro"]))
        mlflow.log_metric("Mean cross validation macro recall", 
            np.mean(cross_validation_results["test_recall_macro"]))
        mlflow.log_metric("Mean cross validation macro f1-score", 
            np.mean(cross_validation_results["test_f1_macro"]))
        mlflow.log_metric("Mean cross validation weighted precision", 
            np.mean(cross_validation_results["test_precision_weighted"]))
        mlflow.log_metric("Mean cross validation macro recall", 
            np.mean(cross_validation_results["test_recall_weighted"]))
        mlflow.log_metric("Mean cross validation macro f1-score", 
            np.mean(cross_validation_results["test_f1_weighted"]))
        if "test_top_5_accuracy" in cross_validation_results.keys():
            mlflow.log_metric(f"Mean top 5 accuracy", 
                np.mean(cross_validation_results["test_top_5_accuracy"]))
            mlflow.set_tag(
                "training_data_url",
                data_filename
                )
        mlflow.set_tag(
            "mlflow.note.content", notes
            )
        # Log the sklearn model and register it
        model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name=model_name,
        input_example=input_example,
        registered_model_name=model_name,
        serialization_format="skops"
        )
