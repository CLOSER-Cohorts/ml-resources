Run the following command to start the MLFlow server, before executing code such as example.py

python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001