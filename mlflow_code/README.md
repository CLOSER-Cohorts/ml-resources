Run the following command to start the MLFlow server, before executing code such as example.py

python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001

Run the following to run on staging (note there are CORS issues). Make sure you have
run .venv/scripts/activate first

C:\Users\qtnzoly\Development\ai_apprenticeship\issue-363_NEW2\ml-resources
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts 127.0.0.1:5000,localhost:5000,STAGING_SERVER_NAME:19015 --cors-allowed-origins STAGING_SERVER_NAME:19015  --backend-store-uri sqlite:///mlflow.db --artifacts-destination file:./mlruns