This is a draft readme for how to train/retrain the am2 model. It will be refined in a later branch/commit.

To check for newly available data for processing: python -m projects.am2_project.src.check_for_data from the root dir of the repo

The mlflow server needs to be started using:

$env:MLFLOW_FLASK_SERVER_SECRET_KEY='my-secret-key'
mlflow server --app-name basic-auth

If running as an IIS service on a server, you need to start the mlflow server with some extra
parameters, in order to specify the allowed host from which requests will come, in order to 
prevent Cross-Origin Resource Sharing (CORS) issues.

mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts 127.0.0.1:5000,localhost:5000,HOSTNAME:19015 --cors-allowed-origins http://HOSTNAME:19015