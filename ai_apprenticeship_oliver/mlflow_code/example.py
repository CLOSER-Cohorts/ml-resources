from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5001")

# Let's simulate different versions of the model
# These could be trained in different scripts or notebooks at different times
versions = [
    {"model_class": LinearRegression, "params": {}},
    {"model_class": DecisionTreeRegressor, "params": {"max_depth": 5}},
    {
        "model_class": GradientBoostingRegressor,
        "params": {"n_estimators": 100, "learning_rate": 0.05, "random_state": 42},
    },
    {
        "model_class": RandomForestRegressor,
        "params": {"n_estimators": 110, "random_state": 42},
    },
]

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train, evaluate, and save each version
for version in versions:
    with mlflow.start_run():
        model_class = version["model_class"]
        params = version["params"]
        model = model_class(**params)
        model.fit(X_train, y_train)
        # Log parameters and metrics using the MLflow APIs
        mlflow.log_param("model_class", model_class.__name__)
        mlflow.log_params(params)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        # Log the sklearn model and register it
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sklearn-model",
            input_example=X[:5],
            registered_model_name="california_housing_model",
        )
        print(
            f"{model_class.__name__} (version {model_info.registered_model_version}): MSE = {mse:.4f}"
        )
