import mlflow
import os

import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

os.environ["GIT_PYTHON_REFRESH"] = "quiet"
mlflow.set_tracking_uri("http://localhost:5000")
# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Start MLflow run
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=3)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 3)
    mlflow.log_metric("accuracy", acc)

    # Log the model
    mlflow.sklearn.log_model(model, "model")
