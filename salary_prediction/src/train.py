import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import whylogs as why
from datetime import datetime, timezone


# Load configuration from the YAML file
with open("/Users/salwad/mlops-edureka-final-project/salary_prediction/src/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set WhyLabs environment variables
os.environ["WHYLABS_API_KEY"] = config['whylabs']['api_key']
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-1"

# Load dataset
data_path = config['data']['path']
data = pd.read_csv(data_path)

# Clean data
data.dropna(axis=0, inplace=True)

X = data[['YearsExperience']]
y = data['Salary']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=config['split']['test_size'], random_state=config['split']['random_state']
)

#Whylabs generate a profile from a Pandas dataframe in Python
results = why.log(data)
os.environ["WHYLABS_API_KEY"] = config['whylabs']['api_key']
os.environ["WHYLABS_DEFAULT_DATASET_ID"] = "model-1" # The selected model project "MODEL-NAME" is "model-0"
results = why.log(data)
results.writer("whylabs").write()

# Initialize MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

# Dictionary of models to train
models = {
    "Linear Regression": LinearRegression(**config['models']['linear_regression']),
    "Random Forest Regressor": RandomForestRegressor(**config['models']['random_forest']),
    "Decision Tree Regressor": DecisionTreeRegressor(**config['models']['decision_tree']),
}

# Track best model
best_model = None
best_r2 = -np.inf
best_model_uri = None

# Train and log models
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log model performance metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
        mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
        mlflow.sklearn.log_model(model, model_name)

        # Update best model tracking
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = model_name
            best_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"

        # Log data and regression metrics to WhyLabs
        feature_data = X_test.copy()
        feature_data["predicted_salary"] = y_pred
        feature_data["targets_output"] = y_test.values

        print(f"Logging data to WhyLogs for {model_name}...")
        results = why.log_regression_metrics(
            feature_data,
            target_column="targets_output",
            prediction_column="predicted_salary"
        )

        # Set dataset timestamp and write profile to WhyLabs
        profile = results.profile()
        profile.set_dataset_timestamp(datetime.now(timezone.utc))  # Timezone-aware datetime
        results.writer("whylabs").write()

        print(f"Finished logging to WhyLabs for {model_name}.")


# result = why.log(pandas=df_target)
# prof_view = result.view()

# result_ref = why.log(pandas=df_reference)
# prof_view_ref = result_ref.view()

# visualization = NotebookProfileVisualizer()
# visualization.set_profiles(target_profile_view=prof_view, reference_profile_view=prof_view_ref)

# visualization.summary_drift_report()

# Best model summary
print(f"Best model: {best_model_name} with R²: {best_r2}")

# Register the best model
if best_model_uri:
    registered_model_name = "BestSalaryPredictionModel"
    mlflow.register_model(model_uri=best_model_uri, name=registered_model_name)
    print(f"Best model registered in MLflow under the name '{registered_model_name}'")