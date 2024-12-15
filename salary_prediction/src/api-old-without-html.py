from flask import Flask, request, jsonify
import mlflow
import pandas as pd
import mlflow.sklearn
import yaml

app = Flask(__name__)

# Load configuration from the YAML file
with open("/Users/salwad/mlops-edureka-final-project/salary_prediction/src/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

# Load best model
logged_model = 'runs:/6bc204c085684c808f56107b09173ab3/Linear Regression'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    experience = data['YearsExperience']
    # Predict on a Pandas DataFrame.
    prediction = loaded_model.predict(pd.DataFrame([[experience]]))
    return jsonify({"YearsExperience": experience, "PredictedSalary": prediction[0]})

if __name__ == "__main__":
    app.run(host=config['flask']['host'], port=config['flask']['port'])