from flask import Flask, request, render_template_string
import mlflow
import pandas as pd
import yaml

app = Flask(__name__)

# Load configuration from the YAML file
with open("/Users/salwad/mlops-edureka-final-project/salary_prediction/src/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Initialize MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

# Load the best model
logged_model = 'runs:/6bc204c085684c808f56107b09173ab3/Linear Regression'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# HTML Template with CSS for enhanced UI
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Salary Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        h1 {
            color: #4CAF50;
            margin-bottom: 20px;
        }
        form {
            background: #fff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 400px;
            width: 100%;
        }
        label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
        }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        input[type="submit"] {
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
        }
        input[type="submit"]:hover {
            background: #45a049;
        }
        .result {
            margin-top: 20px;
            background: #e8f5e9;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #c8e6c9;
        }
    </style>
</head>
<body>
    <h1>Salary Prediction</h1>
    <form method="POST">
        <label for="experience">Years of Experience:</label>
        <input type="text" id="experience" name="experience" placeholder="Enter your experience in years">
        <input type="submit" value="Predict Salary">
    </form>
    {% if prediction %}
    <div class="result">
        <h2>Predicted Salary: ${{ prediction }}</h2>
    </div>
    {% endif %}
</body>
</html>
"""
@app.route("/")
def home():
    return "Welcome to the Salary Prediction API! Use the /predict endpoint to make predictions."

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        try:
            # Get user input
            experience = float(request.form["experience"])
            # Predict using the loaded model
            prediction = loaded_model.predict(pd.DataFrame([[experience]]))[0]
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template_string(HTML, prediction=prediction)

if __name__ == "__main__":
    app.run(host=config['flask']['host'], port=config['flask']['port'])