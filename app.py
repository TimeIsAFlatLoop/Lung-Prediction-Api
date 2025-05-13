from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Lung Cancer Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    input_data = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(input_data)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)
