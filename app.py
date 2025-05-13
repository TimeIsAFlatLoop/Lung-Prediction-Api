from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load your pre-trained model (ensure model.pkl is in the same dir)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Categorical mapping
GENDER_MAP = {"M": 0, "F": 1}

# The 15 input feature names, in the same order your model expects
FEATURE_NAMES = [
    "GENDER", "AGE", "SMOKING", "YELLOW_FINGERS", "ANXIETY",
    "PEER_PRESSURE", "CHRONIC_DISEASE", "FATIGUE", "ALLERGY",
    "WHEEZING", "ALCOHOL_CONSUMING", "COUGHING",
    "SHORTNESS_OF_BREATH", "SWALLOWING_DIFFICULTY", "CHEST_PAIN"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify(error="No input data provided"), 400

    # Validate all features are present
    missing = [f for f in FEATURE_NAMES if f not in data]
    if missing:
        return jsonify(error=f"Missing features: {', '.join(missing)}"), 400

    # Build DataFrame
    df_input = pd.DataFrame([data], columns=FEATURE_NAMES)

    # Encode GENDER
    df_input["GENDER"] = df_input["GENDER"].map(GENDER_MAP)
    # Ensure numeric dtype
    df_input = df_input.astype(float)

    # Predict
    pred = model.predict(df_input)[0]
    proba = model.predict_proba(df_input)[0]

    return jsonify({
        "prediction": "YES" if pred == 1 else "NO",
        "probabilities": {"NO": proba[0], "YES": proba[1]}
    })

# Basic error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify(error="Server error"), 500

if __name__ == "__main__":
    # Render sets PORT env var automatically
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
