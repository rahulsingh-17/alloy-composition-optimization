from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

models = joblib.load("composition_models.pkl")

# required propersties on which model is going to train
# order must match exactly what was used during training
PROPERTY_ORDER = [
    "Hardness (HV)",
    "Electrical conductivity (%IACS)",
    "Ultimate tensile strength (MPa)",
    "Yield strength (MPa)"
]

# typical average values from the dataset used as fallback
# when user does not provide a property
FALLBACK_VALUES = {
    "Hardness (HV)":                    215.0,
    "Electrical conductivity (%IACS)":   39.0,
    "Ultimate tensile strength (MPa)":  691.0,
    "Yield strength (MPa)":             569.0
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    print("Received data:", data)

    # read values from request, use fallback if not provided
    hardness         = float(data["hardness"])         if "hardness"         in data else FALLBACK_VALUES["Hardness (HV)"]
    conductivity     = float(data["conductivity"])     if "conductivity"     in data else FALLBACK_VALUES["Electrical conductivity (%IACS)"]
    tensile_strength = float(data["tensile_strength"]) if "tensile_strength" in data else FALLBACK_VALUES["Ultimate tensile strength (MPa)"]
    yield_strength   = float(data["yield_strength"])   if "yield_strength"   in data else FALLBACK_VALUES["Yield strength (MPa)"]

    user_input = np.array([[hardness, conductivity, tensile_strength, yield_strength]])

    composition = {}
    for element, model in models.items():
        predicted_wt = model.predict(user_input)[0]
        predicted_wt = max(0, round(float(predicted_wt), 4))
        composition[element] = predicted_wt

    # sort highest to lowest
    composition = dict(sorted(composition.items(), key=lambda x: x[1], reverse=True))

    # remove near-zero elements
    composition = {k: v for k, v in composition.items() if v > 0.001}

    print("Predicted composition:", composition)
    return jsonify(composition)

if __name__ == "__main__":
    app.run(debug=True)
