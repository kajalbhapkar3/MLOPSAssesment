# app/main.py

from flask import Flask, request, jsonify
from src.prediction.predict import load_model, make_prediction

app = Flask(__name__)
model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    prediction = make_prediction(model, data)
    return jsonify({"prediction": prediction})

@app.route("/health", methods=["GET"])
def health():
    return "API is live!"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
