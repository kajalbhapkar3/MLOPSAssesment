import pandas as pd
import joblib

def load_model(model_path="app/model.pkl"):
    model = joblib.load(model_path)
    return model

def make_prediction(model, input_data):
    """
    Takes a model and input dictionary,
    Returns a prediction.
    """
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return int(prediction[0])
