import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def evaluate():
    df = pd.read_csv('../data/processed/train.csv')
    model = joblib.load('../../app/model.pkl')

    features = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    X = df[features]
    y = df['msrp']

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"✅ Test MSE: {mse}")

if __name__ == '__main__':
    evaluate()
