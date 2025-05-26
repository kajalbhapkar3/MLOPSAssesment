# import pandas as pd
# import joblib
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# def retrain():
#     print("🔁 Starting model retraining...")

#     # Load preprocessed data
#     data = pd.read_csv("../data/processed/train.csv")
#     X = data.drop("msrp", axis=1)
#     y = data["msrp"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Retrain model
#     model = DecisionTreeRegressor(max_depth=5)
#     model.fit(X_train, y_train)

#     # Evaluate
#     preds = model.predict(X_test)
#     mse = mean_squared_error(y_test, preds)
#     print(f"✅ Retrained model MSE: {mse}")

#     # Save model
#     joblib.dump(model, "app/model.pkl")
#     print("💾 Retrained model saved.")

# if __name__ == "__main__":
#     retrain()


import pandas as pd
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def retrain():
    print("🔁 Starting model retraining...")

    # Load preprocessed data
    data = pd.read_csv("../data/processed/train.csv")
    
    # Separate target
    y = data["msrp"]
    X = data.drop("msrp", axis=1)

    # One-hot encode categorical columns
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Retrain model
    model = DecisionTreeRegressor(max_depth=5)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    print(f"✅ Retrained model MSE: {mse}")

    # Save model
    joblib.dump(model, "../../app/model.pkl")
    print("💾 Retrained model saved.")

if __name__ == "__main__":
    retrain()
