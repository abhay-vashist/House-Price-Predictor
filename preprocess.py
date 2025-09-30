import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(dataset_path="house_data.csv"):
    # Load dataset
    df = pd.read_csv(dataset_path)
    df = df.dropna()  # remove missing values

    # Separate features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Encode categorical variables (like location)
    X = pd.get_dummies(X, drop_first=True)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
