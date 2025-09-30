from preprocess import load_and_preprocess
from model import train_models
from predict import predict_price

if __name__ == "__main__":
    # Step 1: Load and preprocess
    X_scaled, y, scaler = load_and_preprocess("house_data.csv")

    # Step 2: Train models
    lr_model, dt_model, X_test, y_test = train_models(X_scaled, y)

    # Step 3: Example prediction
    # Example house features should match your dataset
    # e.g. {"area": 2000, "rooms": 3, "location_suburb": 1, "location_city": 0}
    sample_house = {
        "area": 2000,
        "rooms": 3,
        "location_suburb": 1,
        "location_city": 0
    }

    feature_names = list(sample_house.keys())
    pred_lr = predict_price(sample_house, lr_model, scaler, feature_names)
    pred_dt = predict_price(sample_house, dt_model, scaler, feature_names)

    print("\nSample House Prediction:")
    print("Linear Regression Prediction:", pred_lr)
    print("Decision Tree Prediction:", pred_dt)
