import numpy as np

def predict_price(sample, model, scaler, feature_names):
    # Convert input dict to array
    sample_df = np.array([sample[f] for f in feature_names]).reshape(1, -1)
    sample_scaled = scaler.transform(sample_df)
    prediction = model.predict(sample_scaled)[0]
    return prediction
