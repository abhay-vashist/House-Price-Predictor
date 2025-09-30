from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} Performance:")
    print("Mean Squared Error:", mse)
    print("R² Score:", r2)

def plot_predictions(y_test, y_pred_lr, y_pred_dt):
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred_lr, label="Linear Regression", alpha=0.6)
    plt.scatter(y_test, y_pred_dt, label="Decision Tree", alpha=0.6, marker="x")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.title("Actual vs Predicted House Prices")
    plt.show()
