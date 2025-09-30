from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from utils import evaluate_model, plot_predictions

def train_models(X, y):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    evaluate_model("Linear Regression", y_test, y_pred_lr)

    # Decision Tree
    dt_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    evaluate_model("Decision Tree", y_test, y_pred_dt)

    # Plot comparison
    plot_predictions(y_test, y_pred_lr, y_pred_dt)

    return lr_model, dt_model, X_test, y_test
