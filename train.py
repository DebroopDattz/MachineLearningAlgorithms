import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DEFAULT_DATA = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Scores": [10, 20, 25, 40, 45, 50, 60, 65, 70, 80],
})


def load_data(path: Path | None = None) -> pd.DataFrame:
    if path and path.exists():
        df = pd.read_csv(path)
    else:
        df = DEFAULT_DATA.copy()
    if "Hours" not in df.columns or "Scores" not in df.columns:
        raise ValueError("Data must contain 'Hours' and 'Scores' columns.")
    return df[["Hours", "Scores"]]


def train_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model: LinearRegression, X: np.ndarray, y: np.ndarray) -> dict:
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, preds)
    return {"mse": mse, "rmse": rmse, "r2": r2, "predictions": preds}


def plot_results(X_train, y_train, X_test, y_test, model, out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, y_train, color="blue", label="Train")
    plt.scatter(X_test, y_test, color="green", label="Test")
    x_line = np.linspace(min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max()), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="red", label="Regression line")
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.title("Linear Regression: Hours vs Scores")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train a Linear Regression model for Hours vs Scores.")
    parser.add_argument("--data", type=Path, help="Path to CSV file with 'Hours' and 'Scores' columns.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for split.")
    parser.add_argument("--model-out", type=Path, default=Path("models/linear_regression.joblib"), help="Path to save trained model.")
    parser.add_argument("--plot-out", type=Path, default=Path("models/regression_plot.png"), help="Path to save plot.")
    args = parser.parse_args()

    df = load_data(args.data)
    X = df[["Hours"]].values
    y = df["Scores"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)

    model = train_model(X_train, y_train)
    train_metrics = evaluate_model(model, X_train, y_train)
    test_metrics = evaluate_model(model, X_test, y_test)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model_out)

    plot_results(X_train, y_train, X_test, y_test, model, args.plot_out)

    print(f"Model saved to: {args.model_out}")
    print(f"Plot saved to: {args.plot_out}")
    print("Train metrics:", {k: float(v) for k, v in train_metrics.items() if k != "predictions"})
    print("Test metrics:", {k: float(v) for k, v in test_metrics.items() if k != "predictions"})


if __name__ == "__main__":
    main()