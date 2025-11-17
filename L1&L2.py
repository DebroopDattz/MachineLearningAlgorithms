import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_classification(
    n_samples=500, n_features=10, n_informative=2, n_redundant=6,
    n_clusters_per_class=1, class_sep=0.8, flip_y=0.1, random_state=42
)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

def evaluate_model(penalty, C, title):
    model = LogisticRegression(penalty=penalty, C=C, solver='liblinear', max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n=== {title} ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))

    # Decision boundary (first two features only)
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))

    grid = np.c_[xx.ravel(), yy.ravel()]
    full_grid = np.zeros((grid.shape[0], X_train.shape[1]))
    full_grid[:, :2] = grid
    Z = model.predict(full_grid).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolor='k')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Evaluate L2 Regularization (Ridge)
evaluate_model('l2', 0.01, 'L2 Regularization (C=0.01) - Underfitting')
evaluate_model('l2', 1, 'L2 Regularization (C=1) - Balanced')
evaluate_model('l2', 100, 'L2 Regularization (C=100) - Overfitting')

# Evaluate L1 Regularization (Lasso)
evaluate_model('l1', 1, 'L1 Regularization (C=1) - Sparse Model')