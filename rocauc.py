import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y, target_names = iris.data, iris.target, iris.target_names

# === Binary Logistic Regression (Setosa vs Versicolor) ===
mask = y < 2
X_bin, y_bin = X[mask], y[mask]

model_bin = LogisticRegression()
model_bin.fit(X_bin, y_bin)

y_pred_bin = model_bin.predict(X_bin)
y_prob_bin = model_bin.predict_proba(X_bin)[:, 1]

print("=== Binary Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y_bin, y_pred_bin))
print(classification_report(y_bin, y_pred_bin, target_names=target_names[:2]))
print(f"ROC-AUC Score: {roc_auc_score(y_bin, y_prob_bin):.2f}")

# Plot Binary ROC Curve
fpr, tpr, _ = roc_curve(y_bin, y_prob_bin)
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.show()

# === Multiclass Logistic Regression ===
model_multi = LogisticRegression(multi_class='ovr', solver='liblinear')
model_multi.fit(X, y)

y_pred_multi = model_multi.predict(X)
y_prob_multi = model_multi.predict_proba(X)
y_bin_multi = label_binarize(y, classes=[0, 1, 2])

print("\n=== Multiclass Logistic Regression ===")
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_multi))
print(classification_report(y, y_pred_multi, target_names=target_names))
print(f"Macro ROC-AUC Score: {roc_auc_score(y_bin_multi, y_prob_multi, average='macro'):.2f}")

# Plot Multiclass ROC Curves
for i in range(3):
    fpr, tpr, _ = roc_curve(y_bin_multi[:, i], y_prob_multi[:, i])
    plt.plot(fpr, tpr, label=f"{target_names[i]} (AUC={auc(fpr, tpr):.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.show()