import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Dataset
data = pd.DataFrame({
    'Marks': [25, 35, 45, 50, 55, 60, 65, 70, 80, 85, 90, 95],
    'Result': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
})

# Features and labels
X = data[['Marks']]
y = data['Result']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Test case
test_marks = [[58]]
prob = model.predict_proba(test_marks)[0][1]
print(f"Test Marks: 58")
print(f"Predicted Class: {'Pass' if prob >= 0.5 else 'Fail'}")
print(f"Predicted Probability of Passing: {prob:.2f}")

# Evaluation
print(f"\nAccuracy: {accuracy_score(y, y_pred) * 100:.2f}%")
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))

# Logistic curve
x_vals = np.linspace(20, 100, 200).reshape(-1, 1)
y_probs = model.predict_proba(x_vals)[:, 1]
plt.scatter(X, y, color='blue')
plt.plot(x_vals, y_probs, color='red')
plt.xlabel('Marks')
plt.ylabel('Pass Probability')
plt.show()
