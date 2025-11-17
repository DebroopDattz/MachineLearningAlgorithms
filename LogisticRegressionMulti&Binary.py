import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import matplotlib.pyplot as plt

iris = load_iris()
X,y = iris.data, iris.target

multimodel = LogisticRegression(multi_class='multinomial',solver='lbfgs',max_iter=200)
multimodel.fit(X,y)
y_pred_multi = multimodel.predict(X)

print(f"Accuracy: {accuracy_score(y,y_pred_multi)}")
print(classification_report(y,y_pred_multi,target_names = iris.target_names))
print("Confusion Matrix:\n",confusion_matrix(y,y_pred_multi))

mask = y<2
X_bin, y_bin = X[mask],y[mask]
binmodel = LogisticRegression()
binmodel.fit(X_bin,y_bin)
y_pred_bin=binmodel.predict(X_bin)

print(f"Accuracy for Binary:{accuracy_score(y_bin,y_pred_bin)}")
print(classification_report(y_bin,y_pred_bin))
print(f"Confusion Matrix for Binary:\n {confusion_matrix(y_bin,y_pred_bin)}")

