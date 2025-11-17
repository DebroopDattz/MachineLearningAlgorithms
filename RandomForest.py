import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select features and target
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
df = df[features + ['Survived']]
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical features
for col in ['Sex', 'Embarked']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split data
X, y = df[features], df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

print("\n--- Decision Tree ---")
print(f"Accuracy: {accuracy_score(y_test, dt_pred):.2f}")
print(f"Depth: {dt.get_depth()} | Leaves: {dt.get_n_leaves()}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print("\n--- Random Forest ---")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(f"Average Tree Depth: {sum(t.get_depth() for t in rf.estimators_) / len(rf.estimators_):.2f}")

# Cross-validation
cv_dt = cross_val_score(dt, X, y, cv=5)
cv_rf = cross_val_score(rf, X, y, cv=5)
print("\n--- Cross-Validation (5-fold) ---")
print(f"Decision Tree: {cv_dt.mean():.2f} ± {cv_dt.std():.2f}")
print(f"Random Forest: {cv_rf.mean():.2f} ± {cv_rf.std():.2f}")

# Feature importance (Random Forest)
importances = pd.Series(rf.feature_importances_, index=features)
importances.sort_values().plot.barh(title="Feature Importance (Random Forest)")
plt.show()

# Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, model, pred, title in zip(
    axes, [dt, rf], [dt_pred, rf_pred], ['Decision Tree', 'Random Forest']
):
    cm = confusion_matrix(y_test, pred)
    ax.imshow(cm, cmap='Blues' if title == 'Decision Tree' else 'Greens')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()