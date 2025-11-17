import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select and clean features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
df = df[features + ['Survived']]
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical features
for col in ['Sex', 'Embarked']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Train/test split
X, y = df[features], df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ensemble models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Collect feature importances
importance_df = []
for name, model in models.items():
    model.fit(X_train, y_train)
    importance_df.append(pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_,
        'Model': name
    }))

importance_df = pd.concat(importance_df)

# Plot comparison
plt.figure(figsize=(9, 5))
sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Model')
plt.title('Feature Importance Comparison - Ensemble Models')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()