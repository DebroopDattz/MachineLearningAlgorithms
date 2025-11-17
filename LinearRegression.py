import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'Hours':[1,2,3,4,5,6,7,8,9,10],
    'Scores':[10,20,25,40,45,50,60,65,70,80]
})
X = data[['Hours']]
y = data['Scores']

model=LinearRegression()
model.fit(X,y)

y_pred = model.predict(X)
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")
print(f"Mean Squared Error: {mean_squared_error(y, y_pred)}")
print(f"R^2 Score: {r2_score(y, y_pred)}")

test_hours = [[6.5]]
print(f"Predicted score for {test_hours[0][0]} is {model.predict(test_hours)[0]}")

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.title('Hours vs Scores Linear Regression')
plt.show()