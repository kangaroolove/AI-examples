from sklearn import linear_model
import numpy as np

X = np.array([[1.0, 2.0], [4.0, 3.0]])
y = np.array([5.5, 8.5])

reg = linear_model.LinearRegression()
reg.fit(X, y)

print(f"w = {reg.coef_}")
print(f"b = {reg.intercept_}")
print(f"score = {reg.score(X, y)}")

target_y = reg.predict(np.array([[6.0, 5.0]]))
print(f"target_y = {target_y}")