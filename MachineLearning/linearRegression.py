from sklearn import linear_model

train_X = [[1.0], [4.0]]
train_y = [5.5, 8.5]

reg = linear_model.LinearRegression()
reg.fit(train_X, train_y)

print(f"w = {reg.coef_}")
print(f"b = {reg.intercept_}")