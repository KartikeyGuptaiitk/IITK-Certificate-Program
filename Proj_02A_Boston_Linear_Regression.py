from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd

BosData = pd.read_csv('BostonHousing.csv')
X = BosData.iloc[:,0:11]
y = BosData.iloc[:, 13] # MEDV: Median value of owner-occupied homes in $1000s

# Boston Housing Dataset is a derived from information collected by 
# the US Census Service concerning housing in the area of Boston MA.

# The 11 regressors/ features are
# CRIM: Per capita crime rate by town
# ZN: Proportion of residential land zoned for lots over 25,000 sq. ft
# INDUS: Proportion of non-retail business acres per town
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# NOX: Nitric oxide concentration (parts per 10 million)
# RM: Average number of rooms per dwelling
# AGE: Proportion of owner-occupied units built prior to 1940
# DIS: Weighted distances to five Boston employment centers
# RAD: Index of accessibility to radial highways
# TAX: Full-value property tax rate per $10,000
# PTRATIO: Pupil-teacher ratio by town

# The response/ target variable is
# MEDV: Median value of owner-occupied homes in $1000s
Xtrain, Xtest, ytrain, ytest =\
 train_test_split(X, y, test_size=0.2, random_state=5)
reg= LinearRegression()
reg.fit(Xtrain, ytrain)
 
ytrainpredict=reg.predict(Xtrain) 
mse= mean_squared_error(ytrain,ytrainpredict);
r2=r2_score(ytrain,ytrainpredict)
     
     





print('Train MSE =', mse)
print('Train R2 score =', r2)
print("\n")

ytestpredict = reg.predict(Xtest)
mse = mean_squared_error(ytest, ytestpredict)
r2 = r2_score(ytest, ytestpredict)
print('Test MSE =', mse)
print('Test R2 score =', r2)


import matplotlib.pyplot as plt

plt.figure()
plt.scatter(ytest, ytestpredict, color='blue', alpha=0.6)
plt.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.grid()
plt.show()