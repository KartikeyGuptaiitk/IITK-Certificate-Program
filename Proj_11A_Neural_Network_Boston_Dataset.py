from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

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

ss = StandardScaler()
X = ss.fit_transform(X)
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state=2)

model=Sequential()
model.add(Dense(128,activation='relu',input_dim=11))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["MSE"])

history=model.fit(Xtrain,ytrain,epochs=150,batch_size=10)

ypred = model.predict(Xtest)
ypred = ypred[:,0]

mse = mean_squared_error(ytest, ypred)
r2 = r2_score(ytest, ypred)
print('Test MSE =', mse)
print('Test R2 score =', r2)

import matplotlib.pyplot as plt
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Training MSE')
plt.grid()
plt.show()
