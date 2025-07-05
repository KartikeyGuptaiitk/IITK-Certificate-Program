from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

california = datasets.fetch_california_housing();
X = california.data;
y = california.target; 

#The data pertains to the houses found in a given California district 
#and some summary stats about them based on the 1990 census data.
#Total number of points = 20640 
#The 8 features are

#MedInc
#HouseAge
#AveRooms
#AveBedrms
#Population
#AveOccup
#Latitude
#Longitude

# The target is
#MedHouseVal

ss = StandardScaler()
X = ss.fit_transform(X)
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size = 0.2,random_state=5)









history = model.fit(Xtrain, ytrain, epochs=150, batch_size=100)
ypred = model.predict(Xtest)
ypred = ypred[:,0]

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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
