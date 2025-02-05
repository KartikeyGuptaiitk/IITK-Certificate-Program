import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


irisset = datasets.load_iris()


#The Iris dataset was used in R.A. Fisher's classic 1936 paper, 
#The Use of Multiple Measurements in Taxonomic Problems, 
#and can also be found on the UCI Machine Learning Repository.

# It includes three iris species with 50 samples each as well as 
#some properties about each flower.
# The 3 species of iris are
#Iris setosa, Iris virginica and Iris versicolor

#The columns in this dataset are:    
#Id
#SepalLengthCm
#SepalWidthCm
#PetalLengthCm
#PetalWidthCm
#Species
X = irisset.data[:50,0:1]
y = irisset.data[:50,1]

reg = LinearRegression().fit(X,y)

yPredict = reg.predict(X)
mse = mean_squared_error(y, yPredict)
r2 = r2_score(y,yPredict)






print('Train MSE =', mse)
print('Train R2 score =', r2)
print("\n")


plt.figure()
plt.scatter(y, yPredict, color='blue', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.title('Actual vs Predicted Values') 
plt.xlabel('Actual SepalWidth')
plt.ylabel('Predicted SepalWidth')
plt.grid()
plt.show()