from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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

X = irisset.data[:100,:]
Y = irisset.target[:100]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.20, random_state = 5) 

plt.figure(1);
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.suptitle('Original IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

lda=LinearDiscriminantAnalysis()
lda.fit(Xtrain,Ytrain)
Ypred=lda.predict(X)




plt.figure(1);
plt.scatter(X[:, 0], X[:, 1], c = Ypred)
plt.suptitle('Predicted IRIS Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

cmat = confusion_matrix(Y, Ypred)
print('Confusion matrix of LDA is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

LDAscore = accuracy_score(Y, Ypred )
print('Accuracy score of LDA is',100*LDAscore,'%\n')