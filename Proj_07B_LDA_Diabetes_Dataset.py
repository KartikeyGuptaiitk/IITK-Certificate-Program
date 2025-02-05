from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

DiabetesData = pd.read_csv('pima_indians_diabetes.csv')

X = DiabetesData.iloc[:, :7].values
Y = DiabetesData.iloc[:, 8].values

scaler = StandardScaler();
X = scaler.fit_transform(X) 

Xtrain, Xtest, Ytrain, Ytest \
= train_test_split(X, Y, test_size = 0.20, random_state = 5) 

plt.figure(1);
plt.scatter(X[:, 1], X[:, 2], c = Y)
plt.suptitle('Original Diabetes Data')
plt.xlabel('Scaled Glucose')
plt.ylabel('Scaled Blood Pressure')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

lda=LinearDiscriminantAnalysis()
lda.fit(Xtrain, Ytrain)
Ypred=lda.predict(X)




plt.figure(1);
plt.scatter(X[:, 1], X[:, 2], c = Ypred)
plt.suptitle('Predicted Diabetes Data')
plt.xlabel('Scaled Glucose')
plt.ylabel('Scaled Blood Pressure')
plt.grid(1,which='both')
plt.axis('tight')
plt.show()

ldascore = accuracy_score(lda.predict(Xtest),Ytest)
print('Accuracy score of LDA Classifier is',100*ldascore,'%\n')

cmat = confusion_matrix(lda.predict(Xtest),Ytest)
print('Confusion matrix of LDA is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

