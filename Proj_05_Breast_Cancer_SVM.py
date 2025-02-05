from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

bcancer = datasets.load_breast_cancer()

#Breast Cancer Wisconsin (Diagnostic) Data Set
#Classes = 2
#Samples per class = 212(M),357(B)
# Samples total = 569
# Dimensionality = 30
# Features are real, positive

# Feature names
#'mean radius' 'mean texture' 'mean perimeter' 'mean area'
# 'mean smoothness' 'mean compactness' 'mean concavity'
# 'mean concave points' 'mean symmetry' 'mean fractal dimension'
# 'radius error' 'texture error' 'perimeter error' 'area error'
# 'smoothness error' 'compactness error' 'concavity error'
# 'concave points error' 'symmetry error' 'fractal dimension error'
# 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
# 'worst smoothness' 'worst compactness' 'worst concavity'
# 'worst concave points' 'worst symmetry' 'worst fractal dimension'

X = bcancer.data
Y = bcancer.target

scaler=StandardScaler();
X=scaler.fit_transform(X)

Xtrain,Xtest,Ytrain,Ytest\
    =train_test_split(X,Y,test_size=0.25,random_state=0)
#linear SVM
svmc=SVC(kernel="linear",random_state=0)
svmc.fit(Xtrain,Ytrain)
Ypred=svmc.predict(Xtest)
svmcscore=accuracy_score(Ypred,Ytest)

print('Accuracy score of Linear SVM Classifier is',100*svmcscore,'%\n')

cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with Linear Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()



# Kernel SVM RBF - Gaussian Kernal

ksvmc = SVC(kernel = 'rbf')
ksvmc.fit(Xtrain, Ytrain)
Ypred = ksvmc.predict(Xtest)
svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of SVM Classifier with RBF Kernel is',100*svmcscore,'%\n')


cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with RBF Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

# Kernel SVM Polynomial 

ksvmc = SVC(kernel = 'poly')
ksvmc.fit(Xtrain, Ytrain)
Ypred = ksvmc.predict(Xtest)
svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of SVM Classifier with Polynomial Kernel is',100*svmcscore,'%\n')


cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with Polynomial Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()



# Kernel SVM Sigmoid 

ksvmc = SVC(kernel = 'sigmoid')
ksvmc.fit(Xtrain, Ytrain)
Ypred = ksvmc.predict(Xtest)
svmcscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of SVM Classifier with Sigmoid Kernel is',100*svmcscore,'%\n')


cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of SVC with Sigmoid Kernel is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()
