import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



DiabetesData = pd.read_csv('pima_indians_diabetes.csv')

#The Pima Indians Diabetes Dataset involves predicting the onset of 
#diabetes within 5 years in Pima Indians given medical details.

#It is a binary (2-class) classification problem. 
#The number of observations for each class is not balanced. There are 768 observations 
#with 8 input variables and 1 output variable. Missing values are believed to be 
#encoded with zero values. The variable names are as follows:

#Number of times pregnant.
#Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
#Diastolic blood pressure (mm Hg).
#Triceps skinfold thickness (mm).
#2-Hour serum insulin (mu U/ml).
#Body mass index (weight in kg/(height in m)^2).
#Diabetes pedigree function.
#Age (years).
#Class variable (0 or 1).
#Goal: Predict the onset of diabetes within 5 years in Pima Indians given medical details.


X = DiabetesData.iloc[:, 0:8].values
Y = DiabetesData.iloc[:, 8].values










cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of Logistic Regression is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

LRscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Logistic Regression is',100*LRscore,'%\n')
