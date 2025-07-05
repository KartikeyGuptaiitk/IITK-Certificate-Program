import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score



purchaseData = pd.read_csv('creditcard.csv')

# The dataset contains transactions made by credit cards in September 2013 by 
# European cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds 
# out of 284,807 transactions. 

# First feature 'Time' - Not used   

# Features V1, V2, … V28 are not named due to confidentiality
# The feature 'Amount' is the transaction Amount
# Total number of features = 29
# the first transaction in the dataset. 

# Feature 'Class' is the response variable 
# and it takes value 1 in case of fraud and 0 otherwise.


X = purchaseData.iloc[:, 1:30].values
Y = purchaseData.iloc[:, 30].values










cmat = confusion_matrix(Ytest, Ypred)
print('Confusion matrix of Logistic Regression is \n',cmat,'\n')

disp = ConfusionMatrixDisplay(confusion_matrix=cmat)
disp.plot()

LRscore = accuracy_score(Ypred,Ytest)
print('Accuracy score of Logistic Regression is',100*LRscore,'%\n')
