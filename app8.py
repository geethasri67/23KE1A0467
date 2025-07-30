# step1:We have to import the dataset to the colab notebook
import pandas as pd # pd is a nick name of pandas
df=pd.read_csv('data csv 2.csv')#df is a variable,read_csv is  a predefined function from pandas package
type(df)
df
#step-2:grouping the input columns and output columns
X=df.iloc[:,:3]
X
Y=df.iloc[:,-1]
Y
X=X.values
X
Y=Y.values
Y
#step-3:Handling the missing data or values
#scenario-1:Missing values in numerical column
#scenario-2:missing values
#scenario-3:

import numpy as np
from sklearn.impute import SimpleImputer

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])


X
#step-4:Handling catogorial data
#Which converts the text data into numeric data.
#step-4:Handling catogorial data
#Which converts the text data into numeric data.
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')#encoder is a method which converts into  text to numeric
X=ct.fit_transform(X)
X
from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()
Y=Le.fit_transform(Y)
Y
#step-5: split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,)
X_train
X_test
Y_train
Y_test
#step-5: split the data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,)
X_train
X_test
Y_train
Y_test
#step6:-future scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])
X_train
X_test
#machine learning
#1.supervisioned learning
# and the model tasks are classifcation or prediction
#2.unsuprervisioned learning-unlabled data,model remembers the patterns
#we can perform prediction also
#3.reinforcement learning-trail and error method,model learns from failures

#logistic regression algorithm=when model perform classification task we have apply logistic regression alogrithm
#linear regression algorithm=when model perform prediction task we have to apply linear regression algorithm.
#step-7:applying algorithm to the training data for
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
Y_pred