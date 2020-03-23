# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing  import LabelEncoder as skl
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

train = pd.read_csv('D:\ESCP\Machine learning with python\\train.csv')
test = pd.read_csv('D:\ESCP\Machine learning with python\\Assignment 2 and 3\\test1.csv')

train.head(10)
test.head(10)

train.columns
test.columns

train.isnull().sum()
test.isnull().sum()

#####Clean datasets#####
train['label']= train['label'].replace(-1,0)
test['label']= test['label'].replace(-1,0)

#First on train
train.loc[train["purchaseTime"] ==-1, 'TimeBeforePurchase?'] = 0 
train.loc[train["purchaseTime"] != -1, 'TimeBeforePurchase?'] =  train["purchaseTime"] - train["visitTime"]

train = train.drop({'id','visitTime','purchaseTime'}, axis = 1)
cat_list = []
for i in range(1,13):
    cat_list.append(len(train["C%d"%i].unique()))
cat_list

train = train.drop({"C1","C10"}, axis=1)

corr = train.corr()
sns.heatmap(corr)

train = train.drop({"C4","N4","N8"}, axis = 1) 

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

train["C5"] = label_encoder.fit_transform(train.iloc[:,4]).astype('int')
train["C2"] = label_encoder.fit_transform(train.iloc[:,2]).astype('int')
train["C3"] = label_encoder.fit_transform(train.iloc[:,3]).astype('int')
train["C6"] = label_encoder.fit_transform(train.iloc[:,5]).astype('int')
train["C7"] = label_encoder.fit_transform(train.iloc[:,6]).astype('int')
train["C8"] = label_encoder.fit_transform(train.iloc[:,7]).astype('int')
train["C9"] = label_encoder.fit_transform(train["C9"]).astype('int')
train["C11"] = label_encoder.fit_transform(train["C11"]).astype('int')
train["C12"] = label_encoder.fit_transform(train["C12"]).astype('int')
corr2 = train.corr()
sns.heatmap(corr2)
train = train.drop({"N9"}, axis = 1) 

#Apply same on test
test.loc[test['purchaseTime'] ==-1, 'TimeBeforePurchase?'] = 0 
test.loc[test['purchaseTime'] != -1, 'TimeBeforePurchase?'] =  test["purchaseTime"] - test["visitTime"]

test_id = test.iloc[:,1]
test = test.drop({'id','visitTime','purchaseTime'}, axis = 1)
test = test.drop({"C1","C10","C4","N4","N8", "N9"}, axis=1)

test["C5"] = label_encoder.fit_transform(test.iloc[:,4]).astype('int')
test["C2"] = label_encoder.fit_transform(test.iloc[:,2]).astype('int')
test["C3"] = label_encoder.fit_transform(test.iloc[:,3]).astype('int')
test["C6"] = label_encoder.fit_transform(test.iloc[:,5]).astype('int')
test["C7"] = label_encoder.fit_transform(test.iloc[:,6]).astype('int')
test["C8"] = label_encoder.fit_transform(test.iloc[:,7]).astype('int')
test["C9"] = label_encoder.fit_transform(test["C9"]).astype('int')
test["C11"] = label_encoder.fit_transform(test["C11"]).astype('int')
test["C12"] = label_encoder.fit_transform(test["C12"]).astype('int')




### BALANCE DATASET AND MODEL ###

len(train["label"])
sum(train["label"])
train.head()


#oversampling
label = train.label
features = train.drop('label', axis=1)

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=123)

X_train, y_train=os.fit_sample(features, label)

X_train = preprocessing.scale(X_train)

model = LogisticRegression().fit(X_train, y_train)

test1= test.iloc[:,1:20]
predictions = model.predict_proba(test1)[:,1]
y_pred = model.predict(test1)

sum(y_pred)
len(y_pred)

final_table = [test_id, predictions]
final_table= pd.DataFrame({"id_client" :test_id, "predicted_prob" :predictions})

final_table.to_csv('assignment2_3.csv', index=False)
