import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LogisticRegression 
from sklearn import tree


train = pd.read_csv('cleaned.csv')
test = pd.read_csv('test.csv')

ans = pd.DataFrame()


# First, predict if the student is enrolled
#train['Is_Enrolled'] = train['Target'].apply(lambda x: 1 if x == 'Enrolled' else 0)


#X_train1=train.drop(columns = ['Target','id','Is_Enrolled'])
#y_train1=train['Is_Enrolled']

X_train1=train.drop(columns = ['Target','id'])
y_train1=train['Target']

clf = tree.DecisionTreeClassifier(max_depth = 5)
clf = clf.fit(X_train1, y_train1)
predictions = clf.predict(test.drop(columns = ['id'])) 

pre = []
for i in predictions:
    if i == 0:
        pre.append('Graduate')
    elif i == 1:
        pre.append('Dropout')
    else:
        pre.append('Enrolled')
ans = pd.DataFrame({'id': test['id'], 'Target': pre})
ans.to_csv('ans.csv', index=False)