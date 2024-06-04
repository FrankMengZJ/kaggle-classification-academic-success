import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 


train = pd.read_csv('cleaned.csv')
test = pd.read_csv('test.csv')

ans = pd.DataFrame()

X_train=train.drop(columns = ['Target','id'])
y_train=train['Target']

model = LogisticRegression(multi_class = 'ovr', solver='saga')
model.fit(X_train, y_train)
predictions = model.predict(test.drop(columns = ['id'])) 

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