import numpy as np
import pandas as pd
import csv

train = pd.read_csv('train.csv')

cleaned_train = train

train['Target'] = train['Target'].replace({'Enrolled': 2, 'Graduate': 0, 'Dropout': 1}).astype(int)


print(cleaned_train['Target'].unique())
print(cleaned_train)
cleaned_train.to_csv('cleaned.csv', index=False)
