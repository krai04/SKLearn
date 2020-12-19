#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.datasets import load_boston

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

boston = load_boston()

print(boston.DESCR)

bos = pd.DataFrame(boston.data, columns = boston.feature_names)

bos.head()

plt.subplots(figsize=(20,15))
sns.heatmap(bos.corr(), annot=True)

bos.drop(['INDUS', 'NOX', 'TAX', 'AGE'], axis=1)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts

X = bos.drop(['LSTAT'],1)
y = bos['LSTAT']

X_train, X_test, y_train, y_test = tts(X, y, test_size=0.3,random_state=42)

lr = LinearRegression(copy_X= True, fit_intercept = True)

lr.fit(X_train, y_train)

lr_pred= lr.predict(X_test)

lr.score(X_test,y_test)

importance = lr.coef_

for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

feature_names = bos.columns

plt.bar([x for x in range(len(importance))], importance)
plt.xticks(range(bos.shape[1]), feature_names)
plt.xticks(rotation=90)
plt.xlim([-1, bos.shape[1]])
plt.show()

