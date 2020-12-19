#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_iris
ld = load_iris()
iris = pd.DataFrame(ld.data, columns = ld.feature_names)
iris
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(iris)
    kmeanModel.fit(iris)
    distortions.append(sum(np.min(cdist(iris, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / iris.shape[0])
    
    # Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#Repeating same steps for wine dataset

from sklearn.datasets import load_wine
ld2 = load_wine()
wine = pd.DataFrame(ld2.data, columns = ld2.feature_names)
wine
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(wine)
    kmeanModel.fit(wine)
    distortions.append(sum(np.min(cdist(wine, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / wine.shape[0])
    
    plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

