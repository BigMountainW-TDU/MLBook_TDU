#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
import pickle


# In[19]:


# pandas.DataFrame を用いてCSVファイルを読み込む（p.34 : 2.3.20)
data = pd.read_csv('test.csv')

# Y : ラベルデータを numpy.array 形式で取得する
Y = data['label'].values

# X : 画像データを numpy.array 形式で取得する．
data = data.drop(['id','label'],axis=1)
X = data.values


# In[20]:


pcaModel = pickle.load(open('pca_mdl.pkl','rb'))
clfModel = pickle.load(open('clf_mdl.pkl','rb'))


# In[21]:


Xpca = pcaModel.transform(X)
predicted = clfModel.predict(Xpca)


# In[24]:


output = pd.read_csv('submission.csv')
output['label'] = predicted


# In[23]:


output.to_csv('submission.csv',columns=['id','label'], index=False)


# In[ ]:




