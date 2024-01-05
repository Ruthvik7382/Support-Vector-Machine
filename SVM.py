#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, roc_curve, auc
from sklearn.utils import shuffle 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm
from sklearn.naive_bayes import CategoricalNB


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# # Data preprocessing

# In[3]:


df = pd.read_csv('malware.csv')
df = shuffle(df)
df = df[['classification', 'os', 'usage_counter', 'prio', 'static_prio', 'normal_prio', 'vm_pgoff', 'vm_truncate_count', 'task_size', 'map_count', 'hiwater_rss', 'total_vm', 'shared_vm', 'exec_vm', 'reserved_vm', 'nr_ptes', 'nvcsw', 'nivcsw', 'signal_nvcsw']]
y = df['classification'] 
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
df['classification'] = y_encoded

df_dummies = pd.get_dummies(df["os"])
df = df.join(df_dummies)
df = df.drop('os', axis=1)
df = df.drop('CentOS', axis=1)

df = df.drop(['usage_counter','normal_prio','vm_pgoff','task_size','hiwater_rss','nr_ptes','signal_nvcsw'],axis=1)

y_encoded = df['classification']
df = df.drop('classification', axis=1)

x= df
y= y_encoded


# In[4]:


from sklearn.preprocessing import StandardScaler
 
# compute required values
scaler = StandardScaler()
model = scaler.fit(df)
scaled_data = model.transform(df)


# In[7]:


df


# In[8]:


from sklearn.preprocessing import MinMaxScaler
df_scaled = scaler.fit_transform(df.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=[
  'prio', 'static_prio', 'vm_truncate_count', 'map_count','total_vm','shared_vm','exec_vm','reserved_vm','nvcsw','nivcsw','Debian','Mac','Ubuntu','Windows'])
 


# In[9]:


df_scaled.head(5)


# # SVM

# In[10]:


from sklearn.model_selection import train_test_split
 
# To calculate the accuracy score of the model
from sklearn.metrics import accuracy_score, confusion_matrix
 
target = y
features = df
X_train, X_test, y_train, y_test = train_test_split(df_scaled, y, test_size = 0.25, random_state = 10)
from sklearn.svm import SVC
 
# Building a Support Vector Machine on train data
svc_model = SVC(C= .01, kernel='linear', gamma= 1)
svc_model.fit(X_train, y_train)
 
prediction = svc_model .predict(X_test)
# check the accuracy on the training set
print(svc_model.score(X_train, y_train))
print(svc_model.score(X_test, y_test))


# In[11]:


from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, roc_curve, auc
acc_LR = accuracy_score(y_test, prediction)
f1_LR = f1_score(y_test, prediction)

fpr1,tpr1,_=roc_curve(y_test,prediction)

roc_auc_LR=auc(fpr1,tpr1)

print('By hold-out evaluation: acc = ',acc_LR)
print('By hold-out evaluation: f1_LR = ',f1_LR)
print('By hold-out evaluation: roc_score_LR = ',roc_auc_LR)


# In[14]:


clf= SVC(C= .01, kernel='linear', gamma= 1)

acc=cross_val_score(clf, df_scaled, y, cv=10, scoring='accuracy').mean()
f1_score_LR = cross_val_score(clf, df_scaled, y, scoring="f1", cv = 10).mean()
auc = make_scorer(roc_auc_score, average='macro', needs_proba=True)
auc_LR = cross_val_score(clf, df_scaled, y, cv=10, scoring=auc).mean()

print('By N-fold Cross Validation: accuracy_LR = ',acc)
print('By N-fold Cross Validation: f1_score_LR = ',f1_score_LR)

