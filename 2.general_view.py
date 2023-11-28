#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


df=pd.read_csv('./all_data_after_process.csv')
df['Dates']=df['Dates'].astype('datetime64[ns]')
df=df.drop(df.columns[[0,1]],axis=1)
df


# In[27]:


df.dtypes


# In[28]:


df.describe()


# In[33]:


a=df.corr()
print(a['TargetPortDailyRtn'].nlargest(15))
print(a['TargetPortDailyRtn'].nsmallest(15))


# In[30]:


plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="plasma",annot=False)
plt.title("Correlation Matrix")

plt.show()


# In[31]:


b=a['TargetPortDailyRtn']
b.sort_values(ascending=False)
print(b)


# In[32]:


plt.scatter(df['ESIndex'],df['TargetPortDailyRtn'],alpha=0.5)
plt.xlabel('ESIndex')
plt.ylabel('TPDR')
plt.show()

