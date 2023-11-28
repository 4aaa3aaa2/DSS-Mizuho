#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df=pd.read_csv('normalized_all_data.csv')
df=df.drop(df.columns[[0,1]],axis=1)
df


# In[4]:


a=df.corr()
corr_info=a['TargetPortDailyRtn']

top_50_cols =corr_info.abs().nlargest(50).index.tolist()
print("Columns with the highest 10 correlations to 'results':")
rankA=0
rankB=0
rankC=0
missing_accept=0
A_list=[]
for col in top_50_cols:
    corr_value = corr_info[col]
    missing_count = df[col].isnull().sum()
    if abs(corr_value)>0.1 and missing_count<=400:
        rankA+=1
        A_list.append(col)
        if missing_count>missing_accept:
            missing_accept=missing_count
    if missing_count<=400 and abs(corr_value)<=0.1 and abs(corr_value)>=0.03:
        rankB+=1
        A_list.append(col)
        if missing_count>missing_accept:
            missing_accept=missing_count
    else:
        rankC+=1
    print(f"Column: {col}, Correlation: {corr_value}, Missing Data: {missing_count}")
print(rankA,rankB,rankC)
print(A_list)
print(missing_accept)

#high_corr_df=df["Dates","Year","Month","Day","DayOfWeek","Holiday"]+top_50_cols]


# In[5]:


df_high_corr=df[A_list]
df_high_corr


# In[6]:


df_high_corr_nomissing=df_high_corr.dropna(axis=0)


# In[7]:


df_high_corr_nomissing


# In[8]:


df_high_corr_nomissing.to_csv('high corr indexes from normalized data, no missing.csv')


# In[16]:


df_high_corr_2=df_high_corr_nomissing.copy()
df_high_corr_2=df_high_corr_2.reset_index(drop=True)
df_high_corr_2


# In[ ]:





# In[17]:


for i in range(0, df_high_corr_2.shape[0]):
    try:
        df_high_corr_2.iloc[i,0] = df_high_corr_2.iloc[i+1, 0]
    except:
        df_high_corr_2.iloc[i, 0] = np.nan


# In[18]:


df_high_corr_2


# In[ ]:


df_high_corr_2.to_csv('TPDR +1 Day, high corr indexes normalized, nomiss.csv')

