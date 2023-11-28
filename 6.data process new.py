#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os 
import pandas as pd
import numpy as np
original_filepath="./original_data/"


# In[4]:


df1=pd.read_excel(original_filepath+'/2023_4_東大DSS用データ_4本値と出来高_20230331.xlsx',sheet_name = None,engine = "openpyxl" )
df2=pd.read_excel(original_filepath+'/2023_4_東大DSS用データ_20230331追加_更新.xlsx',sheet_name = None,engine = "openpyxl")


# In[5]:


original_val=df1['Sheet1']
orig_mkt_data=df2['市場データ_final']
orig_idx_data=df2['経済指標データ_final']


# In[6]:


mkt_data=orig_mkt_data.loc[6:]
mkt_data.columns=list(orig_mkt_data.iloc[5])
mkt_data=mkt_data.astype({'Dates':'datetime64[ns]'})
mkt_data.head()


# In[7]:


idx_data=orig_idx_data.loc[6:]
idx_data.columns=list(orig_idx_data.iloc[5])
idx_data=idx_data.astype({'Dates':'datetime64[ns]'})
idx_data.head()


# In[15]:


df_all=mkt_data
df_all=pd.merge(mkt_data,idx_data,on="Dates")
df_all=pd.merge(df_all,original_val,on='Dates')
df_all.head(10)


# In[17]:


col1='ESIndex'
col2='TargetPortDailyRtn'
columns=df_all.columns.tolist()
index1 = columns.index(col1)
index2 = columns.index(col2)
columns[index1], columns[index2] = columns[index2], columns[index1]
df_all=df_all[columns]
df_all


# In[18]:


eliminate_list = ["USGG3YRIndex","USYC2Y3YIndex","USYC3Y5YIndex","USYC3Y7YIndex","USYC3Y10Index","USYC3Y20Index","USYC3Y30Index","BF020305Index","BF020307Index","BF020310Index","BF020320Index","BF020330Index","BF030507Index","BF030510Index","BF030520Index","BF030530Index","BF030710Index","BF030720Index","BF030730Index","BF031020Index","BF031030Index"]
for i in range(len(df_all.columns)):
    if df_all.isnull().all()[i] == True:
        eliminate_list.append(df_all.columns[i])
df_all = df_all.drop(eliminate_list, axis=1)
df_all.shape


# In[20]:


df_new=df_all.copy()
df_new=df_new.fillna(method='ffill')
df_new


# In[21]:


df_left=df_new.iloc[:,:2]
df_left


# In[22]:


df_right=df_new.iloc[:,2:]
df_right


# In[23]:


df_right=df_right.apply(lambda x: (x-x.mean())/ x.std(), axis=0)#定常化操作
df_right.head()


# In[24]:


normalized_df=pd.concat([df_left,df_right],axis=1)
normalized_df


# In[27]:


normalized_df['TargetPortDailyRtn'].min()


# In[34]:


import matplotlib.pyplot as plt
data=normalized_df['TargetPortDailyRtn']
plt.scatter(data.index,data,alpha=0.5,s=1)
plt.show()


# In[49]:


ddf=normalized_df
a=ddf.corr()
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
    missing_count = ddf[col].isnull().sum()
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


# In[54]:


df_high_corr=ddf[A_list]
df_high_corrnomissing=df_high_corr.dropna(axis=0)
df_high_corr.iloc[353,:]


# In[56]:


df_high_corrnomissing.to_csv('norml high corr indexes, TPDR no norml, nomiss.csv')


# In[40]:


b=df_new.corr()
corr_info_new=b['TargetPortDailyRtn']

top_50_cols_new =corr_info.abs().nlargest(50).index.tolist()
print("Columns with the highest 10 correlations to 'results':")
rank1=0
rank2=0
rank3=0
missing_accept=0
B_list=[]
for col in top_50_cols_new:
    corr_value = corr_info_new[col]
    missing_count = df_new[col].isnull().sum()
    if abs(corr_value)>0.1 and missing_count<=400:
        rank1+=1
        B_list.append(col)
        if missing_count>missing_accept:
            missing_accept=missing_count
    if missing_count<=400 and abs(corr_value)<=0.1 and abs(corr_value)>=0.03:
        rank2+=1
        B_list.append(col)
        if missing_count>missing_accept:
            missing_accept=missing_count
    else:
        rank3+=1
    print(f"Column: {col}, Correlation: {corr_value}, Missing Data: {missing_count}")
print(rank1,rank2,rank3)
print(B_list)
print(missing_accept)


# In[45]:


df_high_corr_not_norml=df_new[B_list]
df_high_corr_nonorml_nomissing=df_high_corr_not_norml.dropna(axis=0)
df_high_corr_nonorml_nomissing.to_csv('NO NORML high corr indexes no miss.csv')


# In[51]:


df_high_corr_nonorml_nomissing


# In[ ]:




