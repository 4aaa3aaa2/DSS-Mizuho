#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import pandas as pd
import numpy as np
import datetime as dt
import holidays

original_filepath="./original_data/"


# read in the excels and take a look at the original data

# In[3]:


df1=pd.read_excel(original_filepath+'/2023_4_東大DSS用データ_4本値と出来高_20230331.xlsx',sheet_name = None,engine = "openpyxl" )
df2=pd.read_excel(original_filepath+'/2023_4_東大DSS用データ_20230331追加_更新.xlsx',sheet_name = None,engine = "openpyxl")


# In[4]:


df1["Sheet1"].head(10)


# In[5]:


df2["市場データ_final"].head(10)


# In[6]:


df2['経済指標データ_final'].head(10)


# seperate the original data and the indexes

# In[7]:


original_val=df1['Sheet1']
orig_mkt_data=df2['市場データ_final']
orig_idx_data=df2['経済指標データ_final']


# In[8]:


mkt_data=orig_mkt_data.loc[6:]
mkt_data.columns=list(orig_mkt_data.iloc[5])
mkt_data=mkt_data.astype({'Dates':'datetime64[ns]'})
mkt_data.head()


# In[9]:


idx_data=orig_idx_data.loc[6:]
idx_data.columns=list(orig_idx_data.iloc[5])
idx_data=idx_data.astype({'Dates':'datetime64[ns]'})
idx_data.head()


# In[10]:


df_all=mkt_data
df_all=pd.merge(mkt_data,idx_data,on="Dates")
df_all=pd.merge(df_all,original_val,on='Dates')
df_all.head(10)


# In[11]:


df_all.shape


# In[12]:


eliminate_list = ["USGG3YRIndex","USYC2Y3YIndex","USYC3Y5YIndex","USYC3Y7YIndex","USYC3Y10Index","USYC3Y20Index","USYC3Y30Index","BF020305Index","BF020307Index","BF020310Index","BF020320Index","BF020330Index","BF030507Index","BF030510Index","BF030520Index","BF030530Index","BF030710Index","BF030720Index","BF030730Index","BF031020Index","BF031030Index"]
for i in range(len(df_all.columns)):
    if df_all.isnull().all()[i] == True:
        eliminate_list.append(df_all.columns[i])
eliminate_list


# In[13]:


df_all = df_all.drop(eliminate_list, axis=1)
df_all.shape


# In[14]:


df_all.head()


# In[15]:


from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
usb = CustomBusinessDay(calendar = USFederalHolidayCalendar())
us_holiday = pd.date_range('1/4/1999', '31/3/2022', freq=usb)


# In[16]:


df_all['Dates']=df_all['Dates'].astype('datetime64[ns]')#if you dont change the form of this col into datetime64, it wont work!!!
df_all.insert(0,'Index2',df_all.index)
df_all.insert(2, 'Year', df_all['Dates'].dt.year)
df_all.insert(3, 'Month', df_all['Dates'].dt.month)
df_all.insert(4, 'Day', df_all['Dates'].dt.day)
df_all.insert(5, 'DayOfWeek', df_all['Dates'].dt.dayofweek)
df_all.insert(6, 'Holiday',np.nan)
df_all


# In[17]:


for i in range(1,df_all.shape[0]):
    df_all.iloc[i,6] = int(df_all.iloc[i,0] in us_holiday)


df_all.iloc[0,6]= int(df_all.iloc[0,6] not in us_holiday)



df = df_all.reset_index()
df = df.drop('index', axis=1)
df.head()


# In[18]:


df.loc[0,'Holiday'] = 0
df = df.astype({'Holiday': 'int'})
df


# In[20]:


df_new=df.copy()


# In[27]:


df_new=df_new.fillna(method='ffill')
df_new


# In[28]:


df_new.to_csv('all_data_after_process.csv')


# In[29]:


df_left=df_new.iloc[:,:7]
df_left


# In[30]:


df_right=df_new.iloc[:,7:]
df_right


# In[31]:


df_right=df_right.apply(lambda x: (x-x.mean())/ x.std(), axis=0)#定常化操作
df_right.head()


# In[32]:


df_right.mean()


# In[33]:


df_right.std()


# In[34]:


normalized_df=pd.concat([df_left,df_right],axis=1)
normalized_df


# In[35]:


normalized_df.to_csv('normalized_all_data.csv')

