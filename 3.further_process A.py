#!/usr/bin/env python
# coding: utf-8

# here, we mainly discuss about the normalized data, and from the 2nd step, the correlation of some indexes to the TargerPortDailyRtn has been shown. Considering there lacks some necessary data in some columns, in this step we will do the process, to keep part of them, or to throw away that column or some rows to get a more complete data.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In this part, we 1stly try to findout the correlationship of all the rest indexes to the TPDR, and keep the largest several ones and the smallest 10. NOTICE THAT WHAT WE FOUND HERE MAYBE NOT THAT SATISFYING BECAUSE SOME DATA REALLY MISSING A LOT!!!

# In[2]:


df=pd.read_csv('normalized_all_data.csv')
df=df.drop(df.columns[[0,1]],axis=1)
df


# In[3]:


a=df.corr()
corr_info=a['TargetPortDailyRtn']
plt.figure(figsize=(180,10))
plt.scatter(range(len(corr_info)), corr_info)
plt.xticks(range(len(a)), a.index)
plt.xlabel('corr')
plt.ylabel('TPDR')
plt.hlines(y=[0,0.2,0.4,0.6,0.8],xmin=0,xmax=len(corr_info), color='red', linestyle='--')

plt.show()

print(a['TargetPortDailyRtn'].nlargest(15))
print(a['TargetPortDailyRtn'].nsmallest(15))


# In[4]:


df.shape


# In[5]:


target_col='TargetPortDailyRtn'
threshold=0.07
relevant_columns = a[a[target_col] > threshold].index.tolist()
#relevant_columns.remove(target_col)
smallest_correlations = a[target_col].nsmallest(10).index.tolist()
print(relevant_columns)
print(smallest_correlations)
columns_to_keep = set(['Dates']+relevant_columns + smallest_correlations)
columns_to_keep


# In[6]:


df_keep=df[columns_to_keep]
df_keep


# till here, the step of finding some data of high correlation is over. but you need to notice that the result is not the final ones because some lacking a lot of data so that the corr seems to be very high.

# in the next step, we again consider all the data, and findout some columns that missing too many data. the missing threshold here set to be 1000  at 1st, and we can adjust it later.

# In[7]:


missing_data_counts = df.isnull().sum()
missing_data_counts.sort_values(ascending=False)
for column_name, missing_count in missing_data_counts.iteritems():
    if missing_count > 1000:
        print("Column '{}' has {} missing values.".format(column_name, missing_count))


# In[15]:


delete_list = missing_data_counts[missing_data_counts > 500].index.tolist()
df_new=df.drop(delete_list,axis=1)
print(len(delete_list))
print(delete_list)
df_new


# In[16]:


df_new_better=df_new.dropna(axis=1)
df_new_better


# In[14]:


c=df_new.corr()
corr_info=c['TargetPortDailyRtn']

top_20_cols =corr_info.abs().nlargest(20).index.tolist()
print("Columns with the highest 10 correlations to 'results':")
for col in top_20_cols:
    corr_value = corr_info[col]
    missing_count = df[col].isnull().sum()
    print(f"Column: {col}, Correlation: {corr_value}, Missing Data: {missing_count}")

#high_corr_df=df_new[["Dates"]+top_20_cols]
high_corr_df=df_new[["Dates","Year","Month","Day","DayOfWeek","Holiday"]+top_20_cols]

'''#print(corr_info.nlargest(15))
#print(corr_info.nsmallest(15))
target_col='TargetPortDailyRtn'
threshold=0.07
relevant_columns_new= c[c[target_col] > threshold].index.tolist()
#relevant_columns.remove(target_col)
smallest_correlations_new = c[target_col].nsmallest(10).index.tolist()
print(relevant_columns_new)
print(smallest_correlations_new)
columns_to_keep_new = set(['Dates']+relevant_columns_new+ smallest_correlations_new)
columns_to_keep_new'''


# In[11]:


high_corr_df


# In[12]:


high_corr_df_new=high_corr_df.dropna(subset=['VGIndex'])
high_corr_df_new


# In[ ]:


high_corr_df_new.to_csv('highest_20_corr_indexes_with_date_details.csv',index=False)


# In[13]:


df_new_better.corr()

'''corr_info_better=d['TargetPortDailyRtn']

top_20_cols_better =corr_info.abs().nlargest(20).index.tolist()
print("Columns with the highest 10 correlations to 'results':")
for col in top_20_cols_better:
    corr_value = corr_info_better[col]
    missing_count = df[col].isnull().sum()
    print(f"Column: {col}, Correlation: {corr_value}, Missing Data: {missing_count}")

#high_corr_df=df_new[["Dates"]+top_20_cols]
high_corr_df_better=df_new_better[["Dates","Year","Month","Day","DayOfWeek","Holiday"]+top_20_cols_better]'''

