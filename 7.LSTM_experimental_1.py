#!/usr/bin/env python
# coding: utf-8

# このipynbについての説明：
# 正規化なしデータを読み込み、NO NORML high corr indexes no miss.csv。meanとstdで全てデータを正規化する。seqを取り、学習と評価データセットに分けて、batchを作り、そしてモデル構築と学習評価を行う。
# 
# 例えば、seq＝１５、１から15日目までの他の指標データ（size：１５×２３、TPDR除く）を入力し、出力値は2から16日目までのこの間のTPDR平均値（size：１）と見なす。何故かこの出力値を平均値と見なすと言えば、出力値を15日毎の平均値とプロットすれば、とても接近だと見える。
# TPDR_val_day16=pred_value* 15 - sum(TPDR_val_day2+TPDR_val_day3+....TPDR_val_day15)

# In[133]:


import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt


# In[134]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[135]:


span=0


# In[136]:


df=pd.read_csv('NO NORML high corr indexes no miss.csv')
df=df.reset_index(drop=True)
df=df.drop(df.columns[[0]],axis=1)
#df.iloc[:,0]*=100
df


# In[137]:


#正規化を試みる
col_mean=df.mean()
col_std=df.std()
def mean_std_normalize(column):
    return (column - column.mean()) / (column.std())

# Normalize all columns of the DataFrame
df_normalized = df.apply(mean_std_normalize)
df_normalized


# In[138]:


df_normalized.shape


# In[139]:


df_normalized=df_normalized.astype(np.float64)


# In[182]:


# for one period, I choose the data ifrom 1st to 15nd day for X, and TPDR value form 2nd to 16th day as Y.

def data_spliter(df,seq):# use this at this time
    df_y=df.iloc[:,0]
    df_x=df.drop(df.columns[[0]],axis=1)
    x_series=[]
    y_series=[]
    
    for i in range(len(df_y)-seq):
        x_series.append(df_x[i:i+seq])
        y_series.append(df_y[i+1:i+1+seq])
        
    #split_pos=int(len(df_y)*0.8)
    split_point=1630
    split_pos=-1630
    x_train=x_series[:split_pos]
    x_test=x_series[split_pos:]
    y_train=y_series[:split_pos]
    y_test=y_series[split_pos:]

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    

    return x_train, y_train, x_test,y_test,split_pos


# In[183]:


seq=15
x_train, y_train, x_test,y_test, pos=data_spliter(df_normalized,seq)
pos


# In[186]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[185]:


y_train.shape


# In[187]:


print(y_train[0])


# In[188]:


x_train=torch.from_numpy(x_train.reshape(-1,seq,23)).type(torch.Tensor)   
y_train=torch.from_numpy(y_train.reshape(-1,seq)).type(torch.Tensor)   
x_test=torch.from_numpy(x_test.reshape(-1,seq,23)).type(torch.Tensor)   
y_test=torch.from_numpy(y_test.reshape(-1,seq)).type(torch.Tensor)   


# In[189]:


print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[190]:


train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=700,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=1,shuffle=False)


# In[191]:


input_dim=23
hidden_dim=256
num_layers=2
output_dim=1


# In[192]:


class LSTM(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,output_dim):
        super().__init__()
        self.input_dim=input_dim
        self.hidden_dim=hidden_dim
        self.num_layers=num_layers
        # x: (batch_size, seq, fea_number/input_size)
        self.lstm=nn.LSTM(input_dim,hidden_dim,num_layers,batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self,x):
        # Set initial hidden states
        # x: (batch_size, time_length, fea_number)
        # h0: (num_layers * num_directions, batch_size, hidden_size)
        # c0: (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,device=x.device).float()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,device=x.device).float()
       # Forward propagate RNN
        out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)        
        # Only need the hidden state of the last time step
        # out: (batch_size, hidden_size)
        out = out[:, -1, :]

        # out: (batch_size, 1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

      


# In[193]:


model=LSTM(input_dim=input_dim,hidden_dim=hidden_dim,num_layers=num_layers,output_dim=output_dim)
optimiser = torch.optim.Adam(model.parameters(), lr=0.008) # 使用Adam优化算法
loss_fn = torch.nn.MSELoss(size_average=True) 


# In[194]:


num_epochs = 180

# 打印模型结构
print(model)


# In[195]:


for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# In[196]:


#you can check the details of input data size here
for i,now in enumerate(train_loader):
    print(len(now))
    print('x info')
    print('batch size',len(now[0]))
    print('seq',len(now[0][0]))
    print('index num', len(now[0][0][0]))
    print(now[0][0][0])
    print('y info')
    print('batch size',len(now[1]))
    print('seq',len(now[1][0]))
    
    print(now[1][0])
    break


# In[197]:


model_name='meanstd_norml_lstm_seq_15_batch_700_epoch_180_lr0.008_y_singlevalue.pth'
train_losses = []
model.to('cuda')

early_stopping_counter=0
patience=10
smallest_loss=88888888

for epoch in range(num_epochs):
    loss_sum = 0
    model.train()
    for i,now in enumerate(train_loader):
        inputs ,labels = now
        
        inputs = inputs.to(device)#
        labels = labels.to(device)    
        outputs = model(inputs)
        
        loss = loss_fn(outputs,labels)
        #print("loss:",loss)
        loss_sum += loss
        optimiser.zero_grad()
        loss.backward()
        
        optimiser.step()
    
    train_losses.append(loss_sum.item())
    if loss_sum.item()<smallest_loss:
        smallest_loss=loss_sum.item()
        save_info='Save the model'
        torch.save(model,model_name)
    else:
        save_info=' '
    print(f"Epoch: {epoch+1}/{num_epochs},Loss:{loss_sum.item()},{save_info}")    


# In[198]:


plt.plot(train_losses)


# In[148]:


model=torch.load('./LSTMmodels/meanstd_norml_lstm_seq_15_batch_700_epoch_180_lr0.008.pth')


# In[199]:


for i,now in enumerate(test_loader):
    print(len(test_loader))
    print(len(now))
    print('x info')
    print('batch size',len(now[0]))
    print('seq',len(now[0][0]))
    print('index num', len(now[0][0][0]))
    print(now[0][0][0])
    print('y info')
    print('batch size',len(now[1]))
    print('seq',len(now[1][0]))
    
    print(now[1][0])
    break


# In[200]:


model.eval()
loss_sum = 0
preds = []
ans = []

with torch.no_grad():
    for inputs,labels in test_loader:
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)

        loss_sum += loss_fn(outputs,labels)
        
        preds.append(outputs[0].to('cpu').detach().numpy().copy())
        #ans.append(labels[0][1].to('cpu').detach().numpy().copy())
        ans.append(labels.to('cpu').detach().numpy().copy())
        pred = outputs.argmax(1)
        ans_label = labels.argmax(1)
        #ans = labels
        


# In[201]:


print(len(test_loader))
print(preds[0])
#preds_array=preds[0].to('cpu').detach().numpy().copy()
#preds_list=preds.tolist()
#preds_list[0]
new_pred=[]
for i in range(len(preds)):
    new_pred.append(preds[i][0])
new_pred
len(new_pred)
new_pred[-10:]


# In[232]:


# prediction data needs denormalize
a=col_mean['TargetPortDailyRtn']
b=col_std['TargetPortDailyRtn']
pred_denorml=[]
for i in range(len(new_pred)):
    denor_pred=new_pred[i]*b+a
    pred_denorml.append(denor_pred)
len(pred_denorml)


# In[203]:


# get the true TPDR value form the original data
ans_array=df.iloc[-1630:,0]
ans_array.reset_index(drop=True)
#ans_array=ans_array.to_list()

len(ans_array)


# In[239]:


pred_improved=[]
ans_ref=df.iloc[-len(pred_denorml)-seq:,0]
for i in range(len(pred_denorml)):
    seq_pred=pred_denorml[i]*seq-(ans_array[i-seq:i+1].sum())
    pred_improved.append(seq_pred)
print(len(pred_improved))
pred_improved[-10:]


# In[252]:


ans_2=[]#15 day mean value in valid dataset
ans_array=df.iloc[-len(pred_denorml):,0]
for i in range(len(ans_array)):
    ans_2.append(ans_array[i:i+seq].mean())
len(ans_2)
ans_2[-10:]


# In[206]:


ans_3=[]# (maximun+minimun)*0.5 for the maximun and minimun in 15 days
for i in range(len(ans_array)):
    mx=ans_array[i].max()
    mi=ans_array[i].min()
    ans_3.append((mx+mi)*0.5)


# In[207]:


ans_4=df.iloc[-len(pred_denorml):,0]
ans_4


# In[245]:


original_filepath="./original_data/"
df_org=pd.read_excel(original_filepath+'/2023_4_東大DSS用データ_20230331追加_更新.xlsx',sheet_name = None,engine = "openpyxl")

TPDR_df=df_org["市場データ_final"].iloc[-1630:,:4]
#pred_imp_df=pd.DataFrame(pred_improved,columns=['TPDR pred'])
#comparision_df=pd.merge(ans_4,pred_imp_df,left_index=True,right_index=True)

#comparision_df
TPDR_df


# In[246]:


TPDR_only_df=TPDR_df.drop(TPDR_df.columns[[1,2]],axis=1)
TPDR_only_df.rename(columns={df.columns[1]:'TPDR true value'},inplace=True)
TPDR_only_df=TPDR_only_df.reset_index(drop=True)
pred_imp_df=pd.DataFrame(pred_improved,columns=['TPDR pred'])
pred_imp_df=pred_imp_df.reset_index(drop=True)
TPDR_compare=pd.concat([TPDR_only_df,pred_imp_df],axis=1)

TPDR_compare.to_csv('TPDR true value and pred value.csv')


# In[247]:


TPDR_compare


# In[256]:


fig = plt.figure(figsize=(20, 4))
plt.scatter(TPDR_compare.iloc[:,0],TPDR_compare.iloc[:,1],color='forestgreen',alpha=0.3,s=2,label='TPDRtrue val')
plt.scatter(TPDR_compare.iloc[:,0],TPDR_compare.iloc[:,2],color='red',alpha=0.3,s=2,label='TPDR pred')
plt.axhline(y=0, color='green', linestyle='-',linewidth=1, alpha=0.2)
leg=plt.legend(loc='lower left')
plt.title('TPDR true value and pred value')
plt.show()


# In[255]:


fig = plt.figure(figsize=(20, 4))
plt.scatter(range(len(ans_2)),ans_2,color='forestgreen',alpha=0.3,s=2,label='mean15D')
#plt.scatter(range(len(ans_4)),ans_4,color='darkcyan',alpha=0.3,s=2,label='realval')
plt.scatter(range(len(pred_denorml)),pred_denorml,color='red',alpha=0.3,s=2,label='pred denml')
#plt.scatter(range(len(pred_improved)),pred_improved,color='blue',alpha=0.3,s=2,label='pred imp')
plt.axhline(y=0, color='green', linestyle='-',linewidth=1, alpha=0.2)
leg=plt.legend(loc='lower left')
plt.title('plotting of denormalized prediction value and the mean value of each 15 days in test set')
plt.show()


# In[167]:


# plot, if you plot with the average of 

fig = plt.figure(figsize=(50, 4))
stt=400
end=600
#end=len(preds_list)

x1=range(stt,end)

size1=[15 if value > 0 else 1 for value in pred_improved]
size2=[15 if value > 0 else 1 for value in ans_4]
for x, y in zip(x1,  pred_improved[stt:end]):
    if y>0:
        plt.scatter(x, y, color='red',alpha=0.5,s=8)
    else:
        plt.scatter(x, y, color='red',alpha=0.5,s=1)

for x, y in zip(x1,  ans_4[stt:end]):
    if y>0:
        plt.scatter(x, y, color='blue',alpha=0.5,s=8)
    else:
        plt.scatter(x, y, color='blue',alpha=0.5,s=1)

plt.axhline(y=0, color='green', linestyle='-',linewidth=1, alpha=0.2)
#plt.scatter(x1, pred_improved[stt:end],color='red',alpha=0.5,s=size1,label='predticted valid value')
#plt.scatter(x1,ans_4[stt:end],color='blue',alpha=0.5,s=size2,label='true valid value')
#leg=plt.legend(loc='lower left')
plt.title('predicted and the TPDR value in 15 days by LSTM model')
plt.show()


# In[165]:


preds_list=pred_improved
fig = plt.figure(figsize=(20, 4))
plt.scatter(range(len(preds_list)),preds_list,color='red',alpha=0.3)
plt.scatter(range(len(ans_4)),ans_4,color='blue',alpha=0.3)
plt.axhline(y=0, color='green', linestyle='-',linewidth=1, alpha=0.2)
plt.show()


# In[69]:





# In[ ]:




