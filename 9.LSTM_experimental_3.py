#!/usr/bin/env python
# coding: utf-8

# このipynbについての説明：
# 正規化なしデータを読み込み、NO NORML high corr indexes no miss.csv。meanとstdで全てデータを正規化する。seqを取り、学習と評価データセットに分けて、batchを作り、そしてモデル構築と学習評価を行う。
# 
# 例えば、seq＝１５、１から15日目までの他の指標データ（size：１５×２３、TPDR除く）を入力し、出力値は2から16日目までのこの間のTPDR平均値（size：１）と見なす。何故かこの出力値を平均値と見なすと言えば、出力値を15日毎の平均値とプロットすれば、とても接近だと見える。
# TPDR_val_day16=pred_value* 15 - sum(TPDR_val_day2+TPDR_val_day3+....TPDR_val_day15)

# In[2]:


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


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[4]:


span=0


# In[195]:


df=pd.read_csv('NO NORML high corr indexes no miss.csv')
df=df.reset_index(drop=True)
df=df.drop(df.columns[[0]],axis=1)
df.iloc[-10:,:]


# In[196]:


#正規化を試みる
col_mean=df.mean()
col_std=df.std()
def mean_std_normalize(column):
    return (column - column.mean()) / (column.std())

# Normalize all columns of the DataFrame
df_normalized = df.apply(mean_std_normalize)
df_normalized


# In[197]:


df_normalized.shape
df_normalized=df_normalized.astype(np.float64)


# In[233]:


# for one period, I choose the data ifrom 1st to 15nd day for X, and TPDR value form 2nd to 16th day as Y.

def data_spliter(df,seq):# use this at this time
    df_y=df.iloc[:,0]
    df_x=df.drop(df.columns[[0]],axis=1)
    
    x_series=[]
    y_series=[]
    
    for i in range(seq-1,len(df_y)-5):
        x_series.append(df_x[i-seq+1:i+1])
        y_series.append(df.iloc[i+1:i+6,0].min())
    #split_pos=int(len(df_y)*0.8)
    split_pos=-1625
    x_train=x_series[:split_pos]
    x_test=x_series[split_pos:]
    y_train=y_series[:split_pos]
    y_test=y_series[split_pos:]

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    

    return x_train, y_train, x_test,y_test,split_pos


# In[280]:


seq=15
x_train, y_train, x_test,y_test, pos=data_spliter(df_normalized,seq)
pos


# In[281]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[282]:


y_test[-30:]


# In[283]:


print(y_train[0])


# In[284]:


x_train=torch.from_numpy(x_train.reshape(-1,seq,23)).type(torch.Tensor)   
y_train=torch.from_numpy(y_train.reshape(-1,1)).type(torch.Tensor)   
x_test=torch.from_numpy(x_test.reshape(-1,seq,23)).type(torch.Tensor)   
y_test=torch.from_numpy(y_test.reshape(-1,1)).type(torch.Tensor)   


# In[285]:


print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[286]:


train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=800,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=1,shuffle=False)


# In[287]:


input_dim=23
hidden_dim=300
num_layers=2
output_dim=1


# In[288]:


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

      


# In[289]:


model=LSTM(input_dim=input_dim,hidden_dim=hidden_dim,num_layers=num_layers,output_dim=output_dim)
optimiser = torch.optim.Adam(model.parameters(), lr=0.01) # 使用Adam优化算法
loss_fn = torch.nn.MSELoss(size_average=True) 


# In[290]:


num_epochs = 180

# 打印模型结构
print(model)


# In[291]:


for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# In[292]:


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


# In[293]:


model_name='meanstd_norml_lstm_seq_15_batch_800_epoch_300_lr0.01_in15*23_out_1_TPDRMinIn5Days.pth'
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


# In[294]:


plt.plot(train_losses[10:150])


# In[25]:


model=torch.load('./LSTMmodels/meanstd_norml_lstm_seq_15_batch_700_epoch_180_lr0.008.pth')


# In[295]:


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


# In[296]:


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
        


# In[297]:


print(len(test_loader))
print(preds[0])
#preds_array=preds[0].to('cpu').detach().numpy().copy()
#preds_list=preds.tolist()
#preds_list[0]
new_pred=[]
for i in range(len(preds)):
    new_pred.append(preds[i])
new_pred
len(new_pred)
new_pred[-10:]


# In[298]:


# prediction data needs denormalize
a=col_mean['TargetPortDailyRtn']
b=col_std['TargetPortDailyRtn']
pred_denorml=[]

for i in range(len(new_pred)):
    denor_pred=new_pred[i]*b+a
    pred_denorml.append(denor_pred)
pred_denorml[-30:]


# In[309]:


ans_real=y_test.tolist()
ans_list=[]
for i in range(seq-1,len(df.iloc[:,0])-5):
    ans_list.append(df.iloc[i+1:i+6,0].min())
ans_true=ans_list[-1625:]
len(ans_true)


# In[300]:


ans_3=true_ans
ans_3[-17:]


# In[301]:


ans_4=ans_array[-len(pred_denorml):]
ans_4


# In[ ]:





# In[312]:


len(pred_denorml)==len(ans_true)


# In[313]:


fig = plt.figure(figsize=(20, 4))

stt=0
#end=600
end=len(new_pred)
plt.scatter(range(len(ans_true[stt:end])),ans_true[stt:end],color='forestgreen',alpha=0.3,s=2,label='TPDR real val min in 5days')

plt.scatter(range(len(pred_denorml[stt:end])),pred_denorml[stt:end],color='red',alpha=0.3,s=1,label='TPDR pred val')

plt.axhline(y=0, color='plum', linestyle='-',linewidth=1, alpha=0.2)
leg=plt.legend(loc='lower left')
plt.title('TPDR MIN in next 5days, real val and pred val')
plt.show()


# In[98]:


# plot, if you plot with the average of 

fig = plt.figure(figsize=(50, 4))
stt=400
end=600
#end=len(preds_list)

x1=range(stt,end)

size1=[15 if value > 0 else 1 for value in pred_denorml]
size2=[15 if value > 0 else 1 for value in ans_3]
for x, y in zip(x1,  pred_denorml[stt:end]):
    if y>0:
        plt.scatter(x, y, color='red',alpha=0.5,s=8)
    else:
        plt.scatter(x, y, color='red',alpha=0.5,s=1)

for x, y in zip(x1,  ans_3[stt:end]):
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


# In[73]:


preds_list=pred_denorml
fig = plt.figure(figsize=(20, 4))
plt.scatter(range(len(preds_list)),preds_list,color='red',alpha=0.3)
plt.scatter(range(len(ans_4)),ans_4,color='blue',alpha=0.3)
plt.axhline(y=0, color='green', linestyle='-',linewidth=1, alpha=0.2)
plt.show()


# In[90]:


a=pd.DataFrame([5,3,6,8,2,7,3,5,9])
a[2:6].product()


# In[ ]:




