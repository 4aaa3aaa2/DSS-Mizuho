#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


# In[3]:


span=0


# In[4]:


df=pd.read_csv('TPDR +1 Day, high corr indexes normalized, nomiss.csv')
df=df.drop(df.columns[[0]],axis=1)
df


# In[5]:


'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
span=50
df=pd.read_csv(f'features_new_{span}.csv')
df.index=df["Dates"]
df=df.drop(df.columns[[0,1,2,6,10,11]],axis=1)#drop the meaningless col, which is the row index
#df=df.drop(['Dates'],axis=1)
df.iloc[:, 10:] = scaler.fit_transform(df.iloc[:, 10:])
#df=df.drop(df.columns[[0]],axis=1)
df=df.drop(df.columns[[5]],axis=1)
df'''


# In[6]:


df.shape


# In[7]:


df=df.astype(np.float64)


# In[8]:


class Hybrid(nn.Module):
    def __init__(self, lstm_input_dim,lstm_hidden_dim, lstm_layers,output_dim) :
        super(Hybrid,self).__init__()
        self.lstm_hidden_dim=lstm_hidden_dim
        self.lstm_layers=lstm_layers
        self.lstm=nn.LSTM(lstm_input_dim,lstm_hidden_dim, lstm_layers, batch_first=True)
        self.fc=nn.Linear(lstm_hidden_dim,output_dim)
    def forward(self,x):
        size, seq_len, _ = x.size()
        x,_ = self.lstm(x)
        x = x.contiguous().view(size,-1)
        x = self.fc(x)
        return x


# In[9]:


input_dim=23
hidden_dim=256
num_layers=2
output_dim=1


# In[10]:


def split_data(features,seq):
    features_y=features.iloc[:,0]
    features_x=features.drop(features.columns[[0]],axis=1)
    data_x=[]
    data_y=[]
    for i in range(len(features_x)-seq):
        data_x.append(features_x[i:i+seq])
        data_y.append(features_y[i:i+seq])
        
    #split_position=4631-span-seq+3
    split_position=5000-seq+3
    x_train=data_x[:split_position]
    x_test=data_x[split_position:]
    y_train=data_y[:split_position]
    y_test=data_y[split_position:]

    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)

    return x_train, y_train, x_test,y_test


# In[31]:


seq=5
x_train, y_train, x_test,y_test=split_data(df,seq)


# In[32]:


x_train.shape


# In[33]:


print(y_train[0])


# In[34]:


x_train=torch.from_numpy(x_train.reshape(-1,seq,23)).type(torch.Tensor)   
y_train=torch.from_numpy(y_train.reshape(-1,seq)).type(torch.Tensor)   
x_test=torch.from_numpy(x_test.reshape(-1,seq,23)).type(torch.Tensor)   
y_test=torch.from_numpy(y_test.reshape(-1,seq)).type(torch.Tensor)   


# In[35]:


print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)


# In[36]:


train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)
train_loader = torch.utils.data.DataLoader(dataset=train,batch_size=16,shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test,batch_size=16,shuffle=False)


# In[37]:


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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).float()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).float()
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

        


# In[38]:


model = LSTM( input_dim=input_dim,hidden_dim=hidden_dim, num_layers=num_layers,output_dim=output_dim)
# 定义优化器和损失函数
optimiser = torch.optim.Adam(model.parameters(), lr=0.013) # 使用Adam优化算法
loss_fn = torch.nn.MSELoss(size_average=True) 


# In[39]:


num_epochs = 130

# 打印模型结构
print(model)


# In[40]:


for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())


# In[41]:


train_losses = []

early_stopping_counter=0
patience=10

for epoch in range(num_epochs):
    model.train()
    y_train_pred=model(x_train)
    loss=loss_fn(y_train_pred,y_train)
    print("Epoch ", epoch, "MSE: ", loss.item())
    train_losses.append(loss.item())
    optimiser.zero_grad()

    # Backward pass
    loss.backward()
    # Update parameters
    optimiser.step()


# In[25]:


plt.plot(train_losses)


# In[26]:


torch.save(model, 'lstm_model.pth')


# In[21]:


model=torch.load('lstm_model.pth')


# In[22]:


model.eval()
loss_sum=0
correct=0
cnt_signal=0
catch_signal=0
preds=[]
ans=[]
with torch.no_grad():
    y_test_pred=model(x_test)
    loss=loss_fn(y_test_pred,y_test)
    preds.append(y_test_pred)
    ans.append(y_test)

print(preds) 
print(f'len preds :{len(preds[0])}')
print('__________')
print(ans)
print(f"len ans:{len(ans[0][0])}")
    
    
    


# In[23]:


preds_array=preds[0].to('cpu').detach().numpy().copy()
preds_list=preds_array.tolist()
preds_list


# In[24]:


ans_array=ans[0].to('cpu').detach().numpy().copy()
ans_array


# In[25]:


ans_1=[]#true value of each day in valid dataset
for i in range(len(ans_array)):
    ans_1.append(ans_array[i][0])


# In[26]:


ans_2=[]#15 day mean value in valid dataset
for i in range(len(ans_array)):
    ans_2.append(ans_array[i].mean())


# In[27]:


ans_3=[]# (maximun+minimun)*0.5 for the maximun and minimun in 15 days
for i in range(len(ans_array)):
    mx=ans_array[i].max()
    mi=ans_array[i].min()
    ans_3.append((mx+mi)*0.5)


# In[43]:


ans_4=[]
for i in range(len(ans_array)):
    ans_4.append(np.median(ans_array[i]))


# In[44]:


ans_4


# In[45]:


plt.plot(preds_list,color='red',alpha=0.5,label='predticted valid value')
plt.plot(ans_4,color='blue',alpha=0.5,label='true valid value')
leg=plt.legend(loc='lower left')
plt.title('predicted and the median value in 15 days by LSTM model')
plt.show()


# In[ ]:




