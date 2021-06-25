#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


# In[2]:


import torchvision.transforms as transforms
torch.set_printoptions(linewidth=120)#Display options for output
torch.set_grad_enabled(True)#Already on by default


# In[3]:


print(torch.__version__)


# In[4]:


print(torchvision.__version__)


# In[5]:


def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# In[6]:


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=60)
        self.out = nn.Linear(in_features=60,out_features=10)
        
    def forward(self,t):
        
        t=t
        t=F.relu(self.conv1(t))
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        t=F.relu(self.conv2(t))
        t=F.max_pool2d(t,kernel_size=2,stride=2)
        t=t.reshape(-1,12*4*4)
        t=F.relu(self.fc1(t))
        t=F.relu(self.fc2(t))
        t = self.out(t)
        return t


# In[7]:


train_set = torchvision.datasets.FashionMNIST(root='./data',train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))


# In[8]:


network = Network()

train_loader = torch.utils.data.DataLoader(train_set,batch_size=100)


# In[9]:


optimizer = optim.Adam(network.parameters(),lr=0.01)


# In[10]:


for epoch in range(0,10):
    total_loss=0
    total_correct=0
    for batch in train_loader: #Get batch
        images,labels = batch
        preds = network(images) #Pass batch
        loss = F.cross_entropy(preds,labels)#Calculate loss
        optimizer.zero_grad()
        loss.backward()#Calculates gradients
        optimizer.step()#Updates weights
        total_loss+=loss.item()
        total_correct+=get_num_correct(preds,labels)
    print("epoch:",epoch,"total_correct:",total_correct,"loss:",total_loss)
        


# In[11]:


total_correct/len(train_set)


# In[ ]:




