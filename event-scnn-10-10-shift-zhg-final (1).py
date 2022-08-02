#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-

from __future__ import print_function
from spiking_model6c_100_10_10event_nopool_noreset_countspike import*
# import torchvision
# import torchvision.transforms as transforms
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import IPython
import sys
import h5py
import numpy as np
# import tensorflow as tf
import scipy.io as sio
import time
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
import torch.utils.data as Data
from self_pytorchtools import EarlyStopping
import operator
from functools import reduce
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[2]:


train_spike=sio.loadmat('../data3event/train_feature1800baseline_zhlei.mat')
train_spike=train_spike['TrainPtns1']
train_label=sio.loadmat('../data3event/train_label1800baseline_zhlei.mat')
train_label=train_label['Class1']
print(train_label.shape)
train_label=reduce(operator.add, train_label.T)
print(train_spike.shape)
train_label.shape
train_spike.shape

test_spike=train_spike
test_label=train_label
# test_spike=np.concatenate((train_spike[30:300],train_spike[330:600],train_spike[630:900],train_spike[930:1200],train_spike[1230:1500],train_spike[1530:1800]),axis=0)
# test_label=np.concatenate((train_label[30:300],train_label[330:600],train_label[630:900],train_label[930:1200],train_label[1230:1500],train_label[1530:1800]),axis=0)

# test_spike=np.concatenate((train_spike[270:300],train_spike[570:600],train_spike[870:900],train_spike[1170:1200],train_spike[1470:1500],train_spike[1770:1800]),axis=0)
# test_label=np.concatenate((train_label[270:300],train_label[570:600],train_label[870:900],train_label[1170:1200],train_label[1470:1500],train_label[1770:1800]),axis=0)
j=10
batch_size1=120
batch_size2=18
batch_size3=162
val_spike2=np.zeros([3*j,100,10,10])
val_label2=np.zeros([3*j])
train_spike2=np.zeros([27*j,100,10,10])
train_label2=np.zeros([27*j])
for i in range(6):
    val_spike3=train_spike[i*300+30*j-3*j:i*300+30*j]
    val_spike2=np.concatenate((val_spike2,val_spike3))

    val_label3=train_label[i*300+30*j-3*j:i*300+30*j]
    val_label2=np.concatenate((val_label2,val_label3))

    train_spike3=train_spike[i*300+0:i*300+30*j-3*j]
    train_spike2=np.concatenate((train_spike2,train_spike3))

    train_label3=train_label[i*300+0:i*300+30*j-3*j]
    train_label2=np.concatenate((train_label2,train_label3))
val_spike=val_spike2[3*j:]      
val_label=val_label2[3*j:]
train_spike=train_spike2[27*j:]      
train_label=train_label2[27*j:]

print(train_spike.shape)
print(train_label.shape)
# print(test_spike.shape)
# print(test_label.shape)
print(val_spike.shape)
print(val_label.shape)

test_spike=torch.tensor(test_spike)
test_label=torch.tensor(test_label)
print(test_spike.shape)
test_dataset = Data.TensorDataset(test_spike,test_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size1, shuffle=True, num_workers=0,drop_last=False)
test_dataset

val_spike=torch.tensor(val_spike)
val_label=torch.tensor(val_label)
print(val_spike.shape)
val_dataset = Data.TensorDataset(val_spike,val_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size2, shuffle=True, num_workers=0,drop_last=False)

train_spike=torch.tensor(train_spike)
train_label=torch.tensor(train_label)
print(train_spike.shape)
train_dataset = Data.TensorDataset(train_spike,train_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size3, shuffle=True, num_workers=0,drop_last=False)
train_dataset


# In[3]:


best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
snn = SCNN()
snn.to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(snn.parameters(), lr=learning_rate)
acc_record_val = list([])
loss_val_record = list([])

patience = 20   # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
early_stopping = EarlyStopping(patience, verbose=True) # 关于 EarlyStopping 的代码可先看博客后面的内容


# In[ ]:


num_epochs=300
for epoch in range(num_epochs):#70
    optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
    running_loss = 0
    start_time = time.time()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):#一次训练：分60个批次
        running_loss = 0
        snn.zero_grad()
        optimizer.zero_grad()
        images = images.float().to(device)
        outputs,spikes,post_spikes = snn(images)#############
        labels_ = torch.zeros(batch_size3, 6).scatter_(1, labels.long().view(-1, 1), 1).to(device)#[10,10]#############
        loss = criterion(outputs, labels_)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += float(labels.size(0))#
        correct += float(predicted.eq(labels.long().to(device)).sum().item())
    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size3,running_loss ))
    acc = 100. * float(correct) / float(total)
    print(' Acc: %.5f' % acc)
    print('Time elasped:', time.time()-start_time)###训练10个批次的时间
    acc_record.append(acc)
    loss_train_record.append(running_loss)
        
#验证集    
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
        for batch_idx, (inputs, targets) in enumerate(val_loader):##一次测试：分30个批次
            inputs = inputs.to(device)#################################.short()
            optimizer.zero_grad()
            outputs,spikes,post_spikes = snn(inputs)
            labels_ = torch.zeros(batch_size2, 6).scatter_(1, targets.view(-1, 1).long(), 1).to(device)####################onehot
            loss = criterion(outputs, labels_)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += float(targets.size(0))#
            correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        loss_val_record.append(val_loss)
        val_Acc=100 * correct / total
        acc_record_val.append(val_Acc)
        print('val_Acc: %.3f,val_loss: %.5f' % (val_Acc,val_loss),'\n')
    early_stopping(val_loss, snn)
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break


# In[ ]:


# summarize history for accuracy
epochs = range(len(acc_record_val)) 
plt.figure()
plt.plot(epochs, acc_record, label='Training acc')
plt.plot(epochs, acc_record_val, label='val acc')
plt.title('Training and val acc ')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure()
plt.plot(epochs, loss_train_record, label='Training loss')
plt.plot(epochs, loss_val_record, label='val loss')
plt.title('Training and val loss ')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# print('Iters:', epoch+1)
# print('Saving..','\n\n')
# state = {
#     'net': snn.state_dict(),
#     'acc': acc,
#     'epoch': epoch,
#     'acc_record': acc_record,
# }
# if not os.path.isdir('checkpoint'):
#     os.mkdir('checkpoint')
# torch.save(state, './checkpoint/' + 'model6baseline_lt' + '.t7')


# In[ ]:


correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
start_time=time.time()
sum_spikes=0
sum_post_spikes=0
with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
    for batch_idx, (inputs, targets) in enumerate(test_loader):##一次测试：分30个批次
        inputs = inputs.to(device)#################################.short()
        optimizer.zero_grad()
        outputs ,spikes,post_spikes= snn(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))#
        correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        if batch_idx %10 ==0:#########################？？？？？？
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)
        sum_spikes+=spikes
        sum_post_spikes+=post_spikes
print('sum_spikes',sum_spikes)
print('sum_post_spikes',sum_post_spikes)
print('Test Accuracy of the model on the %d test images: %.3f' % (len(test_dataset),(100 * correct / total)),'\n')
print('Time elasped:', time.time()-start_time)###训练10个批次的时间


# In[ ]:


train_spike=sio.loadmat('../data3event/train_feature1800zuoshang_zhlei.mat')
train_spike=train_spike['TrainPtns1']
train_label=sio.loadmat('../data3event/train_label1800zuoshang_zhlei.mat')
train_label=train_label['Class1']
print(train_label.shape)
train_label=reduce(operator.add, train_label.T)
print(train_spike.shape)
train_label.shape
train_spike.shape

test_spike=train_spike
test_label=train_label

# test_spike=np.concatenate((train_spike[270:300],train_spike[570:600],train_spike[870:900],train_spike[1170:1200],train_spike[1470:1500],train_spike[1770:1800]),axis=0)
# test_label=np.concatenate((train_label[270:300],train_label[570:600],train_label[870:900],train_label[1170:1200],train_label[1470:1500],train_label[1770:1800]),axis=0)
test_spike=torch.tensor(test_spike)
test_label=torch.tensor(test_label)
print(test_spike.shape)
test_dataset = Data.TensorDataset(test_spike,test_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size1, shuffle=True, num_workers=0,drop_last=False)
test_dataset

correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
start_time=time.time()
sum_spikes=0
sum_post_spikes=0
with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
    for batch_idx, (inputs, targets) in enumerate(test_loader):##一次测试：分30个批次
        inputs = inputs.to(device)#################################.short()
        optimizer.zero_grad()
        outputs ,spikes,post_spikes= snn(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))#
        correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        if batch_idx %10 ==0:#########################？？？？？？
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        sum_spikes+=spikes
        sum_post_spikes+=post_spikes

print('sum_spikes',sum_spikes)
print('sum_post_spikes',sum_post_spikes)
print('Test Accuracy of the model on the %d test images: %.3f' % (len(test_dataset),(100 * correct / total)),'\n')
print('Time elasped:', time.time()-start_time)###训练10个批次的时间


# In[ ]:


train_spike=sio.loadmat('../data3event/train_feature1800zuoxia_zhlei.mat')
train_spike=train_spike['TrainPtns1']
train_label=sio.loadmat('../data3event/train_label1800zuoxia_zhlei.mat')
train_label=train_label['Class1']
print(train_label.shape)
train_label=reduce(operator.add, train_label.T)
print(train_spike.shape)
train_label.shape
train_spike.shape

test_spike=train_spike
test_label=train_label

# test_spike=np.concatenate((train_spike[270:300],train_spike[570:600],train_spike[870:900],train_spike[1170:1200],train_spike[1470:1500],train_spike[1770:1800]),axis=0)
# test_label=np.concatenate((train_label[270:300],train_label[570:600],train_label[870:900],train_label[1170:1200],train_label[1470:1500],train_label[1770:1800]),axis=0)
test_spike=torch.tensor(test_spike)
test_label=torch.tensor(test_label)
print(test_spike.shape)
test_dataset = Data.TensorDataset(test_spike,test_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size1, shuffle=True, num_workers=0,drop_last=False)
test_dataset

correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
start_time=time.time()
sum_spikes=0
sum_post_spikes=0
with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
    for batch_idx, (inputs, targets) in enumerate(test_loader):##一次测试：分30个批次
        inputs = inputs.to(device)#################################.short()
        optimizer.zero_grad()
        outputs ,spikes,post_spikes= snn(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))#
        correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        if batch_idx %10 ==0:#########################？？？？？？
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        sum_spikes+=spikes
        sum_post_spikes+=post_spikes

print('sum_spikes',sum_spikes)
print('sum_post_spikes',sum_post_spikes)
print('Test Accuracy of the model on the %d test images: %.3f' % (len(test_dataset),(100 * correct / total)),'\n')
print('Time elasped:', time.time()-start_time)###训练10个批次的时间


# In[ ]:


train_spike=sio.loadmat('../data3event/train_feature1800youshang_zhlei.mat')
train_spike=train_spike['TrainPtns1']
train_label=sio.loadmat('../data3event/train_label1800youshang_zhlei.mat')
train_label=train_label['Class1']
print(train_label.shape)
train_label=reduce(operator.add, train_label.T)
print(train_spike.shape)
train_label.shape
train_spike.shape

test_spike=train_spike
test_label=train_label

# test_spike=np.concatenate((train_spike[270:300],train_spike[570:600],train_spike[870:900],train_spike[1170:1200],train_spike[1470:1500],train_spike[1770:1800]),axis=0)
# test_label=np.concatenate((train_label[270:300],train_label[570:600],train_label[870:900],train_label[1170:1200],train_label[1470:1500],train_label[1770:1800]),axis=0)
test_spike=torch.tensor(test_spike)
test_label=torch.tensor(test_label)
print(test_spike.shape)
test_dataset = Data.TensorDataset(test_spike,test_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size1, shuffle=True, num_workers=0,drop_last=False)
test_dataset

correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
start_time=time.time()
sum_spikes=0
sum_post_spikes=0
with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
    for batch_idx, (inputs, targets) in enumerate(test_loader):##一次测试：分30个批次
        inputs = inputs.to(device)#################################.short()
        optimizer.zero_grad()
        outputs ,spikes,post_spikes= snn(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))#
        correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        if batch_idx %10 ==0:#########################？？？？？？
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        sum_spikes+=spikes
        sum_post_spikes+=post_spikes

print('sum_spikes',sum_spikes)
print('sum_post_spikes',sum_post_spikes)
print('Test Accuracy of the model on the %d test images: %.3f' % (len(test_dataset),(100 * correct / total)),'\n')
print('Time elasped:', time.time()-start_time)###训练10个批次的时间


# In[ ]:


train_spike=sio.loadmat('../data3event/train_feature1800youxia_zhlei.mat')
train_spike=train_spike['TrainPtns1']
train_label=sio.loadmat('../data3event/train_label1800youxia_zhlei.mat')
train_label=train_label['Class1']
print(train_label.shape)
train_label=reduce(operator.add, train_label.T)
print(train_spike.shape)
train_label.shape
train_spike.shape

test_spike=train_spike
test_label=train_label

# test_spike=np.concatenate((train_spike[270:300],train_spike[570:600],train_spike[870:900],train_spike[1170:1200],train_spike[1470:1500],train_spike[1770:1800]),axis=0)
# test_label=np.concatenate((train_label[270:300],train_label[570:600],train_label[870:900],train_label[1170:1200],train_label[1470:1500],train_label[1770:1800]),axis=0)
test_spike=torch.tensor(test_spike)
test_label=torch.tensor(test_label)
print(test_spike.shape)
test_dataset = Data.TensorDataset(test_spike,test_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size1, shuffle=True, num_workers=0,drop_last=False)
test_dataset

correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
start_time=time.time()
sum_spikes=0
sum_post_spikes=0
with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
    for batch_idx, (inputs, targets) in enumerate(test_loader):##一次测试：分30个批次
        inputs = inputs.to(device)#################################.short()
        optimizer.zero_grad()
        outputs ,spikes,post_spikes= snn(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))#
        correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        if batch_idx %10 ==0:#########################？？？？？？
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        sum_spikes+=spikes
        sum_post_spikes+=post_spikes

print('sum_spikes',sum_spikes)
print('sum_post_spikes',sum_post_spikes)
print('Test Accuracy of the model on the %d test images: %.3f' % (len(test_dataset),(100 * correct / total)),'\n')
print('Time elasped:', time.time()-start_time)###训练10个批次的时间


# In[ ]:


train_spike=sio.loadmat('../data3event/train_feature1800random_zhlei.mat')
train_spike=train_spike['TrainPtns1']
train_label=sio.loadmat('../data3event/train_label1800random_zhlei.mat')
train_label=train_label['Class1']
print(train_label.shape)
train_label=reduce(operator.add, train_label.T)
print(train_spike.shape)
train_label.shape
train_spike.shape

test_spike=train_spike
test_label=train_label

# test_spike=np.concatenate((train_spike[270:300],train_spike[570:600],train_spike[870:900],train_spike[1170:1200],train_spike[1470:1500],train_spike[1770:1800]),axis=0)
# test_label=np.concatenate((train_label[270:300],train_label[570:600],train_label[870:900],train_label[1170:1200],train_label[1470:1500],train_label[1770:1800]),axis=0)
test_spike=torch.tensor(test_spike)
test_label=torch.tensor(test_label)
print(test_spike.shape)
test_dataset = Data.TensorDataset(test_spike,test_label)#torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size1, shuffle=True, num_workers=0,drop_last=False)
test_dataset

correct = 0
total = 0
# optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)
start_time=time.time()
sum_spikes=0
sum_post_spikes=0
with torch.no_grad():###所以如果有不想被track的计算部分可以通过这么一个上下文管理器包裹起来。这样可以执行计算，但该计算不会在反向传播中被记录。
    for batch_idx, (inputs, targets) in enumerate(test_loader):##一次测试：分30个批次
        inputs = inputs.to(device)#################################.short()
        optimizer.zero_grad()
        outputs ,spikes,post_spikes= snn(inputs)
        _, predicted = outputs.max(1)
        total += float(targets.size(0))#
        correct += float(predicted.eq(targets.long().to(device)).sum().item())##################2
        if batch_idx %10 ==0:#########################？？？？？？
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        sum_spikes+=spikes
        sum_post_spikes+=post_spikes

print('sum_spikes',sum_spikes)
print('sum_post_spikes',sum_post_spikes)
print('Test Accuracy of the model on the %d test images: %.3f' % (len(test_dataset),(100 * correct / total)),'\n')
print('Time elasped:', time.time()-start_time)###训练10个批次的时间
