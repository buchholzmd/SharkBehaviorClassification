import os
import yaml
import pywt
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from utils import *
from datasets.SharkBehaviorDataset import SharkBehaviorDataset

########################### model params ###########################
with open('./configs/config.yml', 'r') as stream:
    config = yaml.safe_load(stream)

expDim     = config['DATA_DIM']
wavelet    = config['WAVELET']
archType   = config['ARCH_TYPE']
testSplit  = config['TEST_SPLIT']
modelType  = config['MODEL_TYPE']
batch_size = config['BATCH_SIZE']

folder_, results_folder, results_file = get_exp_paths(config)
    
########################### save/load ###########################

data_folder = os.path.join('./datasets/data', expDim)

train_X, train_Y = load_data(data_folder, 'train')
val_X, val_Y = load_data(data_folder, 'val')

########################### print folder ###########################

print(data_folder)

########################### expand dims ###########################

cnn = (modelType == 'cnn') or (modelType == 'rcnn')

if cnn and expDim == '2d':
    train_X = np.expand_dims(train_X, axis=1)
    val_X   = np.expand_dims(val_X, axis=1)

########################### load up dataset ###########################

train_dataset = SharkBehaviorDataset(train_X, labels=train_Y, train=True)
val_dataset   = SharkBehaviorDataset(val_X, labels=val_Y, train=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True)

model, optimizer, sched = get_model(config)
criterion = nn.CrossEntropyLoss()

########################### train ###########################

train(model, train_loader, {})

'''
mean_train_losses = []
mean_val_losses = []

mean_train_acc = []
mean_val_acc = []
minLoss = 99999
maxValacc = -99999
for epoch in range(500):
    print('EPOCH: ',epoch+1)
    train_acc = []
    val_acc = []
    
    running_loss = 0.0
    
    model.train()
    count = 0
    for images, labels in train_loader:
        labels = labels.squeeze()
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        
        train_acc.append(accuracy(outputs, labels))
        
        loss.backward()
        optimizer.step()        
        
        running_loss += loss.item()
        count +=1
        
    sched.step()
    print('Training loss:.......', running_loss/count)
    mean_train_losses.append(running_loss/count)
        
    model.eval()
    count = 0
    val_running_loss = 0.0
    for images, labels in val_loader:
        labels = labels.squeeze()
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        accuracy(outputs, labels)
        
        
        loss = criterion(outputs, labels)

        val_acc.append(accuracy(outputs, labels))
        val_running_loss += loss.item()
        count +=1

    mean_val_loss = val_running_loss/count
    print('Validation loss:.....', mean_val_loss)
    
    print('Training accuracy:...', np.mean(train_acc))
    print('Validation accuracy..', np.mean(val_acc))
    
    mean_val_losses.append(mean_val_loss)
    
    mean_train_acc.append(np.mean(train_acc))
    
    val_acc_ = np.mean(val_acc)
    mean_val_acc.append(val_acc_)
    
    if mean_val_loss < minLoss:
        torch.save(model.state_dict(), './'+folder_+'/_loss.pth' )
        print(f'NEW BEST LOSS_: {mean_val_loss} ........old best:{minLoss}')
        minLoss = mean_val_loss
        print('')
        
    if val_acc_ > maxValacc:
        torch.save(model.state_dict(), './'+folder_+'/_acc.pth' )
        print(f'NEW BEST ACC_: {val_acc_} ........old best:{maxValacc}')
        maxValacc = val_acc_
        
    if epoch%500 == 0 :
        torch.save(model.state_dict(), './'+folder_+'/save_'+str(epoch)+'.pth' )
        print(f'DIV 200: Val_acc: {val_acc_} ..Val_loss:{mean_val_loss}')
        
    torch.save(model.state_dict(), './'+folder_+'/_last.pth' )
'''

########################### disp loss/acc ###########################

train_acc_series = pd.Series(mean_train_acc)
val_acc_series = pd.Series(mean_val_acc)
train_acc_series.plot(label="train")
val_acc_series.plot(label="validation")
plt.legend()

train_acc_series = pd.Series(mean_train_losses)
val_acc_series = pd.Series(mean_val_losses)
train_acc_series.plot(label="train")
val_acc_series.plot(label="validation")
plt.legend()
 