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

test_X, test_Y = load_data(data_folder, 'test', testSplit=testSplit)

########################### expand dims ###########################

cnn = (modelType == 'cnn') or (modelType == 'rcnn')

if cnn and expDim == '2d':
    test_X  = np.expand_dims(test_X, axis=1)

########################### load up dataset ###########################

test_dataset  = SharkBehaviorDataset(test_X, labels=test_Y, train=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

model, _, _ = get_model(config)

########################### test ###########################

if modelType == 'cnn':
    if expDim == '1d':
        if archType == 'VGG1':
            model = SharkVGG1(1)
        elif archType == 'VGG2':
            model = SharkVGG2(1)
        elif archType == 'Inception':
            model = Sharkception(1)
    
    elif expDim == '2d':
        if archType == 'VGG1':
            model = Shark2dVGG1(1)
        elif archType == 'VGG2':
            model = Shark2dVGG2(1)
        elif archType == 'Inception':
            model = Sharkception2d(1)
            
elif modelType == 'rnn':
    hidden_size = 128
    dat_size = 50
    out_size = 4
    
    if archType == 'LSTM':
        model = SharkLSTM(dat_size, hidden_size, out_size)
    elif archType == 'GRU':
        model = SharkGRU(dat_size, hidden_size, out_size)
        
elif modelType == 'rcnn':
    
    if expDim == '1d':
        model = SharkRCNN(1, 4)
    
    elif expDim == '2d':
        model = SharkRCNN2d(1, 4)

model.cuda()

state_dict = torch.load('./'+ folder_ + '/_acc.pth')
# state_dict = torch.load('exp/'+ folder_ +'/_loss.pth')
# state_dict = torch.load('exp/'+ folder_ +'/_last.pth')
# state_dict = torch.load('exp/'+ folder_ +'/save_1000.pth') #0,200...
model.load_state_dict(state_dict)
model.eval()

########################### inference ###########################

preds = []
ys = []
probs = np.empty((test_X.shape[0],4))
correct = 0

with torch.no_grad():
    model.eval
    count = 0
#     preds = []
#     ys = []
    val_running_loss = 0.0
    for images, labels in test_loader:
#         tots +=1
        labels = labels.squeeze()
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(images)
        prob = nn.Softmax()(outputs).cpu().numpy()
        
        _, pred = torch.max(outputs,1)
        ys = np.concatenate([ys, labels.cpu().numpy()])
        preds = np.concatenate([preds, pred.cpu().numpy()])
        
        #probs.append(prob.tolist())
        #print(prob.shape)
        probs = np.vstack([probs, prob])

        
########################### acc ###########################

# Accuracy --8000
print('Accuracy: ', (ys == preds).sum() / ys.shape[0])     

########################### confusion matrix ###########################

cm = confusion_matrix(ys, preds, [0,1,2,3])
print(cm)

########################### classification ###########################

print(classification_report(ys, preds, digits=4))

########################### ROC AUC ###########################

#print(probs)
probs1 = probs[test_X.shape[0]:]

print(test_X.shape[0])
print(probs1.shape)

print(metrics.roc_auc_score(ys, probs1, multi_class='ovo', labels=[0,1,2,3]))

########################### write results file ###########################

with open(os.path.join(results_folder, results_file), 'w+') as f:
    f.write("Test accuracy\n")
    f.write("----------------------------------\n")
    f.write(str((ys == preds).sum() / ys.shape[0]))
    f.write('\n\n')
    f.write("Confusion matrix\n")
    f.write("----------------------------------\n")
    f.write(str(confusion_matrix(ys, preds, [0,1,2,3])))
    f.write('\n\n')
    f.write("Classification report\n")
    f.write("----------------------------------\n")
    f.write(str(classification_report(ys, preds, digits=4)))
    f.write('\n\n')
    f.write("AUC ROC\n")
    f.write("----------------------------------\n")
    f.write(str(metrics.roc_auc_score(ys, probs1, multi_class='ovo', labels=[0,1,2,3])))
    