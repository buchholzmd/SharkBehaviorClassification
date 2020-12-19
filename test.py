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

parser = argparse.ArgumentParser()

parser.add_argument('--write',
                    action='store_true',
                    help='Write results to file')

args = parser.parse_args()

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
    test_X = np.expand_dims(test_X, axis=1)

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
    hidden_size = config['HIDDEN_DIM']
    if expDim == '1d':
        dat_size = 1
    else:
        dat_size = 6
        
    out_size = config['NUM_CLASSES']
    num_layers = config['NUM_LAYERS']
    fc_dim = config['FC_DIM']

    if archType == 'LSTM':
        model = SharkLSTM(dat_size,
                          hidden_size, 
                          out_size, 
                          num_layers=num_layers,
                          fc_dim=fc_dim)

    elif archType == 'GRU':
        model = SharkGRU(dat_size, 
                         hidden_size, 
                         out_size, 
                         num_layers=num_layers,
                         fc_dim=fc_dim)
        
elif modelType == 'rcnn':
    model = SharkRCNN(3)

model.cuda()

print(folder_)
state_dict = torch.load('./'+ folder_ + '/_acc.pth')
# state_dict = torch.load('./'+ folder_ + '/_loss.pth')
# state_dict = torch.load('./'+ folder_ + '/_last.pth')
model.load_state_dict(state_dict)
model.eval()

########################### inference ###########################

ys = []
preds = []
probs = np.empty((test_X.shape[0],4))

with torch.no_grad():
    model.eval
    
    count = 0
    for sequences, labels in test_loader:
        labels = labels.squeeze()
        sequences = Variable(sequences.cuda())
        labels = Variable(labels.cuda())
        
        outputs = model(sequences)
        prob = nn.Softmax()(outputs).cpu().numpy()
        
        _, pred = torch.max(outputs,1)
        ys = np.concatenate([ys, labels.cpu().numpy()])
        preds = np.concatenate([preds, pred.cpu().numpy()])
        
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

if args.write:
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
    