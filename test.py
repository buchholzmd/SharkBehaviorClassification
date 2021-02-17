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

expDim      = config['DATA_DIM']
wavelet     = config['WAVELET']
archType    = config['ARCH_TYPE']
testSplit   = config['TEST_SPLIT']
modelType   = config['MODEL_TYPE']
batch_size  = config['BATCH_SIZE']
test_path   = config['TEST_PATH']
attn        = config['ATTENTION'] == 'attn'
seq_len     = config['SEQ_LEN']
num_classes = config['NUM_CLASSES']

paths_dict = get_exp_paths(config)

folder_ = paths_dict['folder']
results_folder = paths_dict['results_dir']
results_file = paths_dict['results_fn']
split_path = paths_dict['split_path']
    
########################### save/load ###########################

test_X, test_Y = load_data(split_path, 'test', testSplit=testSplit)

########################### expand dims ###########################

cnn = (modelType == 'cnn') or (modelType == 'rcnn')

if cnn :
    if expDim == '2d':
        test_X = np.expand_dims(test_X, axis=1)
    else:
        test_X = np.transpose(test_X, (0,2,1))

########################### load up dataset ###########################

test_dataset  = SharkBehaviorDataset(test_X, labels=test_Y, train=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model, _, _ = get_model(config, test=True)

if modelType == 'svdkl':
    kernel_learning = True
else:
    kernel_learning = False

########################### test ###########################

model_path = os.path.join(folder_,test_path)
print("Loading model from path: ", model_path)

if kernel_learning:
    likelihood = model['likelihood']
    model = model['model']

    # model = model.cpu()
    # likelihood = likelihood.cpu()

    print(folder_)
    state_dict = torch.load(model_path)
    # state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    likelihood.load_state_dict(state_dict['likelihood'])

    model.eval()
    likelihood.eval()

else:
    print(folder_)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()

########################### inference ###########################

ys = []
preds = []
probs = []

with torch.no_grad(), gpytorch.settings.num_likelihood_samples(32) if kernel_learning else suppress():
    count = 0
    for sequences, labels in test_loader:
        labels = labels.squeeze()
        
        outputs = model(sequences)

        if kernel_learning and attn:
            outputs, attn_weights = model(sequences)
        else:
            outputs = model(sequences)

        if kernel_learning:
            batch_size = sequences.size(0)
            # This gives us 32 samples from the predictive distribution
            # Take the mean over all samples
            outputs = likelihood(outputs).probs.mean(0)

            if attn:
                outputs = outputs.reshape((batch_size, seq_len, num_classes))
                prob = torch.bmm(attn_weights, outputs).squeeze().cpu()

            else:
                prob = outputs.cpu()

        else:
            prob = torch.softmax(outputs, dim=1).cpu()
        
        pred = prob.argmax(-1)

        ys = np.concatenate([ys, labels.numpy()])
        preds = np.concatenate([preds, pred.numpy()])
        probs.append(prob.numpy())

probs = np.concatenate(probs)
        
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
print(probs.shape)

print(metrics.roc_auc_score(ys, probs, multi_class='ovo', labels=[0,1,2,3]))

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
        f.write(str(metrics.roc_auc_score(ys, probs, multi_class='ovo', labels=[0,1,2,3])))
    