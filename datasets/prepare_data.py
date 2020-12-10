import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from torch.autograd import Variable

import seaborn as sns

import h5py
#import pywt

# Loads all xml files in the categorical directory
def get_data(dir):
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()
    df_test = pd.DataFrame()
    for file in os.listdir(dir):
        print(file)
        df = pd.read_excel(dir + '/' + file, usecols=[1,2,3,4,5,6,7,8])
        
        print('total:', len(df), '60%: ', len(df)*.6, '20%: ',len(df)*.2)
        
        train = int(len(df)*.6)
        val = int(train + len(df)*.2)
        test = int(val + len(df)*.2)
        df_train = df_train.append(df.iloc[:train])
        df_val = df_val.append(df.iloc[train:val])
        df_test = df_test.append(df.iloc[val:test])

    return df_train, df_val, df_test

if wavelet:
    #train_f_cA, train_f_cD = pywt.dwt(train_f['ODBA'], 'db2')
    logf = np.vstack(pywt.dwt(train_f['ODBA'], 'haar')).T
    logr = np.vstack(pywt.dwt(train_r['ODBA'], 'haar')).T
    logs = np.vstack(pywt.dwt(train_s['ODBA'], 'haar')).T
    logn = np.vstack(pywt.dwt(train_n['ODBA'], 'haar')).T
    
    logvf = np.vstack(pywt.dwt(val_f['ODBA'], 'haar')).T
    logvr = np.vstack(pywt.dwt(val_r['ODBA'], 'haar')).T
    logvs = np.vstack(pywt.dwt(val_s['ODBA'], 'haar')).T
    logvn = np.vstack(pywt.dwt(val_n['ODBA'], 'haar')).T
    
    logtf = np.vstack(pywt.dwt(test_f['ODBA'], 'haar')).T
    logtr = np.vstack(pywt.dwt(test_r['ODBA'], 'haar')).T
    logts = np.vstack(pywt.dwt(test_s['ODBA'], 'haar')).T
    logtn = np.vstack(pywt.dwt(test_n['ODBA'], 'haar')).T
    
    plt.hist(train_f['ODBA'], bins='auto')
    plt.show()
    
    plt.hist(pywt.dwt(train_f['ODBA'], 'haar')[0], bins='auto')
    plt.show()
    
    plt.hist(pywt.dwt(train_f['ODBA'], 'haar')[1], bins='auto')
    plt.show()
    
    print(train_f['ODBA'])
    print(pywt.dwt(train_f['ODBA'], 'haar')[0])

elif expDim is '1d':
    logf= np.log10(train_f['ODBA'])
    logr= np.log10(train_r['ODBA'])
    logs= np.log10(train_s['ODBA'])
    logn= np.log10(train_n['ODBA'])

    logvf= np.log10(val_f['ODBA'])
    logvr= np.log10(val_r['ODBA'])
    logvs= np.log10(val_s['ODBA'])
    logvn= np.log10(val_n['ODBA'])

    logtf= np.log10(test_f['ODBA'])
    logtr= np.log10(test_r['ODBA'])
    logts= np.log10(test_s['ODBA'])
    logtn= np.log10(test_n['ODBA'])
    
    plt.hist(train_f['ODBA'], bins='auto')
    plt.show()
    
    plt.hist(logf, bins='auto')
    plt.show()

elif expDim is '2d':
    logf= train_f[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logr= train_r[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logs= train_s[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logn= train_n[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()

    logvf= val_f[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logvr= val_r[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logvs= val_s[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logvn= val_n[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()

    logtf= test_f[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logtr= test_r[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logts= test_s[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()
    logtn= test_n[['X_static', 'Y_static', 'Z_static', 'X_dynamic', 'Y_dynamic', 'Z_dynamic']].dropna().to_numpy()

meanf = np.mean(logf)
stdf = np.std(logf)

meanr = np.mean(logr)
stdr = np.std(logr)

means = np.mean(logs)
stds = np.std(logs)

meann = np.mean(logn)
stdn = np.std(logn)

mean_ = (meanf+meanr+means+meann)/4
std_ = (stdf+stdr+stds+stdn)/4

train_feed_X = (logf-mean_)/std_
train_rest_X = (logr-mean_)/std_
train_swim_X = (logs-mean_)/std_
train_ndm_X = (logn-mean_)/std_

val_feed_X = (logvf-mean_)/std_
val_rest_X = (logvr-mean_)/std_
val_swim_X = (logvs-mean_)/std_
val_ndm_X = (logvn-mean_)/std_

test_feed_X = (logtf-mean_)/std_
test_rest_X = (logtr-mean_)/std_
test_swim_X = (logts-mean_)/std_
test_ndm_X = (logtn-mean_)/std_

if expDim is '2d':
    train_feed_X_nx = train_feed_X
    train_swim_X_nx = train_swim_X
    train_rest_X_nx = train_rest_X
    train_ndm_X_nx = train_ndm_X

    val_feed_X_nx = val_feed_X
    val_swim_X_nx = val_swim_X
    val_rest_X_nx = val_rest_X
    val_ndm_X_nx = val_ndm_X

    test_feed_X_nx = test_feed_X
    test_swim_X_nx = test_swim_X
    test_rest_X_nx = test_rest_X
    test_ndm_X_nx = test_ndm_X

else:
    train_feed_X_nx = np.expand_dims(train_feed_X, axis=1)
    train_swim_X_nx = np.expand_dims(train_swim_X, axis=1)
    train_rest_X_nx = np.expand_dims(train_rest_X, axis=1)
    train_ndm_X_nx = np.expand_dims(train_ndm_X, axis=1)

    val_feed_X_nx = np.expand_dims(val_feed_X, axis=1)
    val_swim_X_nx = np.expand_dims(val_swim_X, axis=1)
    val_rest_X_nx = np.expand_dims(val_rest_X, axis=1)
    val_ndm_X_nx = np.expand_dims(val_ndm_X, axis=1)

    test_feed_X_nx = np.expand_dims(test_feed_X, axis=1)
    test_swim_X_nx = np.expand_dims(test_swim_X, axis=1)
    test_rest_X_nx = np.expand_dims(test_rest_X, axis=1)
    test_ndm_X_nx = np.expand_dims(test_ndm_X, axis=1)

