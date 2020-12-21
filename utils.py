import os
import h5py
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import Variable

from collections import Counter
from networks.rcnn import SharkRCNN
from networks.rnn import SharkLSTM, SharkGRU
from sklearn.model_selection import train_test_split
from networks.attention_rnn import SharkAttentionLSTM
from networks.cnn import SharkVGG, SharkVGG2d

def accuracy(out, labels):
    _, pred = torch.max(out,1)
    correct = (pred == labels).sum().item()
    acc = 100*correct/len(labels)
    return acc

def train(model, train_loader, val_loader, parameters, optimizer, sched, path, rnn=False):
    folder_ = path
    
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    mean_train_losses = []
    mean_val_losses = []

    mean_train_acc = []
    mean_val_acc = []
    minLoss = 99999
    maxValacc = -99999
    for epoch in range(500):
        print('EPOCH: ', epoch+1)
        train_acc = []
        val_acc = []

        running_loss = 0.0

        model.train()
        
        count = 0
        for sequences, labels in train_loader:
            labels = labels.squeeze()
            sequences = Variable(sequences.cuda())
            labels = Variable(labels.cuda())
            
            outputs = model(sequences)

            optimizer.zero_grad()
            loss = criterion(outputs, labels)

            train_acc.append(accuracy(outputs, labels))

            loss.backward()
            
            if rnn:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                
            optimizer.step()        

            running_loss += loss.item()
            count +=1
        
        if epoch % 25 == 0:
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
            
    torch.save(model.state_dict(), './'+folder_+'/_last.pth')

def get_exp_paths(config):
    split       = config['SPLIT']
    split_path  = config['SPLIT_PATH']
    expDim      = config['DATA_DIM']
    wavelet     = config['WAVELET']
    archType    = config['ARCH_TYPE']
    testSplit   = config['TEST_SPLIT']
    modelType   = config['MODEL_TYPE']
    attention   = config['ATTENTION']
    
    paths_dict = {}

    paths_dict['folder'] = os.path.join('./models', attention + modelType, expDim, archType)

    paths_dict['results_dir'] = os.path.join('./results', testSplit)

    if wavelet:
        paths_dict['results_fn'] = 'wavelet_' + archType + '_' + expDim + '_' + testSplit + '.txt'
    else:
        paths_dict['results_fn'] = attention + archType + '_' + expDim + '_' + testSplit + '.txt'

    if not os.path.exists(paths_dict['folder']):
        os.makedirs(paths_dict['folder'])
    
    print('Folder:' + str(paths_dict['folder']))

    if not os.path.exists(paths_dict['results_dir']):
        os.makedirs(paths_dict['results_dir'])

    print('Results Folder:' + str(paths_dict['results_dir']))
    print('Results File:' + str(paths_dict['results_fn']))
    
    if not os.path.exists(split_path):
        os.makedirs(split_path)
        
    split_path = os.path.join(split_path, split + '_split')
    paths_dict['split_path'] = os.path.join(split_path, expDim)
    
    print('Data split Folder:' + str(paths_dict['split_path']))
    
    return paths_dict

def load_data(infile, dataset, testSplit=''):
    '''
        This function loads the image data from a HDF5 file 
        Args:
          outfile: string, path to read file from
          
        Returns:
          f["image"][()]: numpy.array, image data as numpy array
    '''
    path = infile + '/' + dataset + '/' + testSplit + '/' + 'data.hdf5'
    
    print("---------------------------------------")
    print("Loading data")
    print("---------------------------------------\n")
    with h5py.File(path, "r") as f:
        X = f["features"][()]
        Y = f["gts"][()]
        
    return X, Y

def get_model(config):
    expDim     = config['DATA_DIM']
    archType   = config['ARCH_TYPE']
    modelType  = config['MODEL_TYPE']
    
    if modelType == 'cnn':
        if expDim == '1d':
            model = SharkVGG(1)

        elif expDim == '2d':
            model = SharkVGG2d(1)

        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

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

        model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=.99, nesterov=False, weight_decay=1e-7)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

    elif modelType == 'rcnn':
        model = SharkRCNN(3)

        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=.99, nesterov=True)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.65)

    return model, optimizer, sched
