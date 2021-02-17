import os
import h5py
import torch
import gpytorch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from collections import Counter
from networks.cnn import SharkVGG
from contextlib import suppress
from networks.rcnn import SharkRCNN
from torch.autograd import Variable
from networks.rnn import SharkLSTM, SharkGRU
from sklearn.model_selection import train_test_split
from networks.rakl import RecurrentKernelLearningModel

def accuracy(probs, labels):
    pred = probs.argmax(-1)
    correct = pred.eq(labels.view_as(pred)).sum().item()
    acc = 100*correct/len(labels)
    return acc

def train(config, model, criterion, train_loader, val_loader, parameters, optimizer, sched, path, rnn=False, kernel_learning=False, model_path=None):
    folder_ = path

    attn        = config['ATTENTION'] == 'attn'
    seq_len     = config['SEQ_LEN']
    num_classes = config['NUM_CLASSES']

    if kernel_learning:
        likelihood = model['likelihood']
        model = model['model']

        if model_path:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict['model'])
            likelihood.load_state_dict(state_dict['likelihood'])
    
    mean_train_losses = []
    mean_val_losses = []

    mean_train_acc = []
    mean_val_acc = []

    minLoss = np.inf
    maxValacc = -np.inf
    for epoch in range(500):
        print('EPOCH: ', epoch+1)
        train_acc = []
        val_acc = []

        running_loss = 0.0

        model.train()

        if kernel_learning:
            likelihood.train()
        
        count = 0
        num_samples = 8
        with gpytorch.settings.num_likelihood_samples(num_samples) if kernel_learning else suppress():
            for sequences, labels in train_loader:
                labels = labels.squeeze()
                sequences = Variable(sequences.cuda())
                labels = Variable(labels.cuda())
                
                if kernel_learning and attn:
                    outputs, attn_weights = model(sequences)

                    batch_size = sequences.size(0)
                    # repeat label for every element in sequence
                    ext_labels = labels.repeat_interleave(seq_len)
                else:
                    outputs = model(sequences)
                    ext_labels = labels

                optimizer.zero_grad()
                if kernel_learning:
                    loss = -criterion(outputs, ext_labels)
                else:
                    loss = criterion(outputs, labels)

                if kernel_learning:
                    # This gives us 8 samples from the predictive distribution
                    # Take the mean over all samples
                    outputs = likelihood(outputs).probs.mean(0)

                    if attn:
                        outputs = outputs.reshape((batch_size, seq_len, num_classes))
                        outputs = torch.bmm(attn_weights, outputs).squeeze()

                else:
                    outputs = torch.softmax(outputs, dim=-1)

                train_acc.append(accuracy(outputs, labels))

                loss.backward()
                
                if rnn or kernel_learning:
                    if kernel_learning:
                        params = model.feature_extractor.parameters()
                    else:
                        params = model.parameters()
                    torch.nn.utils.clip_grad_norm_(params, 5)
                    
                optimizer.step()        

                running_loss += loss.item()
                count +=1
        
        # 25
        if epoch % 50 == 0:
            sched.step()
            
        print('Training loss:.......', running_loss/count)
        mean_train_losses.append(running_loss/count)

        model.eval()

        if kernel_learning:
            likelihood.eval()

        count = 0
        val_running_loss = 0.0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16) if kernel_learning else suppress():
            for sequences, labels in val_loader:
                labels = labels.squeeze()
                sequences = Variable(sequences.cuda())
                labels = Variable(labels.cuda())

                if kernel_learning and attn:
                    outputs, attn_weights = model(sequences)
                else:
                    outputs = model(sequences)

                if kernel_learning:
                    batch_size = sequences.size(0)
                    # This gives us 16 samples from the predictive distribution
                    # Take the mean over all samples
                    outputs = likelihood(outputs).probs.mean(0)

                    if attn:
                        outputs = outputs.reshape((batch_size, seq_len, num_classes))
                        outputs = torch.bmm(attn_weights, outputs).squeeze()

                else:
                    outputs = torch.softmax(outputs, dim=1)

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
            if kernel_learning:
                state_dict = {'model': model.state_dict(),
                              'likelihood': likelihood.state_dict()}
                torch.save(state_dict, './'+folder_+'/_loss.pth')
            else:
                torch.save(model.state_dict(), './'+folder_+'/_loss.pth' )
            print(f'NEW BEST LOSS_: {mean_val_loss} ........old best:{minLoss}')
            minLoss = mean_val_loss
            print('')

        if val_acc_ > maxValacc:
            if kernel_learning:
                state_dict = {'model': model.state_dict(),
                              'likelihood': likelihood.state_dict()}
                torch.save(state_dict, './'+folder_+'/_acc.pth')
            else:
                torch.save(model.state_dict(), './'+folder_+'/_acc.pth' )
            print(f'NEW BEST ACC_: {val_acc_} ........old best:{maxValacc}')
            maxValacc = val_acc_

        if epoch%500 == 0 :
            if kernel_learning:
                state_dict = {'model': model.state_dict(),
                              'likelihood': likelihood.state_dict()}
                torch.save(state_dict, './'+folder_+'/save_'+str(epoch)+'.pth')
            else:
                torch.save(model.state_dict(), './'+folder_+'/save_'+str(epoch)+'.pth')
            print(f'DIV 200: Val_acc: {val_acc_} ..Val_loss:{mean_val_loss}')
            
    
    if kernel_learning:
        state_dict = {'model': model.state_dict(),
                      'likelihood': likelihood.state_dict()}
        torch.save(state_dict, './'+folder_+'/_last.pth')
    else:
        torch.save(model.state_dict(), './'+folder_+'/_last.pth')

    train_acc_series = pd.Series(mean_train_acc)
    val_acc_series = pd.Series(mean_val_acc)
    train_acc_series.plot(label="train")
    val_acc_series.plot(label="validation")
    plt.legend()
    plt.savefig('./train_acc.png')

    plt.clf()

    train_acc_series = pd.Series(mean_train_losses)
    val_acc_series = pd.Series(mean_val_losses)
    train_acc_series.plot(label="train")
    val_acc_series.plot(label="validation")
    plt.legend()
    plt.savefig('./train_loss.png')

def get_exp_paths(config):
    split       = config['SPLIT']
    split_path  = config['SPLIT_PATH']
    expDim      = config['DATA_DIM']
    wavelet     = config['WAVELET']
    archType    = config['ARCH_TYPE']
    testSplit   = config['TEST_SPLIT']
    modelType   = config['MODEL_TYPE']
    attention   = config['ATTENTION']
    seq_len     = config['SEQ_LEN']
    
    paths_dict = {}

    paths_dict['folder'] = os.path.join('./models', str(seq_len), attention + modelType, expDim, archType)

    paths_dict['results_dir'] = os.path.join('./results', split + '_' + str(seq_len))
    paths_dict['results_dir'] = os.path.join(paths_dict['results_dir'], testSplit)

    paths_dict['results_fn'] = wavelet + '_' + attention + archType + '_' + expDim + '.txt'

    if not os.path.exists(paths_dict['folder']):
        os.makedirs(paths_dict['folder'])
    
    print('Folder:' + str(paths_dict['folder']))

    if not os.path.exists(paths_dict['results_dir']):
        os.makedirs(paths_dict['results_dir'])

    print('Results Folder:' + str(paths_dict['results_dir']))
    print('Results File:' + str(paths_dict['results_fn']))
    
    if not os.path.exists(split_path):
        os.makedirs(split_path)
        
    split_path = os.path.join(split_path, split + '_split_' + str(seq_len))
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
    path = os.path.join(infile, dataset)
    path = os.path.join(path, testSplit)
    path = os.path.join(path, 'data.hdf5')
    
    print("---------------------------------------")
    print("Loading data")
    print("---------------------------------------\n")
    with h5py.File(path, "r") as f:
        X = f["features"][()]
        Y = f["gts"][()]
        
    return X, Y

def get_model(config, test=False):
    expDim     = config['DATA_DIM']
    seq_len    = config['SEQ_LEN']
    archType   = config['ARCH_TYPE']
    modelType  = config['MODEL_TYPE']
    attn       = config['ATTENTION'] == 'attn'
    attn_dim   = config['ATTN_DIM']
    
    if modelType == 'cnn':
        model = SharkVGG(1, dim=expDim, non_local=attn)

        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)

        kernel_learning = False

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
                              fc_dim=fc_dim,
                              attention=attn,
                              attn_dim=attn_dim)

        elif archType == 'GRU':
            model = SharkGRU(dat_size, 
                             hidden_size, 
                             out_size, 
                             num_layers=num_layers,
                             fc_dim=fc_dim,
                             attention=attn,
                             attn_dim=attn_dim)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.999, nesterov=False, weight_decay=1e-7)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.65)

        kernel_learning = False  

    elif modelType == 'rcnn':
        model = SharkRCNN(3)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.65)

        kernel_learning = False

    elif modelType == 'svdkl':
        hidden_size = config['HIDDEN_DIM']
        if expDim == '1d':
            dat_size = 1
        else:
            dat_size = 6
            
        num_classes = config['NUM_CLASSES']
        num_layers = config['NUM_LAYERS']

        model = RecurrentKernelLearningModel(dat_size, 
                                             hidden_size,
                                             seq_len,
                                             num_layers=num_layers,
                                             attention=attn,
                                             attn_dim=attn_dim)

        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)

        # 1d hyperparameters
        # rnn_wd = 1e-3
        # gpl_lr = 1e-3

        # dkl_wd = 0
        # dkl_lr = 1e-2

        # momentum = 0.99
        # nesterov = True

        # gamma = 0.65

        # 2d hyperparameters
        rnn_wd = 1e-4
        gpl_lr = 1e-3

        dkl_wd = 0 
        dkl_lr = 1e-1 

        momentum = 0.9
        nesterov = True

        gamma = 0.75

        if attn:
            atn_wd = 1e-3
            optimizer = torch.optim.SGD([
                {'params': model.feature_extractor.gru.parameters(), 'weight_decay': rnn_wd},
                {'params': model.feature_extractor.attn.parameters(), 'weight_decay': atn_wd},
                {'params': model.gp_layer.hyperparameters(), 'lr': gpl_lr},
                {'params': model.gp_layer.variational_parameters()},
                {'params': likelihood.parameters()},
            ], lr=dkl_lr, momentum=momentum, nesterov=nesterov, weight_decay=dkl_wd)
        else:
            optimizer = torch.optim.SGD([
                {'params': model.feature_extractor.gru.parameters(), 'weight_decay': rnn_wd},
                {'params': model.gp_layer.hyperparameters(), 'lr': gpl_lr},
                {'params': model.gp_layer.variational_parameters()},
                {'params': likelihood.parameters()},
            ], lr=dkl_lr, momentum=momentum, nesterov=nesterov, weight_decay=dkl_wd)
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)


        model = {'model': model,
                 'likelihood': likelihood}

        kernel_learning = True

    if not test:
        if kernel_learning:
            model['model'] = model['model'].cuda()
            model['likelihood'] = model['likelihood'].cuda()
        else:
            model = model.cuda()
        
    return model, optimizer, sched
