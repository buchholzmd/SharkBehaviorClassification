import os
import h5py
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from networks.rcnn import SharkRCNN2d
from networks.rnn import SharkLSTM, SharkGRU
from networks.attention_rnn import SharkAttentionLSTM
from networks.cnn import SharkVGG1, SharkVGG2, Sharkception
from networks.cnn2d import Shark2dVGG1, Shark2dVGG2, Sharkception2d

def accuracy(out, labels):
    _, pred = torch.max(out,1)
    correct = (pred == labels).sum().item()
    acc = 100*correct/len(labels)
    return acc

def train(model, train_loader, val_loader, parameters, sched, path):
    folder_ = path
    
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=.99, nesterov=True)
    
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
            
    torch.save(model.state_dict(), './'+folder_+'/_last.pth')

def get_exp_paths(config):
    expDim    = config['DATA_DIM']
    wavelet   = config['WAVELET']
    archType  = config['ARCH_TYPE']
    testSplit = config['TEST_SPLIT']
    modelType = config['MODEL_TYPE']

    folder_ = os.path.join('./models', 'Attention' + modelType, expDim, archType)

    results_folder = os.path.join('./results', testSplit)

    if wavelet:
        results_file = 'wavelet_' + archType + '_' + expDim + '_' + testSplit + '.txt'
    else:
        results_file = 'Attention' + archType + '_' + expDim + '_' + testSplit + '.txt'

    if not os.path.exists(folder_):
        os.makedirs(folder_)
    
    print('Folder:' + str(folder_))

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    print('Results Folder:' + str(results_folder))
    print('Results File:' + str(results_file))
    
    return folder_, results_folder, results_file

def write(data, outfile):
    '''
        This function writes the pre-processed image data to a HDF5 file
        Args:
          data: numpy.array, image data as numpy array
          outfile: string, path to write file to
    '''
    print("---------------------------------------")
    print("Saving data")
    print("---------------------------------------\n")
    with h5py.File(outfile, "w") as f:
        f.create_dataset("shark_data", data=data, dtype=data.dtype)

def load_data(infile, dataset, testSplit=''):
    '''
        This function loads the image data from a HDF5 file 
        Args:
          outfile: string, path to read file from
          
        Returns:
          f["image"][()]: numpy.array, image data as numpy array
    '''
    X_path = infile + '/' + testSplit + '/' + dataset + '_data.hdf5'
    Y_path = infile + '/' + testSplit + '/' + dataset + '_gt.hdf5'
    
    print("---------------------------------------")
    print("Loading data")
    print("---------------------------------------\n")
    with h5py.File(X_path, "r") as f:
        X = f["shark_data"][()]
        
    print("---------------------------------------")
    print("Loading data")
    print("---------------------------------------\n")
    with h5py.File(Y_path, "r") as f:
        Y = f["shark_data"][()]
        
    return X, Y

def get_model(config):
    expDim     = config['DATA_DIM']
    archType   = config['ARCH_TYPE']
    modelType  = config['MODEL_TYPE']
    
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

        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=.0)#, momentum=.9, nesterov=True)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=.99, nesterov=True)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10000], gamma=0.1)

    elif modelType == 'rnn':
        hidden_size = 128
        if expDim == '1d':
            dat_size = 1
        else:
            dat_size = 6
            
        out_size = 4
        num_layers = 2
        fc_dim = 512

        if archType == 'LSTM':
            model = SharkAttentionLSTM(dat_size,
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
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=.99, nesterov=True)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10000,20000], gamma=0.1)

    elif modelType == 'rcnn':
        if expDim == '1d':
            model = SharkRCNN(1, 4)

        elif expDim == '2d':
            model = SharkRCNN2d(1, 4)

        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=.0)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=.99, nesterov=True)
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10000], gamma=0.1)

    return model, optimizer, sched