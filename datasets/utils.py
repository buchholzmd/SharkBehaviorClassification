import os
import h5py
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from collections import Counter

def write(data, gts, outfile):
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
        f.create_dataset("features", data=data, dtype=data.dtype)
        f.create_dataset("gts", data=gts, dtype=gts.dtype)

def load_data(config):
    all_data   = config['DATA_PATH']
    
    dfs = []
    for i in range(7):
        paths = [all_data + '/feeding/csv/Feeding_25Hz_',
                 all_data + '/swimming/csv/Swimming_25Hz_',
                 all_data + '/resting/csv/Resting_25Hz_',
                 all_data + '/ndm/csv/NDM_25Hz_']

        df = pd.concat((pd.read_csv(path + str(i+1) + '.csv',
                                    index_col=['Date_Time'],
                                    parse_dates=['Date_Time'],
                                    infer_datetime_format=True) for path in paths), ignore_index=False, sort=False).iloc[:, 1:9]


        df = df.replace(to_replace={"Non directed motion": "NDM"})
        dfs.append(df)
        
    return dfs
    
def split_data(dfs, config):
    split_type = config['SPLIT']
    
    df_dict = {}
    if split_type == 'experiment':
        # Train: 1, 2, 3, 4, 7
        # Val: 6
        # Test: 5
        df_dict['train'] = pd.concat([dfs[0], dfs[1], dfs[2], dfs[3], dfs[6]])
        df_dict['val'] = dfs[5]
        df_dict['test'] = dfs[4]
        
    elif split_type == 'full':
        data = pd.concat(dfs)
        
        df_dict['train'], df_dict['test'] = train_test_split(data,
                                                             test_size=0.25, 
                                                             random_state=33)

        df_dict['train'], df_dict['val'] = train_test_split(df_dict['train'],
                                                            test_size=0.2, 
                                                            random_state=33)
        
        df_dict['train'] = df_dict['train'].sort_index()
        df_dict['val'] = df_dict['val'].sort_index()
        df_dict['test'] = df_dict['test'].sort_index()
           
    else:
        print("Not a valid split type (full/experiment)")
        exit()
    
    print("*"*44)
    print('Train shape:', df_dict['train'].shape)
    print('Val   shape:', df_dict['val'].shape)
    print('Test  shape:', df_dict['test'].shape)
    print("="*44)
    print()
    
    return df_dict
    
def normalize(df_dict, features):
    for column in features:
        mean = np.mean(df_dict['train'][column])
        std  = np.std(df_dict['train'][column])

        df_dict['train'][column] = df_dict['train'][column].map(lambda x: (x-mean)/std)
        df_dict['val'][column]   = df_dict['val'][column].map(lambda x: (x-mean)/std)
        df_dict['test'][column]  = df_dict['test'][column].map(lambda x: (x-mean)/std)
        
    return df_dict
    
def group_times(df):
    time_diff = df.index.to_series().diff()
    breaks = time_diff > pd.Timedelta('1s')
    groups = breaks.cumsum()
    
    df['Group'] = groups
    
    return df  

def sample_sequences(df, features, num_samples=None, seq_len=50, dims=1, test=False):
    X = []
    Y = []
    
    label_list = ['Feeding', 'Swimming', 'Resting', 'NDM']
    
    for idx, label in enumerate(label_list):
        print(str(idx) + ": " + label)
        
        class_df = df.loc[df['Label'] == label]
        groups = class_df['Group'].unique()
        if not test:
            X_class = np.zeros((num_samples, seq_len, dims), dtype=np.float32)
            Y_class = np.full((num_samples, 1), idx, dtype=np.int64)
            
            for i in range(num_samples):
                chunk_idx = groups[np.random.randint(len(groups))]
                
                data = class_df.loc[class_df['Group'] == chunk_idx][features].to_numpy()
                
                while(len(data) <= seq_len):
                    chunk_idx = groups[np.random.randint(len(groups))]

                    data = class_df.loc[class_df['Group'] == chunk_idx][features].to_numpy()
                
                rand = np.random.randint(len(data)-seq_len)
                
                X_class[i] = data[rand:rand+seq_len]
                
        else:
            X_class = []
            Y_class = []
            for group in groups:
                data = class_df.loc[class_df['Group'] == group][features].to_numpy()
                
                num_samples = len(data)//50
                
                X_group = np.zeros((num_samples, seq_len, dims), dtype=np.float32)
                Y_group = np.full((num_samples, 1), idx, dtype=np.int64)
                
                for i in range(num_samples):
                    X_group[i] = data[seq_len*i:seq_len*(i+1)]

                X_class.append(X_group)
                Y_class.append(Y_group)

            X_class, Y_class = np.concatenate(X_class), np.concatenate(Y_class)

            print(label + " -- num test points: " + str(Y_class.shape[0]))
                    
        X.append(X_class)
        Y.append(Y_class)
        
    return np.concatenate(X), np.concatenate(Y)