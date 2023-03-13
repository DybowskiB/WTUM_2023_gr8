import pandas as pd
import numpy as np
import torch


class LSTMDataset(torch.utils.data.Dataset):
  def __init__(self, train=False, 
               validate=False,
               test=False,
               window_size=200):

    self.df_traffic_train = pd.read_csv(" .csv", index_col=[0])
    self.df_traffic_val = pd.read_csv(" .csv", index_col=[0])
    self.df_traffic_test = pd.read_csv(" .csv", index_col=[0])
    
    if train: #process train dataset
      features = self.df_traffic_train
      target = self.df_traffic_train.Sunspots
    elif validate: #process validate dataset
      features = self.df_traffic_val
      target = self.df_traffic_val.Sunspots
    else: #process test dataset
      features = self.df_traffic_test
      target = self.df_traffic_test.Sunspots
    
    self.x, self.y = [], []
    for i in range(len(features) - window_size):
        v = features.iloc[i:(i + window_size)].values
        self.x.append(v)
        self.y.append(target.iloc[i + window_size])  
    self.num_sample = len(self.x)
    
  def __getitem__(self, index):
    x = self.x[index].astype(np.float32)
    y = self.y[index].astype(np.float32)
    return x, y


  def __len__(self):
    return self.num_sample