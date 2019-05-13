# Train and validate

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as pt_utils
import torch.optim as optim
import os
import sys
import pandas as pd
import utils
import agent

# Set a random seed
seed= 52469
np.random.seed(seed)

# Epochs
num_epochs = 10
# Learning rate
learning_rate = 0.001

# logs dir
OUT_DIR = os.path.abspath('.')
logs_dir = os.path.join(OUT_DIR,'logs')
if not os.path.exists(logs_dir):
    os.mkdir(logs_dir) 

# Read train data csv
CSV_DIR = os.path.join(os.path.abspath('.'),'data','train_board_pos.csv')
df = pd.read_csv(CSV_DIR,header=None,error_bad_lines=False)

# PANDAS version
pd_version = pd.__version__.split('.')
if int(pd_version[0]) >= 0 and int(pd_version[1]) >= 24 and int(pd_version[2]) >= 0:  # PD version >= 0.24.2
    data = df.to_numpy(dtype=np.int32)
else:
    data = df.values
    data = data.astype(np.int32)

# Compute pip count
pip_o, pip_x = utils.pip_count(data)
# Compute reward
reward = utils.cal_reward(pip_x,pip_o)

# Preprocess
data_processed = utils.data_preprocess(data)

# Preapre for dataloader creation
x = data_processed
y = reward

batch_size = 256

# Shuffle indices
N = len(data)
idx = np.random.permutation(N)
idx_train, idx_val = idx[:1000000], idx[1000000:]

# Split data into train and validation
X = {'train': x[idx_train], 'val': x[idx_val]}
Y = {'train': y[idx_train], 'val': y[idx_val]}

print ('Training dataset shape:   ', X['train'].shape, 'max: ',np.max(X['train']), 'min: ',np.min(X['train']), 'dtype: ', X['train'].dtype)
print ('Training target shape:    ', Y['train'].shape, 'max: ',np.max(Y['train']), 'min: ',np.min(Y['train']), 'dtype: ', Y['train'].dtype)
print ('Validation dataset shape: ', X['val'].shape, 'max: ',np.max(X['val']), 'min: ',np.min(X['val']), 'dtype: ', X['val'].dtype)
print ('Validation target shape:  ', Y['val'].shape, 'max: ',np.max(Y['val']), 'min: ',np.min(Y['val']), 'dtype: ', Y['val'].dtype)

# Create pytorch datasets for train and validation
datasets = {i: pt_utils.TensorDataset(torch.from_numpy(X[i]).float(),torch.from_numpy(Y[i]).long()) for i in ['train', 'val']}
# Create pytorch dataloaders for train and validation
dataloaders_dict = {i: pt_utils.DataLoader(datasets[i], batch_size=batch_size, shuffle=False, num_workers=4) for i in ['train', 'val']}

# Call agent model
agent_net = agent.BG_Agent()
# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
agent_net = agent_net.to(device)
# Optimizer
optimizer = optim.Adam(agent_net.parameters(), lr=learning_rate)
# Loss function
criterion = nn.CrossEntropyLoss()
# Train
agent.train_model(agent_net, dataloaders_dict, criterion, optimizer, logs_dir, num_epochs=num_epochs)

