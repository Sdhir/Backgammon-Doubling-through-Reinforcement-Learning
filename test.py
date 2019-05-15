# Test and evaluate

import numpy as np
import torch
import torch.utils.data as pt_utils
import os
import sys
import pandas as pd
import utils
import agent

# best weights file
WTS_FILE = 'logs/wt_best_ep5.pth'
# Read test data csv
CSV_DIR = os.path.join(os.path.abspath('.'),'data','26bit_reverse100.txt') # test_double.csv
df = pd.read_csv(CSV_DIR,header=None,error_bad_lines=False)

# PANDAS version
pd_version = pd.__version__.split('.')
if int(pd_version[0]) >= 0 and int(pd_version[1]) >= 24 and int(pd_version[2]) >= 0:  # PD version >= 0.24.2
    data = df.to_numpy(dtype=np.int32)
else:
    data = df.values
    data = data.astype(np.int32)

if data.shape[1]==27:
    test_data = data[:,1:]
    Y_gt = data[:,0]
else:
    test_data = data

# Preprocess
data_processed = utils.data_preprocess(test_data)

x_test = data_processed
print ('Testing dataset shape:   ', x_test.shape, 'max: ',np.max(x_test), 'min: ',np.min(x_test), 'dtype: ', x_test.dtype)

batch_size = 256

# Create pytorch datasets for train and validation
test_datasets = pt_utils.TensorDataset(torch.from_numpy(x_test).float())
# Create pytorch dataloaders for train and validation
testloader = pt_utils.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=4)

# Call Agent model
agent_net = agent.BG_Agent()
# Detect device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
agent_net = agent_net.to(device)
# load weights file
agent_net.load_state_dict(torch.load(WTS_FILE))
# Set model to testing phase
agent_net.eval()
# Predict
Y_pred = agent.predict(agent_net,testloader)
np.save('Y_pred.npy',Y_pred)

if data.shape[1]==27:
    acc = utils.evaluate(Y_gt, Y_pred)
    print("Prediction accuracy: ",acc)
    


