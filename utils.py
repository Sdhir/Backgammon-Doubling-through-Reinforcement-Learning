# Utilities file

import numpy as np

# x - Player
# o - Opponent
# Pip - denote the number of positions a player needs to move his/her 
# checkers to bring home
def pip_count(x):
    # Opponent
    bit_pos_o = np.linspace(25,0,26,dtype=np.uint8)
    x_o = -1 * x.copy()
    x_o[x_o<0] = 0
    x_o = x_o.astype(np.uint8)
    pip_o = np.sum(bit_pos_o*x_o,axis=1)
    # Player
    bit_pos_x = np.linspace(0,25,26,dtype=np.uint8)
    x_x = 1 * x.copy()
    x_x[x_x<0] = 0
    x_x = x_x.astype(np.uint8)
    pip_x = np.sum(bit_pos_x*x_x,axis=1)
    return pip_o, pip_x

# Preprocess data
def data_preprocess(data):
    # scale units
    data_max = np.max(abs(data), axis = 1)
    data_norm = data.T/data_max
    return data_norm.T

# Reward calculation
def cal_reward(pip_x,pip_o):
    max_pips = np.maximum(pip_x,pip_o)
    adv =  (pip_x.astype(np.int8) - pip_o.astype(np.int8))/max_pips
    reward = np.zeros_like(adv)
    reward[adv>=0] = 1
    return reward

# Evaluation - accuracy
def evaluate(Y_gt, Y_pred):
    acc = np.sum(Y_gt == Y_pred)/len(Y_pred)
    return acc
