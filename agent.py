# -*- coding: utf-8 -*-
# Agent model

import numpy as np
import torch
import torch.nn as nn
import os
import time
import copy
import sys

# Agent Action Network - MLP
class BG_Agent(nn.Module):
    def __init__(self):
        super(BG_Agent, self).__init__()
        self.inputSize = 26
        self.hiddenSize1 =  512
        self.hiddenSize2 =  64
        self.outputSize = 2
        
        self.fc1 = nn.Linear(self.inputSize, self.hiddenSize1)
        self.act1 = torch.nn.ReLU()
        self.fc2 = nn.Linear(self.hiddenSize1, self.hiddenSize2)
        self.act2 = torch.nn.ReLU()
        self.fc3 = nn.Linear(self.hiddenSize2, self.outputSize)
    
    def forward(self, x):
        x = self.act1((self.fc1(x)))
        x = self.act2((self.fc2(x)))
        out = self.fc3(x)
        return out 

# Training agent
def train_model(model, dataloaders, criterion, optimizer, log_dir, num_epochs=25):
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    
    # Initialize
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    
    # Save history to csv
    csv = open(os.path.join(log_dir,'eval_history.csv'),'w')
    csv.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
    
    # Iterate through epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 12)

        # Each epoch has a training and validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            # loss and accuracy
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for in_state, labels in dataloaders[phase]:
                in_state = in_state.to(device)
                labels = labels.to(device)

                # zero the weight and bias gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs 
                    outputs = model(in_state)
                    # calculate loss
                    loss = criterion(outputs, labels)
                    
                    # Argmax
                    _, preds = torch.max(outputs, 1)

                    # backward pass and update only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # stats
                running_loss += loss.item() * in_state.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Loss
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # Accuracy
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{}_loss: {:.4f} {}_acc: {:.4f} '.format(phase, epoch_loss, phase, epoch_acc))
            # train loss nad accuracy
            if phase == 'train':
                tr_loss = epoch_loss
                tr_acc = epoch_acc.cpu().data.numpy()
            # validation loss nad accuracy
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc.cpu().data.numpy()

            # deep copy the best model based on validation accuracy 
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
        
        # Update csv
        csv.write(str(epoch)+','+str(tr_loss)+','+str(tr_acc)+','+str(val_loss)+','+str(val_acc)+'\n')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    check_pt = "wt_best_ep{}.pth".format(best_epoch)
    # save model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(),os.path.join(log_dir,check_pt))
    return

# Testing agent
def predict(model,testloader):
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize prediction array
    Y_pred = np.array([])
    
    # No grads when testing
    with torch.no_grad():
        for test_data in testloader:
            # Predicted output - logits
            outputs = model(test_data[0].to(device))
            # Argmax
            predicted = torch.argmax(outputs.data, dim=1)
            # Update prediction array
            Y_pred = np.append(Y_pred,predicted.cpu().numpy())
    Y_pred = np.array(Y_pred)
    return Y_pred
    