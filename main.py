import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import os
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(13, 16, 5, 3, 3, 1)
        self.conv2 = nn.Conv1d(16, 32, 5, 3, 3, 1)
        self.conv3 = nn.Conv1d(32, 64, 5, 3, 3, 1)
        self.conv4 = nn.Conv1d(64, 128, 5, 3, 3, 1)
        self.pool = nn.MaxPool1d(7)
        self.fc = nn.Linear(128, n_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        x = F.selu(self.conv1(x))
        x = F.selu(self.conv2(x))
        x = F.selu(self.conv3(x))
        x = F.selu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, 128)
        return self.fc(x)
        
class FC(nn.Module):
    def __init__(self, input_dim, n_classes):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, n_classes)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        return self.fc(x)
        
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(seg_size, 1024)
        self.fc2 = nn.Linear(1024, seg_size)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def segment(x, size): 
    x = x.transpose(0,1)
    s = size*x.shape[0]
    x = x[:, :s*(x.shape[1]//(size*x.shape[0]))]
    x = x.view(x.shape[0], -1, size).transpose(0,1)
    return x

def load_labels(pth):
    curr_label = 0
    topic2label = {}
    file2label = {}
    with open(pth, "r") as f:
        for line in f.readlines():
            if len(line) > 1:
                fname, topic = line.split(" ")
                if topic not in topic2label:
                    topic2label[topic] = curr_label
                    curr_label += 1
                file2label[fname[:-2]] = topic2label[topic]
    return file2label, curr_label
    
def load_data(pth, file2label):
    x = []
    y = []
    for root, dirs, files in os.walk(pth):
        for name in files:
            prefix = name.split("_")[0]
            if name.endswith('txt') and prefix in file2label:
                print name
                feats = torch.tensor(np.loadtxt(os.path.join(root,name))).float()
                print "original shape: ", feats.shape
                feats = segment(feats, 500)
                print "concatenated: ", feats.shape
                label = file2label[prefix]
                x.append(feats)
                y += [label]*feats.shape[0]
    x = torch.cat(x, dim=0)
    y = torch.tensor(y)
    return x, y
   
     
file2label, n_labels = load_labels("labels.lst")


"""
trX, trY = load_data("train", file2label)
teX, teY = load_data("test", file2label)

torch.save(trX, 'trX.pt')
torch.save(trY, 'trY.pt')
torch.save(teX, 'teX.pt')
torch.save(teY, 'teY.pt')


quit()
"""

b = 256

trX = torch.load("trX.pt").cuda()
trY = torch.load("trY.pt").cuda()
teX = torch.load("teX.pt").cuda()
teY = torch.load("teY.pt").cuda()
print trX.shape, trY.shape, teX.shape, teY.shape


for i in range(6):
    print i, (trY == i).float().mean().item()

for i in range(6):
    print i, (teY == i).float().mean().item()


model = Net(n_labels).cuda()
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss().cuda()

s = time.time()
for i in range(9999):
    for batch_idx in range(trX.shape[0]//b):
        start, end = batch_idx*b, (batch_idx+1)*b
        data, target = trX[start:end], trY[start:end]
        
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    with torch.no_grad():
        output = model(teX)
        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(teY.view_as(pred)).sum().item() / float(len(teY))
        print "acc: ", acc
