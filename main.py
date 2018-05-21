import torchaudio
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
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 7, 3, 5, 2)
        self.conv2 = nn.Conv1d(8, 32, 7, 3, 5, 2)
        self.conv3 = nn.Conv1d(32, 64, 7, 3, 5, 2)
        self.conv4 = nn.Conv1d(64, 128, 7, 3, 5, 2)
        self.fc = nn.Linear(128*1234, 1)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x).view(-1, 128*1234)
        return self.fc(x)
        
class FC(nn.Module):
    def __init__(self, n_classes):
        super(FC, self).__init__()
        self.fc = nn.Linear(13, n_classes)
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
    
def segment(x):
    return x[:seg_size*(x.shape[0]//seg_size)].view(-1, seg_size)
    
def load_data(pth, file2label):
    x = []
    y = []
    for root, dirs, files in os.walk(pth):
        for name in files:
            prefix = name.split("_")[0]
            if name.endswith('txt') and prefix in file2label:
                feats = np.loadtxt(name)
                label = file2label[prefix]
                x.append(torch.tensor(feats).float())
                y += [label]*feats.shape[0]
    x = torch.cat(x, dim=0).cuda()
    y = torch.tensor(y).cuda()
    return x, y
    
def split(x, y):
    p = torch.randperm(y.shape[0])
    x, y = x[p], y[p]
    
    cut_point = 9*(y.shape[0]//10)
    trX, teX = x[:cut_point], x[cut_point:]
    trY, teY = y[:cut_point], y[cut_point:]
    return trX, trY, teX, teY
              
file2label, n_labels = load_labels("labels.lst")

x, y = load_data(".", file2label)

trX, trY, teX, teY = split(x,y)
print trX.shape, teX.shape
quit()

model = FC(n_labels).cuda()
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss().cuda()

s = time.time()
for i in range(9999):
    output = model(trX)
    loss = criterion(output, trY)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    #print "%s --- %s"%(i,loss.item())

    with torch.no_grad():
        output = model(teX)
        pred = output.data.max(1, keepdim=True)[1]
        acc = pred.eq(teY.view_as(pred)).sum().item() / float(len(teY))
        print "acc: ", acc


"""
s = x[:1]
rec = model(x)

s = s[0]*std + mu
rec = rec[0]*std + mu

print s
print rec
torchaudio.save('s.wav', s.unsqueeze(1).cpu(), sr)
torchaudio.save('rec.wav', rec.unsqueeze(1).cpu(), sr)
"""
