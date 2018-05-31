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
import sys
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()
        #c_in, c_out, kernel, stride, padding, dilation
        self.n_classes = n_classes
        self.dropout = nn.Dropout(0.0)
        self.width = 32
        
        kernel_size = 5
        padding = (kernel_size-1)/2
        stride = 1
        
        self.conv1 = nn.Conv1d(13, self.width, kernel_size, stride, padding, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(self.width)
        
        self.conv2 = nn.Conv1d(self.width, 2*self.width, kernel_size, stride, padding, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(2*self.width)
        
        self.conv3 = nn.Conv1d(2*self.width, 4*self.width, kernel_size, stride, padding, 4, bias=False)
        self.bn3 = nn.BatchNorm1d(4*self.width)
        
        self.conv4 = nn.Conv1d(4*self.width, 8*self.width, kernel_size, stride, padding, 8, bias=False)
        self.bn4 = nn.BatchNorm1d(8*self.width)
        
        #self.conv5 = nn.Conv1d(8*self.width, 16*self.width, kernel_size, stride, padding, 16, bias=False)
        #self.bn5 = nn.BatchNorm1d(16*self.width)
        
        self.conv6 = nn.Conv1d(8*self.width, self.n_classes, kernel_size, stride, padding, 1, bias=False)
        self.globalpool = nn.AdaptiveMaxPool1d(1)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool(F.relu(self.bn4(self.conv4(x)))))
        #x = self.dropout(self.pool(F.relu(self.bn5(self.conv5(x)))))
        x = self.conv6(x)
        x = self.globalpool(x)
        x = x.view(-1, self.n_classes)
        return 0, x
        
        
        
        
class Net2(nn.Module):
    def __init__(self, n_classes):
        super(Net2, self).__init__()
        self.n_classes = n_classes
        kernel_size = 5
        padding = (kernel_size-1)/2
        stride = 1
        
        self.fc1 = nn.Conv1d(13, 16, kernel_size, stride, padding, bias=False)
        self.fc2 = nn.Conv1d(16, 32, kernel_size, stride, padding, bias=False)
        self.fc3 = nn.Conv1d(32, 64, kernel_size, stride, padding, bias=False)
        self.globalpool = nn.AdaptiveMaxPool1d(1)
        self.fc4 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
        x = self.globalpool(x).squeeze()
        x = self.fc4(x)
        return 0, x
    
    





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
                file2label[fname] = topic2label[topic]
    return file2label, curr_label
    
def load_data(pth, file2label):
    x = []
    y = []
    max_len = 0
    for root, dirs, files in os.walk(pth):
        for name in files:
            prefix = name.split("_")[0]
            if name.endswith('txt') and prefix in file2label:
                print name
                feats = torch.tensor(np.loadtxt(os.path.join(root,name))).float()
                if feats.shape[0] > max_len: max_len = feats.shape[0]
                label = file2label[prefix]
                x.append(feats)
                y += [label]
    for i in range(len(x)): x[i] = torch.cat((x[i], torch.zeros(max_len - x[i].shape[0], x[i].shape[1]))).unsqueeze(0)
    x = torch.cat(x, dim=0).transpose(1,2)
    y = torch.tensor(y)
    return x, y
    
    
    
def remap_labels(trY, teY):
    dic = {}
    curr_label = 0
    for y in trY:
        y = y.item()
        if y not in dic:
            dic[y] = curr_label
            curr_label += 1
            
    for i in range(len(trY)): trY[i] = dic[trY[i].item()]
    for i in range(len(teY)): teY[i] = dic[teY[i].item()]
        
    return trY, teY
    
    
    
    
    
    
    

file2label, n_labels = load_labels("labelsAll.lst")

if sys.argv[1] == "preproc":
    data, labels = load_data("data", file2label)

    torch.save(data, 'data.pt')
    torch.save(labels, 'labels.pt')
    
elif sys.argv[1] == "preproc2":
    data, labels = load_data("train", file2label)
    test_data, test_labels = load_data("test", file2label)
    
    print test_labels
    
    torch.save(torch.tensor(data), 'trX.pt')
    torch.save(torch.tensor(labels), 'trY.pt')
    torch.save(torch.tensor(test_data), 'teX.pt')
    torch.save(torch.tensor(test_labels), 'teY.pt')
    
elif sys.argv[1] == "split":
    data = torch.load("data.pt").numpy()
    labels = torch.load("labels.pt")
    print data.shape, labels.shape
    
    labels, _ = remap_labels(labels, [])
    labels = labels.numpy()

    #for i in range(len(labels)):
    #    data[i] += labels[i]*5

    test_data = None
    test_labels = None
    for i in range(labels.max().item()+1):
        n_test = int((labels == i).sum() * 0.15)
        indices = np.where(labels==i)[0][:n_test]
        
        if test_data is None: test_data = data[indices]
        else: test_data = np.vstack((test_data, data[indices]))
        
        if test_labels is None: test_labels = labels[indices]
        else: test_labels = np.concatenate((test_labels, labels[indices]))
        
        data = np.delete(data, indices, axis=0)
        labels = np.delete(labels, indices, axis=0)
        
    torch.save(torch.tensor(data), 'trX.pt')
    torch.save(torch.tensor(labels), 'trY.pt')
    torch.save(torch.tensor(test_data), 'teX.pt')
    torch.save(torch.tensor(test_labels), 'teY.pt')
  
elif sys.argv[1] == "train":
    b = 8

    trX = torch.load("trX.pt").cuda()
    trY = torch.load("trY.pt").cuda()

    teX = torch.load("teX.pt").cuda()
    teY = torch.load("teY.pt").cuda()

    trY, teY = remap_labels(trY, teY)

    n_labels = trY.max().item() + 1
    
    trX = F.avg_pool1d(trX, 5)
    teX = F.avg_pool1d(teX, 5)

    model = Net2(n_labels).cuda()
    
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss().cuda()

    length = 1500
    s = time.time()
    for i in range(9999):
        #p = torch.randperm(trX.shape[0])
        #trX, trY = trX[p], trY[p]
        
        model.train()
        acc = 0
        n_points = 0
        for batch_idx in range(trX.shape[0]//b):
            start, end = batch_idx*b, (batch_idx+1)*b
            data, target = trX[start:end], trY[start:end]
            #data, target = trX, trY
            
            t_s = np.random.randint(0, data.shape[2]-length)
            data = data[:,:,t_s:t_s+length]
            #data = data[:,:,:500]
            
            attention, output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pred = output.max(1, keepdim=True)[1]
            acc += pred.eq(target.view_as(pred)).sum().item()
            n_points += len(target)
            
        print "Train acc: ", acc/float(n_points)
        
        """
        if i % 20 == 0:
            attetion = attention[0].view(-1).data.cpu().numpy()
            plt.figure()
            plt.plot(attetion)
            plt.savefig("test_attention_%s.png"%i)
        """

        model.eval()
        with torch.no_grad():
            data, target = teX, teY
            #data = data[:,:,:]

            attention, output = model(data)
            
            #_, pred = output.topk(5, 1, True, True)
            #pred = pred.t()
            #acc5 = pred.eq(target.view(1, -1).expand_as(pred)).view(-1).sum(0).item() / float(len(teY))
            
            pred = output.data.max(1, keepdim=True)[1]
            #print "test pred: ", pred.view(-1)
            acc = pred.eq(target.view_as(pred)).sum().item() / float(len(target))
            print "Test acc: ", acc
           



