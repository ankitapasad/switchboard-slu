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
        width = 32
        
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(13, width, kernel_size, stride, padding, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, 2*width, kernel_size, stride, padding, 2, bias=False)
        self.bn2 = nn.BatchNorm1d(2*width)
        self.conv3 = nn.Conv1d(2*width, 4*width, kernel_size, stride, padding, 4, bias=False)
        self.bn3 = nn.BatchNorm1d(4*width)
        
        self.pool = nn.MaxPool1d(2)
        self.globalpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(4*width, n_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dropout(self.pool(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(F.relu(self.bn3(self.conv3(x)))))

        x = self.globalpool(x).squeeze()
        
        x = self.fc(x)
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
                file2label[fname] = topic2label[topic]
    return file2label, topic2label, curr_label


def load_data(pth, file2label):
    conv2seg = {} # convid -> list of segments
    conv2label = {} #convid -> label
    max_len = 0
    for root, dirs, files in os.walk(pth):
        for name in files:
            prefix = name.split(".")[0]
            if name.endswith('txt') and prefix in file2label:
                print name
                conv_id = prefix.split("-")[0]
                feats = torch.tensor(np.loadtxt(os.path.join(root,name))).float()
                if feats.shape[0] > max_len: max_len = feats.shape[0]
                conv2label[conv_id] = file2label[prefix]
                if conv_id not in conv2seg: conv2seg[conv_id] = []
                conv2seg[conv_id].append(feats)

    for conv in conv2seg:
        conv_segs = conv2seg[conv]
        for i in range(len(conv_segs)):
            conv_segs[i] = torch.cat((conv_segs[i], torch.zeros(max_len - conv_segs[i].shape[0], conv_segs[i].shape[1]))).transpose(0,1)
        seg_tensor = torch.stack(conv_segs)
        conv2seg[conv] = seg_tensor

    return conv2seg, conv2label


def remap_labels(labels):
    dic = {}
    curr_label = 0
    for k in labels:
        y = labels[k]
        if y not in dic:
            dic[y] = curr_label
            curr_label += 1
    for k in labels: labels[k] = dic[labels[k]]
    return labels




file2label, topic2label, n_labels = load_labels("labelsUtt.lst")

if sys.argv[1] == "preproc":
    data, labels = load_data("segmented", file2label)
    torch.save(data, 'data.pt')
    torch.save(labels, 'labels.pt')
    
    
elif sys.argv[1] == "subsample":
    data = torch.load("data.pt")
    labels = torch.load("labels.pt")
    
    toi = [358, 349, 353, 356, 351, 340]
    
    label2topic = {v: k for k, v in topic2label.iteritems()}
    
    to_del = []
    for conv in labels:
        label = labels[conv]
        topic = int(label2topic[label])
        if topic not in toi: to_del.append(conv)
        
    for conv in to_del:
        data.pop(conv, None)
        labels.pop(conv, None)
        
    torch.save(data, 'sdata.pt')
    torch.save(labels, 'slabels.pt')
    

elif sys.argv[1] == "split":
    data = torch.load("sdata.pt")
    labels = torch.load("slabels.pt")
    
    labels = remap_labels(labels)
    
    test_data = {}
    test_labels = {}
    
    nplabels = np.asarray(labels.values())
    for label in range(nplabels.max()+1):
        n_test = int((nplabels == label).sum() * 0.15)
        convs = filter(lambda k: labels[k] == label, labels)[:n_test]
        
        for c in convs:
            test_data[c] = data[c]
            test_labels[c] = labels[c]
            data.pop(c, None)
            labels.pop(c, None)
        
    torch.save(data, 'trX.pt')
    torch.save(labels, 'trY.pt')
    torch.save(test_data, 'teX.pt')
    torch.save(test_labels, 'teY.pt')
  
elif sys.argv[1] == "train":
    b = 8

    trX = torch.load("trX.pt")
    trY = torch.load("trY.pt")

    teX = torch.load("teX.pt")
    teY = torch.load("teY.pt")

    n_labels = max(trY.values()) + 1

    model = Net2(n_labels).cuda()
    
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss().cuda()

    for i in range(9999):
        model.train()
        acc = 0
        n_points = 0
        gacc = 0
        gn_points = 0.
        
        for conv in trX:
            data, label = trX[conv].cuda(), trY[conv]
            seg_label = torch.zeros(data.shape[0]).long().cuda() + label

            output = model(data)
            global_output = output.max(0)[0].unsqueeze(0)
            
            loss = criterion(output, seg_label) + criterion(global_output, seg_label[:1])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pred = output.max(1, keepdim=True)[1]
            gpred = global_output.max(1, keepdim=True)[1]
            
            acc += pred.eq(seg_label.view_as(pred)).sum().item()
            gacc += (gpred == seg_label[:1])[0].item()
            
            n_points += len(seg_label)
            gn_points += 1
            
        print "\nTrain acc: ", acc/float(n_points)
        print "Global Train acc: ", gacc/float(gn_points)

        model.eval()
        acc = 0
        n_points = 0
        gacc = 0
        gn_points = 0.
        with torch.no_grad():
            for conv in teX:
                data, label = teX[conv].cuda(), teY[conv]

                seg_label = torch.zeros(data.shape[0]).long().cuda() + label
                
                output = model(data)
                global_output = output.max(0)[0].unsqueeze(0)
                
                pred = output.data.max(1, keepdim=True)[1]
                gpred = global_output.max(1, keepdim=True)[1]

                acc += pred.eq(seg_label.view_as(pred)).sum().item()
                gacc += (gpred == seg_label[:1])[0].item()
                
                n_points += len(seg_label)
                gn_points += 1
            
            print "Test acc: ", acc/float(n_points)
            print "Global Test acc: ", gacc/float(gn_points)
