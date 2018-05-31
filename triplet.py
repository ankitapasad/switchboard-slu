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
import random

class Triplet(nn.Module):
    def __init__(self, embedder):
        super(Triplet, self).__init__()
        self.embedder = embedder

    def forward(self, anchor, neg, pos):
        ea = self.embedder(anchor)
        en = self.embedder(neg)
        ep = self.embedder(pos)
        
        dist_a = F.pairwise_distance(ea, en, 2)
        dist_b = F.pairwise_distance(ea, ep, 2)
        return dist_a, dist_b, ea, en, ep
        
class Embedder(nn.Module):
    def __init__(self, emb_size):
        super(Embedder, self).__init__()
        self.emb_size = emb_size
        
        kernel_size = 5
        padding = (kernel_size-1)/2
        stride = 1
        
        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(13, 16, kernel_size, stride, padding, 1, bias=False)
        self.conv2 = nn.Conv1d(16, 32, kernel_size, stride, padding, 2, bias=False)
        self.conv3 = nn.Conv1d(32, 64, kernel_size, stride, padding, 4, bias=False)
        self.conv4 = nn.Conv1d(64, emb_size, kernel_size, stride, padding, 8, bias=False)
        
        self.globalpool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.conv4(x)
        x = self.globalpool(x)
        return x.squeeze()
        





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
    x = []
    y = []
    max_len = 0
    for root, dirs, files in os.walk(pth):
        for name in files:
            prefix = name.split(".")[0]
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





def triplet_sample(data, labels, anchor_data, anchor_labels):
    pos_data, neg_data = anchor_data.clone(), anchor_data.clone()
    for i, label in enumerate(anchor_labels):
        pos_index = np.random.choice(np.where(labels==label)[0])
        neg_index = np.random.choice(np.where(labels!=label)[0])
        pos_data[i] = data[pos_index]
        neg_data[i] = data[neg_index]
    return pos_data, neg_data
    
def accuracy(pos_dist, neg_dist):
    pred = (pos_dist - neg_dist).cpu().data
    return (pred > 0).sum().item()
    
def create_embedding_table(embedder, trX):
    return embedder(trX)
    
    
    


file2label, topic2label, n_labels = load_labels("labelsAll.lst")

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
    labels = torch.load("labels.pt").numpy()
    print data.shape, labels.shape

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
    b = 16

    trX = torch.load("trX.pt").cuda()
    trY = torch.load("trY.pt")
    nptrY = trY.numpy()
    
    #for i in range(len(trY)): trX[i,:,:10] = trY[i]

    loi = [topic2label["%s\n"%i] for i in [358, 349, 353, 356, 351, 340]]

    sidx = np.concatenate([np.where(nptrY == y)[0] for y in loi])
    strY = nptrY[sidx]

    teX = torch.load("teX.pt").cuda()
    teY = torch.load("teY.pt").cuda()
    
    #for i in range(len(teY)): teX[i,:,:10] = teY[i]
    
    trY = trY.cuda()
    
    trX = trX[:,:,:15000]
    teX = teX[:,:,:15000]
    
    trX = F.avg_pool1d(trX, 5)
    teX = F.avg_pool1d(teX, 5) # B x 13 x 3000
    
    mu = trX.mean(2, keepdim=True)
    trX -= mu
    mu = teX.mean(2, keepdim=True)
    teX -= mu
    
    std = trX.std(2, keepdim=True)
    trX /= std
    std = teX.std(2, keepdim=True)
    teX /= std
    
    embedder = Embedder(100).cuda()
    model = Triplet(embedder).cuda()
    
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in trainable_params])
    print "Number of parameters: {}".format(params)
    
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.MarginRankingLoss(0.2).cuda()
    
    target = torch.tensor((b,)).fill_(1).float().cuda()

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
            anchor_data, anchor_labels = trX[start:end], trY[start:end]
            
            pos_data, neg_data = triplet_sample(trX, nptrY, anchor_data, anchor_labels)
            
            """
            t_s = np.random.randint(0, anchor_data.shape[2]-length)
            anchor_data = anchor_data[:,:,t_s:t_s+length]
            
            t_s = np.random.randint(0, pos_data.shape[2]-length)
            pos_data = pos_data[:,:,t_s:t_s+length]
            
            t_s = np.random.randint(0, neg_data.shape[2]-length)
            neg_data = neg_data[:,:,t_s:t_s+length]
            """
            
            neg_dist, pos_dist, ea, en, ep = model(anchor_data, neg_data, pos_data)

            loss = criterion(neg_dist, pos_dist, target) #+ 1e-4 * (ea.norm(2) + en.norm(2) + ep.norm(2))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            acc += accuracy(neg_dist, pos_dist)
            n_points += len(anchor_labels)
            
        print "Train acc: ", acc/float(n_points)
        
        if i % 10 == 0:
            model.eval()
            acc = 0
            n_points = 0
            with torch.no_grad():
                emb_data = embedder(teX)
                emb_table = create_embedding_table(embedder, trX[sidx])
                
                for i in range(len(teY)):
                    e, y = emb_data[i], teY[i]
                    if y in loi:
                        dists = ((emb_table - e)**2).sum(dim=1).cpu().numpy()
                        pred = strY[np.argmin(dists)]
                        acc += (pred == y).item()
                        n_points += 1

                print n_points
                print "Test acc: ", acc / float(n_points)
            





