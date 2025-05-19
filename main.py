# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch_geometric.nn import GCNConv
import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.nn import GCNConv,GATConv
from torch import nn
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score, average_precision_score
from data_preprocessing import load_drug_mol_data
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import degree
from torch_geometric.nn import  GENConv




class GCNnet(torch.nn.Module):
    def __init__(self, in_channelsD, in_channelsM, hidden_channels, out_channels):
        super().__init__()
        #self.conv1 = GATConv(hidden_channels, hidden_channels,heads=4)
        self.conv1 = GATConv(hidden_channels, out_channels, heads=2)  #消融TF的时候GAT用这一层，别的不动，注释掉27行，然后下面把74，75行注释掉就行（正常情况用27行这一层GAT）
        self.conv2 = GCNConv(hidden_channels * 4, out_channels)
        self.fc1 = nn.Linear(out_channels * 4,out_channels * 2)
        self.fc2 = nn.Linear(out_channels * 2, 1)
        self.mlp1 = nn.Linear(in_channelsD,hidden_channels )
        self.mlp2 = nn.Linear(in_channelsM,hidden_channels)
        self.conv3 = GCNConv(58, hidden_channels )
        #self.convtf1 = TransformerConv(hidden_channels, hidden_channels, heads=4, dropout=0.1)  #消融GAT的时候用这一层，别的不动，注释掉36行和下面的72行（正常情况用36行这层）
        self.convtf1 = TransformerConv(hidden_channels*4, hidden_channels, heads=4, dropout=0.1)
        self.convtf2 = TransformerConv(hidden_channels * 4, out_channels, heads=2, dropout=0.1)
        #self.convtf3 = TransformerConv(out_channels * 2, out_channels, heads=1, dropout=0.1)
        self.bn1 = nn.BatchNorm1d(hidden_channels * 4)
        #deg = [56, 6534, 10683, 8420, 369]
        #deg = torch.tensor(deg)
        #self.fc11 = nn.Linear(6,58)
        self.conv11 = GATConv(58, hidden_channels,heads=1, edge_dim = 6)
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.mlpduiqi = nn.Linear(512,256)
        self.conv123123 = GCNConv(hidden_channels, out_channels*2)
    def forward(self, xd, xm, edge_index, edge_index_test):
        #max_degree = -1
        #for i in range(len(drug_smiles)):
        #    d = degree(drug_smiles[i][1][1], num_nodes=drug_smiles[i][0].shape[0], dtype=torch.long)
        #    max_degree = max(max_degree, int(d.max()))

        ## Compute the in-degree histogram tensor
        #deg = torch.zeros(max_degree + 1, dtype=torch.long)
        #for i in range(len(drug_smiles)):
        #    d = degree(drug_smiles[i][1][1], num_nodes=drug_smiles[i][0].shape[0], dtype=torch.long)
        #    deg += torch.bincount(d, minlength=deg.numel())

        vector = []
        for i in range(len(drug_smiles)):
            #edge_attr = drug_smiles[i][2]
            #temp = self.conv11(drug_smiles[i][0], drug_smiles[i][1], edge_attr).relu()
            temp = self.conv3(drug_smiles[i][0],drug_smiles[i][1]).relu()
            temp = torch.mean(temp, dim = 0).reshape(1,-1)
            vector.append(temp)
        matrix = torch.cat(vector,dim = 0)
        xd = self.mlp1(xd).relu()
        #xd = torch.cat([xd,matrix],dim = 1) #消融双视图的时候注释掉这一行
        #xd = self.mlpduiqi(xd).relu() #消融双视图的时候注释掉这一行
        #xd = (xd + matrix)/2    #消融双视图的加和时候注释掉这一行
        xm = self.mlp2(xm).relu()
        x = torch.cat([xd,xm],dim = 0)
        x = self.conv123123(x, edge_index).relu()

        #x = self.convtf1(x, edge_index).relu()
        #x = self.convtf2(x, edge_index)

        #x = self.bn(x)
        #x = self.convtf3(x, edge_index)
        #x = self.conv1(x, edge_index).relu()
        #x = self.conv2(x, edge_index)
        #scores = torch.cat([x[edge_index_test[0]], x[edge_index_test[1]]], dim=1)
        #scores = self.fc1(scores).relu()
        #scores = self.fc2(scores)
        #scoress = torch.sigmoid(scores)
        #scores = scores.reshape(-1)

        #features = x[edge_index_test[0]] * x[edge_index_test[1]]
        #features = torch.cat([x[edge_index_test[0]] , x[edge_index_test[1]]],dim = -1)
        scores = (x[edge_index_test[0]] * x[edge_index_test[1]]).sum(dim=-1)

        #if scoress.shape[0]<5000:
        #np.savetxt('scores.txt',scoress.detach().numpy())
        return scores

def train(model,optimizer,criterion,xd, xm, edge_index, edge_index_test,edge_index_test_label,istrain = bool):
    if istrain:

        model.train()
        optimizer.zero_grad()
        out = model.forward(xd, xm, edge_index, edge_index_test)
        out2 = 1 / (1 + torch.exp(-out))
        #out = model.forward(xd, xm, edge_index, edge_index_test).squeeze(1)
        out1 = out2.cpu().detach().numpy()
        pred = (out1 >= 0.5).astype(int)
        #print(out.shape,edge_index_test_label.shape)
        #print(out.shape,edge_index_test_label.shape)
        loss = criterion(out, edge_index_test_label)
        acc = metrics.accuracy_score(edge_index_test_label, pred)
        f1_score = metrics.f1_score(edge_index_test_label.cpu().detach().numpy(), pred)
        precision = metrics.precision_score(edge_index_test_label.cpu().detach().numpy(), pred)
        recall = metrics.recall_score(edge_index_test_label.cpu().detach().numpy(), pred)
        loss.backward()
        optimizer.step()
    else:
        #model.train()
        model.eval()
        #torch.save(edge_index_test_label, 'label2.pt')
        out = model.forward(xd, xm, edge_index, edge_index_test)
        out2 = 1 / (1 + torch.exp(-out))
        #out = model.forward(xd, xm, edge_index, edge_index_test).squeeze(1)
        out1 = out2.cpu().detach().numpy()
        pred = (out1 >= 0.5).astype(int)
        loss = criterion(out, edge_index_test_label)
        acc = metrics.accuracy_score(edge_index_test_label, pred)
        f1_score = metrics.f1_score(edge_index_test_label.cpu().detach().numpy(), pred)
        precision = metrics.precision_score(edge_index_test_label.cpu().detach().numpy(), pred)
        recall = metrics.recall_score(edge_index_test_label.cpu().detach().numpy(), pred)
        #loss.backward()
        #optimizer.step()
    return loss,roc_auc_score(edge_index_test_label.cpu().detach().numpy(), out2.cpu().detach().numpy()),f1_score,precision,recall,acc,average_precision_score(edge_index_test_label.cpu().detach().numpy(), out2.cpu().detach().numpy())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    drug_sim_feature = torch.tensor(pd.read_csv(r'D:\code\DMI\data\drug_sim1.csv').values).float()
    microb_sim_feature = torch.tensor(pd.read_csv(r'D:\code\DMI\data\microb_sim1.csv').values).float()
    drug_microb_adj = pd.read_csv(r'D:\code\DMI\data\drug_microb1.csv').values
    #drug_smiles = pd.read_csv(r'D:\code\DMI\data\drug_smiles1.csv')
    drug_smiles = load_drug_mol_data()
    drug_microb = np.block([[np.zeros((drug_microb_adj.shape[0],drug_microb_adj.shape[0])), drug_microb_adj],
                       [np.transpose(drug_microb_adj), np.zeros((drug_microb_adj.shape[1],drug_microb_adj.shape[1]))]])
    #np.savetxt('dataset1.txt', drug_microb)
    #print(drug_microb.shape)
    edges = []
    num_nodes = drug_microb.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if drug_microb[i, j] == 1:
                edges.append([i, j])
    edges = np.array(edges)
    neg_sample = []
    while True:
        neg = (np.random.randint(drug_microb_adj.shape[0]), np.random.randint(drug_microb_adj.shape[1]))
        if drug_microb_adj[neg] == 0:
            if neg not in neg_sample:
                neg_sample.append(neg)
            if len(neg_sample) == edges.shape[0]*1:
                break
    neg_sample = np.array(neg_sample)
    neg_sample[:,1] += 999
    sample_all = np.vstack((edges,neg_sample))
    np.savetxt(r'D:\code\DMIbidui\Graph2MDA-master\data\data\adjd1', edges)
    sample_all = torch.Tensor(sample_all).long()
    label_all = np.hstack((np.ones(edges.shape[0]),np.zeros(neg_sample.shape[0])))
    label_all = torch.Tensor(label_all)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)
    j = 0
    for train_idx, test_idx in stratified_kfold.split(sample_all, label_all):
        result = []
        X_train, X_test = sample_all[train_idx], sample_all[test_idx]
        y_train, y_test = label_all[train_idx], label_all[test_idx]
        temp = []
        temp1 = []
        for i in range(X_test.shape[0]):
            if X_test[i][0] in X_train[:,0] and X_test[i][1] in X_train[:,1]:
                temp.append(np.array(X_test[i][:]))
                temp1.append(np.array(y_test[i]))

        X_test = torch.tensor(np.array(temp))
        y_test = torch.tensor(np.array(temp1))


        '''X_train = torch.cat([X_train,X_test],dim = 0)
        y_train = torch.cat([y_train,y_test], dim = -1)
        X_traintemp = X_train.tolist()
        list=[]
        for i in range(176):
            for j in range(76):
                list.append([i,j+176])
        test = [row for row in list if row not in X_traintemp]
        test1 = np.array(test)
        np.savetxt('index.txt', test1)
        y_test = np.hstack((np.ones(int(len(test)/2)),np.zeros(int(len(test)/2))))
        X_test = torch.tensor(test)
        y_test = torch.tensor(y_test)'''



        model = GCNnet(drug_sim_feature.shape[1],microb_sim_feature.shape[1],256,128)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        for epoch in range(1, 200):
            loss,train_auc,f1,p,r,acc,aupr = train(model,optimizer,criterion,drug_sim_feature, microb_sim_feature, X_train.T, X_train.T,y_train,True)
            testloss, test_auc, testf1, testp, testr, testacc,testaupr = train(model,optimizer,criterion,drug_sim_feature, microb_sim_feature, X_train.T, X_test.T,y_test,False)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f},Trainauc: {train_auc:.4f},Trainacc: {acc:.4f},Trainaupr: {aupr:.4f}')
            print(f'F1: {f1:.4f},p: {p:.4f},r: {r:.4f}')
            print(f' testLoss: {testloss:.4f},testauc: {test_auc:.4f},Testacc: {testacc:.4f},Testaupr: {testaupr:.4f}')
            print(f'testF1: {testf1:.4f},testp: {testp:.4f},testr: {testr:.4f}')
            result.append([test_auc,testaupr])
        #break
        j = j + 1
        np.savetxt('12/result'+ str(j) +'.txt', result)  #数据保存在3那个文件夹下


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
