# -*- coding: utf-8 -*-
# @Time         : 2022/7/23 17:39
# @Author       : Yufan Liu
# @Description  : Main training
from model import GCNModel, GraphCombine
import torch
import torch.nn as nn
from data import ProteinGraph
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef

DEVICE = 'cuda'
EPOCHS = 500

model = GCNModel().to(DEVICE)
# model = GraphCombine().to(DEVICE)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 17])).to(DEVICE)

# train TR640 & TE639
train_data = ProteinGraph(data_name='TR1154', root='./Dataset_pkl')
test_data = ProteinGraph(data_name='TE125', root='./Dataset_pkl')

train_batch = DataLoader(train_data, batch_size=64)
test_batch = DataLoader(test_data, batch_size=9999999)

for epoch in range(EPOCHS):
    loss_list = []
    for batch in train_batch:
        batch.to(DEVICE)
        model.train()
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

    if np.mean(loss_list) < 0.005:
        break
    print(f"Epoch {epoch + 1} / {EPOCHS}, loss {np.mean(loss_list)}")

# only calculate metrics for positive labels
for batch in test_batch:
    model.to('cpu')
    model.eval()
    predicted_value = model(batch.x, batch.edge_index)
    prediction = predicted_value.argmax(1)

    print("AUC Value:")
    print(roc_auc_score(batch.y.tolist(), predicted_value.detach().numpy()[:, 1]))

    print("Recall, Precision Value:")
    print(classification_report(batch.y.tolist(), prediction.tolist()))

    print("Specificity Value:")
    tn, fp, fn, tp = confusion_matrix(batch.y.tolist(), prediction.tolist()).ravel()
    print(tn / (tn + fp))  # 检测比例较高的负类别是否真的有意义？

    print("MCC Value:")
    print(matthews_corrcoef(batch.y.tolist(), prediction.tolist()))