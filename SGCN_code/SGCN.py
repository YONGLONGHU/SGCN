import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv1
from torch_geometric.datasets import Planetoid
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
# 记录开始时间
start_time = time.time()
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]
print(data)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv1(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 128, dataset.num_classes)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    c=out[data.train_mask]
    print(c)
    loss.backward()
    optimizer.step()
    return loss

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    c = out[pred[data.test_mask]]
    print(c)
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc
testAcc = []
losses = []
for epoch in range(1, 301):
    loss = train()
    print(f'train loss: {loss:.4f}')
    losses.append(loss.item())
    test_acc = test()
    testAcc.append(test_acc)
    print(f'Test Accuracy: {test_acc:.4f}')
# 记录结束时间
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
