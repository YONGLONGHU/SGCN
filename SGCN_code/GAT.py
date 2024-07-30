import torch
from torch import Tensor
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

# 加载Cora数据集
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]
print(data)

class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=8,concat=False)
        self.conv2 = GATConv(in_feats, out_feats,heads=8,concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(self.conv2(x, edge_index), dim=1)

        return x

model = GAT(dataset.num_features, 16, dataset.num_classes, 4)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def test():
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = test_correct.sum().item() / data.test_mask.sum().item()
    return test_acc

for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')