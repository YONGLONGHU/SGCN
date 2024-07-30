import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.datasets import Planetoid

# 1.加载Cora数据集
dataset = Planetoid(root='data/', name='Pubmed')


# 2.定义GraphSAGE网络
class GraphSAGE(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = pyg_nn.SAGEConv(num_node_features, 128)
        self.conv2 = pyg_nn.SAGEConv(128, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
epochs = 300  # 学习轮数
lr = 0.01  # 学习率
num_node_features = dataset.num_node_features  # 每个节点的特征数
num_classes = dataset.num_classes  # 每个节点的类别数
data = dataset[0].to(device)  # Cora的一张图

# 3.定义模型
model = GraphSAGE(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
loss_function = nn.NLLLoss()  # 损失函数

# 训练模式
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(data)

    loss = loss_function(pred[data.train_mask], data.y[data.train_mask])  # 损失
    correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item()  # epoch正确分类数目
    acc_train = correct_count_train / data.train_mask.sum().item()  # epoch训练精度

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("【EPOCH: 】%s" % str(epoch + 1))
        print('训练损失为：{:.4f}'.format(loss.item()), '训练精度为：{:.4f}'.format(acc_train))

print('【Finished Training！】')

# 模型验证
model.eval()
pred = model(data)

# 训练集（使用了掩码）
correct_count_train = pred.argmax(axis=1)[data.train_mask].eq(data.y[data.train_mask]).sum().item()
acc_train = correct_count_train / data.train_mask.sum().item()
loss_train = loss_function(pred[data.train_mask], data.y[data.train_mask]).item()

# 测试集
correct_count_test = pred.argmax(axis=1)[data.test_mask].eq(data.y[data.test_mask]).sum().item()
acc_test = correct_count_test / data.test_mask.sum().item()
loss_test = loss_function(pred[data.test_mask], data.y[data.test_mask]).item()

#print('Train Accuracy: {:.4f}'.format(acc_train), 'Train Loss: {:.4f}'.format(loss_train))
print('Test  Accuracy: {:.4f}'.format(acc_test), 'Test  Loss: {:.4f}'.format(loss_test))
