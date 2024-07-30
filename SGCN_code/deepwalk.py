import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec

# 1.加载Cora数据集
dataset = Planetoid(root='data/', name='Cora')
data = dataset[0]

# 2.定义模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

# deepwalk模型
model = Node2Vec(edge_index=data.edge_index,
                 embedding_dim=128,  # 节点维度嵌入长度
                 walk_length=5,  # 序列游走长度
                 context_size=4,  # 上下文大小
                 walks_per_node=1,  # 每个节点游走1个序列
                 p=1,
                 q=1,
                 sparse=True  # 权重设置为稀疏矩阵
                 ).to(device)

# 迭代器
loader = model.loader(batch_size=128, shuffle=True)
# 优化器
optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

# 3.开始训练
model.train()

for epoch in range(1, 401):
    total_loss = 0  # 每个epoch的总损失
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))  # 计算损失
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # 使用逻辑回归任务进行测试生成的embedding效果
    with torch.no_grad():
        model.eval()  # 开启测试模式
        z = model()  # 获取权重系数，也就是embedding向量表

        # z[data.train_mask] 获取训练集节点的embedding向量
        acc = model.test(z[data.train_mask], data.y[data.train_mask],
                         z[data.test_mask], data.y[data.test_mask],
                         max_iter=150)  # 内部使用LogisticRegression进行分类测试

    # 打印指标
    print(f'Epoch: {epoch:04d}, Loss: {total_loss:.4f}, Acc: {acc:.4f}')

# 可视化节点的embedding
with torch.no_grad():
    # 不同类别节点对应的颜色信息
    colors = [
        '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535',
        '#ffd700'
    ]

    model.eval()  # 开启测试模式
    # 获取节点的embedding向量，形状为[num_nodes, embedding_dim]
    z = model(torch.arange(data.num_nodes, device=device))
    # 使用TSNE先进行数据降维，形状为[num_nodes, 2]
    z = TSNE(n_components=2).fit_transform(z.detach().numpy())
    y = data.y.detach().numpy()

    plt.figure(figsize=(8, 8))

    # 绘制不同类别的节点
    for i in range(dataset.num_classes):
        # z[y==0, 0] 和 z[y==0, 1] 分别代表第一个类的节点的x轴和y轴的坐标
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.show()
