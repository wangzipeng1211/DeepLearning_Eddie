import random

import torch
from d2l import torch as d2l


# ================================================
# 3.2.1 生成数据集
# ================================================
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）。
print('features:', features[0],'\nlabel:', labels[0])

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)


# ================================================
# 3.2.2 读取数据集
# ================================================
# 定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据。
# 定义一个data_iter函数， 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。 
# 每个小批量包含一组特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 读取第一个小批量数据样本并打印。 
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


# ================================================
# 3.2.3 初始化模型参数
# ================================================
# 通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重， 并将偏置初始化为0。
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# 在初始化参数之后，我们的任务是更新这些参数，直到这些参数足够拟合我们的数据。 
# 每次更新都需要计算损失函数关于模型参数的梯度。 
# 有了这个梯度，我们就可以向减小损失的方向更新每个参数。 


# ================================================
# 3.2.4 定义模型
# ================================================
# 定义模型，将模型的输入和参数同模型的输出关联起来。
def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b


# ================================================
# 3.2.5 定义损失函数
# ================================================
def squared_loss(y_hat, y):  #@save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# ================================================
# 3.2.6 定义优化算法
# ================================================
# 在每一步中，使用从数据集中随机抽取的一个小批量，然后根据参数计算损失的梯度。 
# 接下来，朝着减少损失的方向更新我们的参数。 下面的函数实现小批量随机梯度下降更新。 
# 该函数接受模型参数集合、学习速率和批量大小作为输入。
# 每一步更新的大小由学习速率lr决定。 
# 因为我们计算的损失是一个批量样本的总和，所以我们用批量大小（batch_size） 来规范化步长，这样步长大小就不会取决于我们对批量大小的选择
# 在每个迭代周期（epoch）中，我们使用data_iter函数遍历整个数据集， 并将训练数据集中所有样本都使用一次（假设样本数能够被批量大小整除）。
def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ================================================
# 3.2.7 训练
# ================================================
# 在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。 
# 计算完损失后，我们开始反向传播，存储每个参数的梯度。 
# 最后，我们调用优化算法sgd来更新模型参数。
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
