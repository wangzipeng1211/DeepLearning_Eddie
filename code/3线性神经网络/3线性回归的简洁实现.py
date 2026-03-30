import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# ================================================
# 3.3.1 生成数据集
# ================================================
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
print('features:', features[0],'\nlabel:', labels[0])


# ================================================
# 3.3.2 读取数据集
# ================================================
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 使用iter构造Python迭代器，并使用next从迭代器中获取第一项。
print(next(iter(data_iter)))


# ================================================
# 3.3.3 定义模型
# ================================================
# nn是神经网络的缩写
from torch import nn

# Sequential类将多个层串联在一起。 
# 当给定输入数据时，Sequential实例将数据传入到第一层， 然后将第一层的输出作为第二层的输入，以此类推。 
# 在下面的例子中，我们的模型只包含一个层，因此实际上不需要Sequential。 
# 但是由于以后几乎所有的模型都是多层的，在这里使用Sequential会让你熟悉“标准的流水线”。
net = nn.Sequential(nn.Linear(2, 1))


# ================================================
# 3.3.4 初始化模型参数
# ================================================
# 通过net[0]选择网络中的第一个图层， 然后使用weight.data和bias.data方法访问参数。 
# 我们还可以使用替换方法normal_和fill_来重写参数值。
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


# ================================================
# 3.3.5 定义损失函数    
# ================================================
loss = nn.MSELoss() # 均方误差使用的是MSELoss类


# ================================================
# 3.3.6 定义优化算法    
# ================================================
trainer = torch.optim.SGD(net.parameters(), lr=0.02)


# ================================================
# 3.3.7 训练
# ================================================
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 要访问参数，我们首先从net访问所需的层，然后读取该层的权重和偏置。
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

