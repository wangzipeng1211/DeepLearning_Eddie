from warnings import filterwarnings
filterwarnings("ignore")

# ================================================
# 2.3.1 标量
# ================================================
import torch
print("2.3.1 标量")
# 仅包含一个数值被称为标量（scalar）
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(f"x + y: {x + y}")
print(f"x - y: {x - y}")
print(f"x * y: {x * y}")
print(f"x / y: {x / y}")
print(f"x ** y: {x ** y}")


# ================================================
# 2.3.2 向量
# ================================================
print("\n2.3.2 向量")
# 向量可以被视为标量值组成的列表。 这些标量值被称为向量的元素（element）或分量（component）。
x = torch.arange(4)
print(f"x: {x}")
print(f"x[3]: {x[3]}") # 通过张量的索引来访问任一元素
print(f"length: {len(x)}") # 张量的长度
print(f"shape: {x.shape}") # 形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。 对于只有一个轴的张量，形状只有一个元素。


# ================================================
# 2.3.3 矩阵
# ================================================
print("\n2.3.3 矩阵")
# 通过指定两个分量 m 和 n 来创建一个形状为 m*n 的矩阵。
A = torch.arange(20).reshape(5, 4)
print(f"A: {A}")
print(f"A.shape: {A.shape}")
print(f"A.size(): {A.size()}")
print(f"A.numel(): {A.numel()}")
print(f"A.T: {A.T}") # 矩阵的转置

# 对称矩阵
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f"对称矩阵 B: \n{B}")
print(f"B==B.T: \n{B == B.T}")


# ================================================
# 2.3.4 张量
# ================================================
print("\n2.3.4 张量")
# 向量是标量的推广，矩阵是向量的推广一样，我们可以构建具有更多轴的数据结构。
X = torch.arange(24).reshape(2, 3, 4)
print(f"X: {X}")
print(f"X.shape: {X.shape}")
print(f"X.size(): {X.size()}")
print(f"X.numel(): {X.numel()}")
print(f"X.T: {X.T}")
print(f"X.permute(2, 1, 0): {X.permute(2, 1, 0)}") # 显式指定维度反转
print(f"X.mT:{X.mT}") # 对最后两个维度进行矩阵转置


# ================================================
# 2.3.5 张量算法的基本性质
# ================================================
print("\n2.3.5 张量算法的基本性质")
# 给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone() # 通过分配新内存，将A的一个副本分配给B。
print(f"A: \n{A}")
print(f"B: \n{B}")
print(f"A + B: \n{A + B}")
print(f"A * B: \n{A * B}") # 两个矩阵的按元素乘法称为Hadamard积（Hadamard product）

# 张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(f"a + X: \n{a + X}")
print(f"a * X: \n{a * X}")


# ================================================
# 2.3.6 降维
# ================================================
print("\n2.3.6 降维")
# 求和函数
x = torch.arange(4, dtype=torch.float32)
print(f"x: {x}")
print(f"x.sum(): {x.sum()}")
# 表示任意形状张量的元素和
print(f"A: \n{A}")
print(f"A.sum(): {A.sum()}")

# 默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。
# 为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定axis=0。 由于输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失。
C = torch.arange(40).reshape(2, 5, 4)
print(f"C: \n{C}")
print(f"C.sum(axis=0): \n{C.sum(axis=0)}")
print(f"C.sum(axis=0).shape: {C.sum(axis=0).shape}")

# 指定axis=1将通过汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失。
print(f"C.sum(axis=1): \n{C.sum(axis=1)}")
print(f"C.sum(axis=1).shape: {C.sum(axis=1).shape}")

# 指定axis=2将通过汇总所有页的元素降维（轴2）。因此，输入轴2的维数在输出形状中消失。
print(f"C.sum(axis=2): \n{C.sum(axis=2)}")
print(f"C.sum(axis=2).shape: {C.sum(axis=2).shape}")

# 指定axis=[0,1]将通过汇总所有页的元素降维（轴0和轴1）。因此，输入轴0和轴1的维数在输出形状中消失。
print(f"C.sum(axis=[0,1]): \n{C.sum(axis=[0,1])}")
print(f"C.sum(axis=[0,1]).shape: {C.sum(axis=[0,1]).shape}")

# 一个与求和相关的量是平均值（mean或average）
print(f"A.mean(): {A.mean()}")
print(f"A.sum() / A.numel(): {A.sum() / A.numel()}")

# 同样，计算平均值的函数也可以沿指定轴降低张量的维度。
print(f"A.mean(axis=0): {A.mean(axis=0)}")
print(f"A.sum(axis=0) / A.shape[0]: {A.sum(axis=0) / A.shape[0]}")

# 非降维求和
print(f"A.sum(axis=1, keepdim=True): \n{A.sum(axis=1, keepdim=True)}")
print(f"A.sum(axis=1, keepdim=True).shape: {A.sum(axis=1, keepdim=True).shape}")

# 可以通过广播将A除以sum_A。
print(f"A / A.sum(axis=1, keepdim=True): \n{A / A.sum(axis=1, keepdim=True)}")

# 沿某个轴计算A元素的累积总和， 比如axis=0（按行计算），可以调用cumsum函数
print(f"A.cumsum(axis=0): \n{A.cumsum(axis=0)}")


# ================================================
# 2.3.7 点积
# ================================================
print("\n2.3.7 点积")
# 点积是相同位置的按元素乘积的和。
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(f"x: {x}")
print(f"y: {y}")
print(f"x.dot(y): {x.dot(y)}")

# 通过执行按元素乘法，然后进行求和来表示两个向量的点积
print(f"torch.sum(x * y): {torch.sum(x * y)}")


# ================================================
# 2.3.8 矩阵-向量积
# ================================================
print("\n2.3.8 矩阵-向量积")
# 矩阵
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"A: \n{A}")
# 向量
x = torch.arange(4, dtype=torch.float32)
print(f"x: {x}")
# 矩阵-向量积
print(f"A * x: \n{A * x}")
print(f"torch.mv(A, x): \n{torch.mv(A, x)}")


# ================================================
# 2.3.9 矩阵-矩阵乘法
# ================================================
print("\n2.3.9 矩阵-矩阵乘法")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 4, dtype=torch.float32)
print(f"A: \n{A}")
print(f"B: \n{B}")
# 矩阵-矩阵乘法
print(f"torch.mm(A, B): \n{torch.mm(A, B)}")
print(f"torch.matmul(A, B): \n{torch.matmul(A, B)}")
print(f"A @ B: \n{A @ B}")


# ================================================
# 2.3.10 范数
# ================================================
print("\n2.3.10 范数")
# 范数是向量空间中的距离度量。
u = torch.tensor([3.0, -4.0])
print(f"u: {u}")
# L2范数是向量元素的平方和的平方根
print(f"torch.norm(u): {torch.norm(u)}")
print(f"torch.norm(u, p=2): {torch.norm(u, p=2)}")
# L1范数是向量元素的绝对值之和
print(f"torch.norm(u, p=1): {torch.norm(u, p=1)}")
print(f"torch.abs(u).sum(): {torch.abs(u).sum()}")
# Frobenius范数是矩阵元素的平方和的平方根
print(f"torch.norm(torch.ones((4, 9), p='fro')): {torch.norm(torch.ones((4, 9)), p='fro')}")



