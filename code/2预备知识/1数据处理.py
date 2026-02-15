import torch

# ================================================
# 2.1.1 入门
# ================================================
print("2.1.1 入门")
# 张量中的每个值都称为张量的 元素（element）。例如，张量 x 中有 12 个元素。除非额外指定，新的张量将存储在内存中，并采用基于CPU的计算。
x = torch.arange(12)
print(f"x: {x}")

# shape属性来访问张量（沿每个轴的长度）的形状 
print(f"x.shape: {x.shape}")

# 如果只想知道张量中元素的总数，即形状的所有元素乘积，可以检查它的大小（size）。因为这里在处理的是一个向量，所以它的shape与它的size相同。
print(f"x.size(): {x.size()}")
print(f"x.numel(): {x.numel()}")

# 改变一个张量的形状而不改变元素数量和元素值
# 可以把张量x从形状为（12,）的行向量转换为形状为（3,4）的矩阵。
X = x.reshape(3, 4) # 可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。-1表示根据其他维度自动推断。
print(f"X: {X}")

# 创建一个形状为（2,3,4）的张量，其中所有元素都设置为0。
a = torch.zeros((2, 3, 4))
print(f"a: {a}")
# 创建一个形状为(2,3,4)的张量，其中所有元素都设置为1。
b = torch.ones((2, 3, 4))
print(f"b: {b}")
# 创建一个形状为(2,3,4)的张量，其中所有元素都设置为随机数。
c = torch.randn((2, 3, 4)) # 每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
print(f"c: {c}")

# 通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。
X1 = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"X1: {X1}")
print(f"X1.shape: {X1.shape}")

X2 = torch.tensor([[[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]]) # 三维张量
print(f"X2: {X2}")
print(f"X2.shape: {X2.shape}")

# ================================================
# 2.1.2 运算符
# ================================================
print(f"\n2.1.2 运算符")
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(f"x + y: {x + y}")
print(f"x - y: {x - y}")
print(f"x * y: {x * y}")
print(f"x / y: {x / y}")
print(f"x ** y: {x ** y}") 
print(f"torch.exp(x): {torch.exp(x)}")
print(f"torch.log(x): {torch.log(x)}")
print(f"torch.sin(x): {torch.sin(x)}")
print(f"torch.cos(x): {torch.cos(x)}")
print(f"torch.tan(x): {torch.tan(x)}")
print(f"torch.sqrt(x): {torch.sqrt(x)}")
print(f"torch.pow(x, 2): {torch.pow(x, 2)}")

# 把多个张量连结（concatenate）在一起， 把它们端对端地叠起来形成一个更大的张量。
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"torch.cat((X, Y), dim=0): \n{torch.cat((X, Y), dim=0)}") # 沿行（轴0，形状中的第一个元素）连结。结果形状为(6,4)。
print(f"torch.cat((X, Y), dim=1): \n{torch.cat((X, Y), dim=1)}") # 沿列（轴1，形状中的第二个元素）连结。结果形状为(3,8)。

# 通过逻辑运算符构建二元张量
print(f"X == Y: {X == Y}")
print(f"X != Y: {X != Y}")
print(f"X > Y: {X > Y}")

# 对张量中的所有元素进行求和，会产生一个单元素张量。
print(f"X.sum(): {X.sum()}")
print(f"X.sum(axis=0): {X.sum(axis=0)}")
print(f"X.sum(axis=1): {X.sum(axis=1)}")
print(f"X.sum(axis=0, keepdim=True): {X.sum(axis=0, keepdim=True)}")
print(f"X.sum(axis=1, keepdim=True): {X.sum(axis=1, keepdim=True)}")
print(f"X.sum(axis=0).shape: {X.sum(axis=0).shape}")
print(f"X.sum(axis=1).shape: {X.sum(axis=1).shape}")
print(f"X.sum(axis=0, keepdim=True).shape: {X.sum(axis=0, keepdim=True).shape}")
print(f"X.sum(axis=1, keepdim=True).shape: {X.sum(axis=1, keepdim=True).shape}")


# ================================================
# 2.1.3 广播机制
# ================================================
# 这种机制的工作方式如下：
# 1. 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状；
# 2. 对生成的数组执行按元素操作。
print(f"\n2.1.3 广播机制")
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(f"a: {a}") # 形状为(3,1)。
print(f"b: {b}") # 形状为(1,2)。
print(f"a + b: {a + b}") # 形状为(3,2)


# ================================================
# 2.1.4 索引和切片
# ================================================
print(f"\n2.1.4 索引和切片")
# 第一个元素的索引是0，最后一个元素索引是-1； 可以指定范围以包含第一个元素和最后一个之前的元素。
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
print(f"X: {X}")
print(f"X[1]: {X[1]}")
print(f"X[1, 2]: {X[1, 2]}")
print(f"X[0:2]: {X[0:2]}")
print(f"X[1:3]: {X[1:3]}")
print(f"X[::2]: {X[::2]}") # 每间隔一个元素，取一个元素。
print(f"X[1:3:2]: {X[1:3:2]}")

# 通过指定索引来将元素写入矩阵。
X[1, 2] = 9
print(f"X: {X}")

# 为多个元素赋值相同的值，我们只需要索引所有元素，然后为它们赋值。
X[0:2, :] = 12
print(f"X: {X}")


# ================================================
# 2.1.5 节省内存
# ================================================
print(f"\n2.1.5 节省内存")
# 运行一些操作可能会导致为新结果分配内存。例如，如果我们用Y = X + Y，我们将取消引用Y指向的张量，而是指向新分配的内存处的张量。
before = id(Y)
Y = Y + X
print(f"id(Y) == before: {id(Y) == before}")

# 执行原地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# 没有重复使用X， 我们也可以使用X[:] = X + Y或X += Y来减少操作的内存开销
before = id(X)
X += Y
print(f"id(X) == before: {id(X) == before}")

# ================================================
# 2.1.6 转换为其他Python对象
# ================================================
print(f"\n2.1.6 转换为其他Python对象")
A = X.numpy()
print(f"type(A): {type(A)}")
B = torch.tensor(A)
print(f"type(B): {type(B)}")

# 大小为1的张量转换为Python标量
a = torch.tensor([3.5])
print(f"a: {a}")
print(f"a.item(): {a.item()}")
print(f"float(a): {float(a)}")
print(f"int(a): {int(a)}")








