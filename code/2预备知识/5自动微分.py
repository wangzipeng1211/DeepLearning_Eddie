import torch

# ================================================
# 2.5.1 一个简单的例子
# ================================================
print("2.5.1 一个简单的例子")
x = torch.arange(4.0)
print(f"x: {x}")

# 在计算关于x的梯度之前，需要一个地方来存储梯度。
x.requires_grad_(True)     # 等价于x=torch.arange(4.0,requires_grad=True)
print(f"x.grad: {x.grad}") # 默认值为None

# 现在计算y
y = 2 * torch.dot(x, x)
print(f"y: {y}")

# 通过调用反向传播函数来自动计算y关于x每个分量的梯度。
y.backward()
print(f"x.grad: {x.grad}")

# 验证结果
print(f"x.grad == 4 * x: {x.grad == 4 * x}")

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值。
x.grad.zero_()

# 现在计算x的另一个函数。
y = x.sum()
print(f"y: {y}")
y.backward()
print(f"x.grad: {x.grad}")

# 验证结果
print(f"x.grad == 1: {x.grad == 1}")


# ================================================
# 2.5.2 非标量变量的反向传播
# ================================================
print("\n2.5.2 非标量变量的反向传播")
x.grad.zero_()
print(f"x: {x}")

y = x * x
print(f"y: {y}")

y.sum().backward() # 等价于y.backward(torch.ones(len(y)))
print(f"x.grad: {x.grad}")


# ================================================
# 2.5.3 分离计算
# ================================================
print("\n2.5.3 分离计算")
# 将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
print(f"y: {y}")
print(f"u: {u}")
print(f"z: {z}")

z.sum().backward()
print(f"x.grad: {x.grad}")
print(f"x.grad == u: {x.grad == u}")

# 随后在y上调用反向传播
x.grad.zero_()
y.sum().backward()
print(f"x.grad: {x.grad}")
print(f"x.grad == 2 * x: {x.grad == 2 * x}")


# ================================================
# 2.5.4 Python控制流的梯度计算
# ================================================
print("\n2.5.4 Python控制流的梯度计算")
# 使用自动微分的一个好处是：
# 即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


a = torch.randn(size=(), requires_grad=True)
print(f"a: {a}")
d = f(a)
print(f"d: {d}")
d.backward()
print(f"a.grad: {a.grad}")
print(f"a.grad == d / a: {a.grad == d / a}")




