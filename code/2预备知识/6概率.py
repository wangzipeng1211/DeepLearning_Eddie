import torch
from torch.distributions import multinomial
from d2l import torch as d2l

# ================================================
# 2.6.1 基本概率论
# ================================================
print("2.6.1 基本概率论")
# 为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。 输出是另一个相同长度的向量：它在索引i处的值是采样结果中i出现的次数。
fair_probs = torch.ones([6]) / 6
print(f"multinomial.Multinomial(1, fair_probs).sample(): {multinomial.Multinomial(1, fair_probs).sample()}")

# 深度学习框架的函数同时抽取多个样本，得到我们想要的任意形状的独立样本数组。
print(f"multinomial.Multinomial(1000, fair_probs).sample(): {multinomial.Multinomial(1000, fair_probs).sample()}")

# 计算相对频率，以作为真实概率的估计
estimates = multinomial.Multinomial(1000, fair_probs).sample()
print(f"estimates: {estimates}")
print(f"estimates / 1000: {estimates / 1000}")  # 相对频率作为估计值

# 可以看到这些概率如何随着时间的推移收敛到真实概率。 让我们进行1000组实验，每组抽取10个样本。
counts = multinomial.Multinomial(10, fair_probs).sample((1000,))
print(f"counts: {counts}")
cum_counts = counts.cumsum(dim=0)
print(f"cum_counts: {cum_counts}")
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
print(f"estimates: {estimates}")

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(), label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()
d2l.plt.show()


