# ================================================
# 2.2.1 读取数据集
# ================================================
print("2.2.1 读取数据集")
import os
os.makedirs(os.path.join(f'code/2预备知识', 'data'), exist_ok=True)
data_file = os.path.join(f'code/2预备知识', 'data', 'house_tiny.csv')
print(data_file)
# 创建一个csv文件，并写入数据
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本。NA表示缺失值。
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 调用read_csv函数
import pandas as pd
data = pd.read_csv(data_file)
print(f"data: \n{data}")

# ================================================
# 2.2.2 处理缺失值
# ================================================
print(f"\n2.2.2 处理缺失值")
# 典型的方法包括插值法和删除法， 其中插值法用一个替代值弥补缺失值，而删除法则直接忽略缺失值。
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(f"inputs: \n{inputs}")

# 对于inputs中的类别值或离散值，我们将“NaN”视为一个类别。 
# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”， pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int) # dtype=int将 True/False 转换为 1/0。
print(f"inputs: \n{inputs}")

# ================================================
# 2.2.3 转换为张量格式
# ================================================
print(f"\n2.2.3 转换为张量格式")
import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
y1 = torch.tensor(outputs.to_numpy(dtype=float))
print(f"X: \n{X}")
print(f"y: \n{y1}")

X, y2 = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(f"\nX: \n{X}")
print(f"y: \n{y2}")

print(f"y1 == y2: {y1 == y2}")










