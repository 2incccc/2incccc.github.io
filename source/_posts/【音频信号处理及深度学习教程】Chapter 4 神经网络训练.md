---
title: 【深度学习教程】Chapter 4 神经网络训练
date: 2023-07-15 01:23:05
tags: [音频信号,深度学习,声学,神经网络,信号处理,人工智能]
categories: 
- 音频信号处理及深度学习教程 
math: true
---
## Chapter 4  神经网络训练

### 神经网络训练
训练的过程是反向传播的过程，利用得到的输出值与预测值的偏差，（损失函数），反向更新模型中的参数

### 逼近的思路理解训练
随机生成一个三阶函数，赋予一组随机参数，得到的输出与 sine 输出值比较，差值 loss 最小的那一组参数就为目标函数。
- 流程：
  1. 根据预测值和标签值得到 loss
  2. Loss 函数对各个参数反向求偏导
  3. 计算每个参数的梯度
  4. 更新参数值
  5. 梯度置 0
  6. 再次循环
<!--more-->
#### 反向传播

![16893535815091689353580690.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16893535815091689353580690.png)



#### 高阶函数构造 sin(x) c

1. 常规思路：loss 对各参数求偏导，计算梯度，更新梯度值，梯度置 0

```python
"""用一个三阶函数找到合适的参数 逼近y=sinx
# 1.构建三阶函数
# 2.给定输入，得到该函数的输出值,共循环500次
# 3.得到该函数的输出值与y=sinx的输出 偏差loss函数
# 4.为了得到loss最小，求该函数的极小值（导数）
# 5.根据梯度值，更新参数
"""

import numpy as np

# 0.sine
x = np.linspace(start=-np.pi,stop=np.pi,num=2000)
y = np.sin(x)

a,b,c,d = np.random.rand(),np.random.rand(),np.random.rand(),np.random.rand()
learning_rate = 1e-6
# 学习率（learning rate）用于控制参数更新的步长。它决定了每一步更新中参数的变化量。
for epoch in range(200000):
    y_pred = a + b*x + c*x**2 + d*x**3
    loss = np.square(y_pred - y).sum()
    grad_y_pre = 2 * (y_pred - y)
    grad_a = grad_y_pre.sum()
    grad_b = (grad_y_pre * x**1).sum()
    grad_c = (grad_y_pre * x**2).sum()
    grad_d = (grad_y_pre * x**3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d
    # 以上为求导过程
    # 根据链式法则，损失函数关于b的偏导数可以表示为：∂loss/∂b = ∂loss/∂y_pred * ∂y_pred/∂b
    if epoch%20 == 0:
    # Epoch（时期）是指将整个训练数据集（dataset）通过神经网络进行前向传播和反向传播的一次完整迭代。
        print(loss)

print(f"y_pred = {a.item()} + {b.item()}*x + {c.item()}*x^2 + {d.item()}*x^3")
```

2. loss.backward():由该函数确定更新后的参数值 （这是一个 PyTorch 库中的函数，输入输出需要为张量）

```python
"""用一个三阶函数找到合适的参数 逼近y=sinx
# 1.构建三阶函数
# 2.给定输入，得到该函数的输出值,共循环500次
# 3.得到该函数的输出值与y=sinx的输出 偏差loss函数
# 4.为了得到loss最小，求该函数的极小值（导数）
# 5.根据梯度值，更新参数
"""

import torch

# 0.sine
x = torch.linspace(start=-torch.pi,end=torch.pi,steps=2000)
y = torch.sin(x)

a,b,c,d = torch.rand((),requires_grad=True),torch.rand((),requires_grad=True),\
          torch.rand((),requires_grad=True),torch.rand((),requires_grad=True)

learning_rate = 1e-6
for epoch in range(2000):
    y_pred = a + b*x + c*x**2 + d*x**3
    loss = torch.square((y_pred - y),).sum()
    debug = 1
    loss.backward() # 使用该函数，代替求导的过程

    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
    debug = 1
    if epoch%20 == 0:
        print(loss)  

print(f"y_pred = {a.item()} + {b.item()}*x + {c.item()}*x^2 + {d.item()}*x^3")
```

3. optimiser.step(): 优化器函数（同上）

```python
"""用一个网络模型 逼近y=sinx
# 1.给定输入，得到sin的输出值为y
# 2.给定输入，根据y=a+b*x+c*x**2+d*x**3,计算^1,^2,^3不同幂次下的结果
# 3.构建网络模型，利用线性层将不同幂次下的结果按一定权重相加，包含线性层Linear(3,1),Flatten()
# 4.将三个结果放入模型得到该函数的输出值,共循环500->2000次
# 4.得到该函数的输出值与y=sinx的输出,偏差loss函数= torch.square(y_pre - y).sum()
# 5.为了得到loss最小，求该函数的极小值（导数）loss.backward()
# 6.根据梯度值，更新参数 param -= learning_rate * param.grad 之后 model.zero_grad()
"""
from torch import nn
import torch

class Liner(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(3,1),nn.Flatten(0,1))
    def forward(self,data):
        output = self.model(data)
        return output

x = torch.linspace(-torch.pi,torch.pi,2_000)
y = torch.sin(x)

# x^1,x^2,x^3
mynn = Liner()
p = torch.tensor([1,2,3])
input = x.unsqueeze(-1).pow(p)
learning_rate = 1e-4
optimiser = torch.optim.RMSprop(params=mynn.model.parameters(),lr=learning_rate)
loss_fn = torch.nn.MSELoss() # 代替square平方求和
for epoch in range(2_000):
    y_pre = mynn(input)
    loss = loss_fn(y_pre,y)
    layer_liner = mynn.model[0]
    layer_flatten = mynn.model[1]
    debug = 1
    optimiser.zero_grad() # 用于将模型参数的梯度归零。
    loss.backward() # 反向更新参数
    optimiser.step() # 优化器，代替-=的过程

    if epoch % 10==0:
        print(loss)
    debug = 1

debug = 1
```

**需要注意：**
`optimiser = torch.optim.RMSprop(params=mynn.model.parameters(),lr=learning_rate)` 定义了一个优化器，参数是模型中的各个参数，`lr` 是学习率。在机器学习和深度学习中，优化器（Optimizer）是一种用于调整模型参数以最小化损失函数的算法或方法。优化器根据模型的梯度信息和指定的优化算法，更新模型参数的值，以便使损失函数达到最小值或接近最小值。在训练神经网络模型的过程中，优化器的作用非常重要。它能够根据损失函数的梯度信息来更新模型参数，使得模型能够逐步调整自身以更好地拟合训练数据。
`loss.Backward()` 的任务是执行反向传播计算梯度。具体来说，它计算损失函数 loss 关于模型参数的梯度，通过使用链式法则将梯度从损失函数传播到模型的每个参数。这样可以获得每个参数相对于损失函数的梯度信息，即参数的更新方向和大小。
`optimiser.step()` 的任务是根据梯度信息更新模型参数的值。它使用优化算法（如 RMSprop）和学习率来计算参数的更新量，并将这个更新量应用到模型的参数上，从而更新参数的值。这样，模型的参数会朝着减小损失函数的方向进行调整。

#### 损失函数与优化器
- 常用的损失函数
	1. 平方损失  输出-预期的平方的求和
	2. **最大似然**处理，输出的结果（似然值）视为概率，再去求得到该结果概率值最大的权重系数 w。已知事情发生的结果，反推发生该结果概率最大的参数 w P(x|w,b)
	3. 交叉熵损失

```python
"""损失函数的使用
# 1.定义两个变量
# 2.损失函数选择L1Loss()，参量选择 均值与取和——(P1-E1)+(P2-E2)+...(PN-EN)/N
# 3.损失函数选择MSELoss()——(P1-E1)^2+(P2-E2)^2+...(PN-EN)^2/N
"""
import torch
from torch import nn

y_pred = torch.tensor([1,2,3],dtype=torch.float32)
y = torch.tensor([1,2,5],dtype=torch.float32)

# 2.损失函数选择L1Loss()，参量选择 均值与取和——(P1-E1)+(P2-E2)+...(PN-EN)/N
loss_l1 = torch.nn.L1Loss(reduction="sum")
result1 = loss_l1(y_pred,y)
print(result1)

# 3.损失函数选择MSELoss()——(P1-E1)^2+(P2-E2)^2+...(PN-EN)^2/N
loss_mse = torch.nn.MSELoss(reduction="sum")
result2 = loss_mse(y_pred,y)
print(result2)
```

- 常用的优化器
	- SGD
	- Adam

#### 构建神经网络全过程
搭建+训练（Chapter 3+4）
下载数据->加载数据->准备模型->设置损失函数->设置优化器->开始训练->最后验证->结果聚合展示

```python
from torch import nn
import torch

# 1.搭建模型
class Mynetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3,32,5,1,2),nn.MaxPool2d(2),
                                   nn.Conv2d(32,32,5,1,2),nn.MaxPool2d(2),
                                   nn.Conv2d(32,64,5,1,2),nn.MaxPool2d(2),
                                   nn.Flatten(),nn.Linear(64*4*4,64),nn.Linear(64,10))
                                   # 包括卷积、池化、线性等等

    def forward(self,data): # 前向驱动函数
        output = self.model(data)
        return output

# 2.得到数据集
input = torch.ones(size=(1,3,32,32),dtype=torch.float32) # 数据集
y = torch.tensor([[1,2,3,4,5,6,7,8,9,10]],dtype=torch.float32)
# 3.调用模型得到输出
mynn = Mynetwork()
loss_fn = torch.nn.MSELoss(reduction="mean") # 损失函数
optimiser = torch.optim.RMSprop(params=mynn.model.parameters(),lr=1e-4) # 定义优化器

for period in range(100):
    print(f"this is period {period+1}:")
    for data in range(1):
        y_pred = mynn(input) # 导入模型
        # 4.计算真实值与输出值之间的偏差loss
        loss = loss_fn(y_pred,y)
        # 7.迭代一次后 梯度置零
        optimiser.zero_grad()
        # 5.计算各参量的梯度值
        loss.backward()
        # 6.用优化器更新参数
        optimiser.step()

        if data % 10 ==0:
            print(loss)

print(f"The final outcome is {y_pred}")

----------------------------------------------------
# output：
this is period 1:
tensor(38.6425, grad_fn=<MseLossBackward0>)
this is period 2:
tensor(34.2981, grad_fn=<MseLossBackward0>)
this is period 3:
tensor(22.4355, grad_fn=<MseLossBackward0>)
this is period 4:
tensor(3.7360, grad_fn=<MseLossBackward0>)
this is period 5:
tensor(4.2519, grad_fn=<MseLossBackward0>)
this is period 6:
tensor(7.7438, grad_fn=<MseLossBackward0>)
this is period 7:
.
.
.
.
tensor(0.1047, grad_fn=<MseLossBackward0>)
this is period 96:
tensor(0.1212, grad_fn=<MseLossBackward0>)
this is period 97:
tensor(0.0982, grad_fn=<MseLossBackward0>)
this is period 98:
tensor(0.1128, grad_fn=<MseLossBackward0>)
this is period 99:
tensor(0.0920, grad_fn=<MseLossBackward0>)
this is period 100:
tensor(0.1051, grad_fn=<MseLossBackward0>)
The final outcome is tensor([[0.8946, 1.8333, 2.8251, 3.7509, 4.7626, 5.6981, 6.6488, 7.6014, 8.5054,
         9.5048]], grad_fn=<AddmmBackward0>)
```

`Tensor(38.6425)`：这部分表示损失函数的数值，即计算得到的具体损失值。在这个例子中，损失函数的值为 38.6425。
`grad_fn=<MseLossBackward0>`：这部分表示损失函数的计算图中的**反向传播函数**。它指示了该张量是通过执行反向传播操作计算得到的，并且在计算图中有一个与之相关的反向传播函数。在这个例子中，使用的是**均方误差损失函数（MSELoss）**，因此显示为 `<MseLossBackward0>。`