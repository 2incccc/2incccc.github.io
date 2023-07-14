---
title: 【音频信号处理及深度学习教程】Chapter 2&3
date: 2023-07-13 00:46:49
tags: [音频信号,深度学习,声学,神经网络,信号处理,人工智能]
categories: 
- 音频信号处理及深度学习教程 
math: true
---

## Chapter 2 Pycharm 入门
### 相关介绍
- Anaconda:包管理平台
- Conda:是环境的管理工具
- CUDA:Compute Unified Device Architecture,，是一种由 NVIDIA 推出的通用并行计算架构，该架构使 GPU 能够解决复杂的计算问题。
- CUDNN ：是针对深度卷积神经网络的加速库。NVIDIA CUDNN 可以在 GPU 上实现高性能现代并行计算
<!--more-->
- pip 和 conda 比较：
	- 依赖项检查
    pip：不一定会展示所需的其他依赖包Conda：列出所需其他依赖包。安装包 时自动安装其依赖项。可以便捷地在包的不同版本中自由切换。
	- 环境管理
    Conda：在不同环境之间进行切换，环境管理较为简单。
	- 对系统自带 python 的影响
    pip：在系统自带 python 中包的更新/回退版本/卸载将影响其他程序。Conda：不会影响系统自带 python。
	- 适用语言
    pip：仅适用于 python。Conda：适用于 python, R, Ruby, Lua, Scala, Java, JavaScript, C/C++, FORTRAN


### 张量 Tensor
张量：是一种特殊的数据结构，在 PyTorch 中，神经网络的输入输出都用张量描述

### 常用的类 Class
什么是类 Class？存储对象和函数的集合，调用 Class 中的函数，完成我们需要实现的功能。
- Dataset
- Dataloader
- SummaryWriter
- nn.Module
上方实例见源码，此处不作赘述

## Chapter 3 构建神经网络

### 机器学习介绍
- 流程：通过传感器来获得数据、预处理、特征提取、特征选择，最后进行推理、预测或者识别，这就是机器学 习的过程。
- AI & DL & ML：机器学习是人工智能的一个分支，而深度学习又是机器学习的下属分支，因此它们是环环紧扣的
  ![16889995409501688999540110.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16889995409501688999540110.png)
机器如何学习?
![16889995869501688999586642.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16889995869501688999586642.png)
概括：准备数据集，先进行人为分析，之后对数据集预处理，再将其分割为 `Train set` 训练集和 `Test set` 测试集，根据**训练集**结合相关**算法**，训练出我们需要的**模型**，（核心部分参数调优），最终机器学习到如何根据输入得到想要的输出，完成学习任务，再利用**测试集**对结果进行评价

#### 如何建设映射关系得到输出？
  通过**正向和反向传播**两个过程，根据大量训练样本，机器学习统计他们之中的规律，从而对未知事件做预测，并且生成模型的输出值与期望值越来越拟合 。
   - 正向传播：通过模型的每一层运行输入数据以进行**预测**，prediction。 
   - 反向传播：使用模型的预测和已知的标签来**计算误差**（有监督学习），
	然后通过模型反向传播，利用误差导数在梯度下降处得到最小误差来优化模型参数完成学习

#### 分类和回归是什么。有什么区别？
  分类模型和回归模型本质一样，都是要建立映射关系，利用一个训练好的模型将 一组变量作为输入，输出其对应的标签。
   • 分类：
   输出对象是离散变量，并且输出空间不是同一个度量空间。上图是由不 同颜色和标签表示的三个类，每一个小的彩色球体代表一个数据样本。
   • 回归：
   输出对象是连续变量，输出空间是同一个度量空间。下图的函数为这个 映射关系，离散的点为实际值(actual value)，对应到函数上的值为输出值 (predicted value)

### 现代机器学习
现代机器学习有一个分支是深度学习
- 区别：不需要认为分析数据
- 特征：
  1. 深度学习也就是现代机器学习，不需要人为去提取数据特征， 而是通过神经网络自动对数据进行高维抽象学习，其输出的结 果准确率大大提升，并且可以完成很多复杂的任务。
  2. 但是调参工作变得更加繁重。因为引入了更加深、更复杂的网 络模型结构，所以需要定义神经网络模型结构、确认损失函数、 确定优化器，最后就是反复调整模型参数的过程。并且很多模 型是黑盒模型。
  ![16890001759511689000175235.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16890001759511689000175235.png)

### 机器学习的分类
#### 传统机器学习
人为进行提取数据
- 流程：
  1. 数据集准备
  2. 人为分析数据
  3. 数据预处理
  4. 数据分割（训练/测试）
  5. 机器学习算法建模
  6. 选择学习任务，评价成果
- 常见算法：
	- 支持向量机
	- 线性判别分析 LDA
	- 主成分分析 PCA
	- 决策树算法
	- 聚类算法
	- 玻尔兹曼机（图模型）

#### 现代机器学习
机器通过学习生成模型，机器对数据进行提取
- 流程
  1. 数据集准备
  2. 数据预处理
  3. 数据分割
  4. 定义神经网络模型
  5. 训练网络

### 机器学习的其他几种分类
- 有无监督学习区别
![16890005969521689000596702.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16890005969521689000596702.png)

### 神经网络概述
####  什么是神经网络？
- 流程：
	- 输入信号乘以权重
	- 累加输入以及偏移量
	- 非线性激活，得到输出
- 结构：
	- 输入层
	- 隐藏层
	- 输出层
#### 如何处理音频的网络？
因为复杂图形，往往由一些基本结构组成。可以用一组正交的基本结构， 将目标图像按照权重调和而成，音频的语谱图也是如此。
在音频信号中，在未标注的声音中发现了20种基本的声音结构，因此其 余的声音可以由这20种基本结构合成。这也就是利用图像识别的思路完 成语音识别的基础。
![16890029569511689002956650.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16890029569511689002956650.png)
#### 什么是深度学习？
具有很多隐藏层的机器学习模型和海量的训练数据，因此可以学习更有用的特征，从而最终提升分类或预测的准 确性。

#### 感知器
网络神经的最基本单元，包含参数 w 和 b 的激活函数

```python
"""定义有三个输入的感知元
# 1.定义感知元函数，分为线性部分 y=w*x+b 和非线性激活部分sigmoid: y=1/(1+math.exp(-y))
# 2.调用该函数，确定输入值与每一个输入对应的权重与偏差值
# 3.得到输出
"""
import math

# 1.定义感知元函数，分为线性部分 y=w*x+b 和非线性激活部分sigmoid: y=1/(1+math.exp(-y))
def neuron(input,weight,bias):
    output = 0
    for x,w in zip(input,weight):
        output += w*x
    output = output + bias
    output = 1 / (1 + math.exp(-output))
    return output
# 2.调用该函数，确定输入值与每一个输入对应的权重与偏差值
input = [1,2,3]
weight = [0.1,0.4,0.2]
bias = 0.02

output = neuron(input,weight,bias)
print(output)
```

#### 浅层学习
主要完成识别，分类等基本任务，其最主要的特点是模型简单，包含输  
入层，隐藏层和输出层。（一层隐藏层）

```python
"""定义浅层学习网络函数(含两个运算层)
# shallow_learning(input,weight1,weight2,bias1,bias2):
# 1.定义线性部分output1 = np.sum(input*weight)+bias
# 2.定义非线性激活部分output1 = 1 /(1 + np.exp(-output1) )
# 3.给定输入与权重系数，得到输出
"""
import math
import numpy as np
class ShallowNet:
    def __init__(self,weight1,bias1,weight2,bias2):
        self.weight1 = weight1
        self.bias1 = bias1
        self.weight2 = weight2
        self.bias2 = bias2
    def calculate(self,x):
        output = np.dot(x,self.weight1) + self.bias1
        output = 1 / (1 + np.exp(-output)) # output使用np数组，此处exp需要用np.exp()  
        y = np.dot(output,self.weight2) + self.bias2
        y = 1  / (1 + np.exp(-y))
        return y    
input = np.array([1,2,3])
weight1 = np.array([[0.1,0.2,0.3,0.4],
                   [0.2,0.3,0.4,0.5],
                   [0.1,0.2,0.4,0.3]])
# np.array([[],[],[]]) 第一层中括号内每一行再用一个中括号，（也是一个np元素）
bias1 = 0.01
weight2 = np.array([[0.3,0.2,0.1],
                    [0.2,0.1,0.3],
                    [0.5,0.6,0.7],
                    [0.1,0.2,0.3]])
bias2 = 0.02
mynn = ShallowNet(weight1=weight1,bias1=bias1,weight2=weight2,bias2=bias2)
output = mynn.calculate(input)
print(output)
```

#### 深度学习
用于特征提取的复杂任务中。其主要特点是通过构建具有很多隐层的机器学习模型和海量的训练数据，来学习更  有用的特征，从而最终提升分类或预测的准确性。（多个隐藏层）
- 与浅层学习的比较
	- 浅层结构算法多用于多数分类、回归等学习方法，其局限性在于有限样本和计算单元情况下对复杂函数的表  示能力有限，针对复杂分类问题其泛化能力受到一定制约。  
	- 深度学习通过学习深层非线性网络结构，实现复杂函数逼近，表征输入数据分布式表示。同时展现了强大的从少数样本集中学习数据集本质特征的能力。<span style="background:#e67575">多层的好处是可以用较少的参数表示复杂的函数。</span>

```python
"""定义深度学习网络的类class(无继承型)
# 1.初始化网络模型
# 根据输入个数,输出个数以及中间层数,定义网络框架
layer
# layer_innum = [input_num]+hidden_layer+[output_num]
# 随机生成weight值,weight个数与每层的输入个数相
同
# current_weight = np.random.rand(self.layer_innum[i],self.layer_innum[i
+1])
# 2.模型计算,每一部分实现线性运算np.dot()与非线性
激活output1=1 /(1 + np.exp(-output1) )
# 3.定义参数调用计算
"""

import numpy as np

class DeepNet:
    def __init__(self,input_num,output_num,hidden_num) -> None:
        self.model = [input_num] +hidden_num +[output_num]
        # 最终得到一个列表，列表中每个值代表每一层元素个数
        self.weight = []
        self.layer_num = len(self.model)
        
        for i in range(self.layer_num-1):
            current_weight = np.random.rand(self.model[i],self.model[i+1]) 
            # 创建给定形状的数组，并使用来自[0,1]的均匀分布的随机样本填充它。
            # 即生成model[i]*model[i+1]个随机数
            self.weight.append(current_weight)
            
        self.bias = 0
        
    def calculate(self,data):
        for w in self.weight:
            y = np.dot(data,w)
            y = 1/(1+np.exp(-y))
            data = y
        return y
    
input_num = 3
output_num = 1
hidden_num = [3,2,3]
```

### 构造隐藏层
#### 卷积层
利用卷积核（kernel）实现数据的特征提取
按照一定权重处理输入对象并累加。与信号处理中的卷积有些不同，这里**不存在时间反演**，只是简单的相乘累加。
- 相关名词
卷积核/kernel = 滤波器 卷积操作的感受野，直观理解就是一个滤波矩阵，普遍使用的卷积核大小为3×3、5×5等；
步长/stride = 帧移 卷积核遍历特征图时每步移动的像素，如步长为1则每次移动1个像素
填充值/padding = padding 处理特征图边界的方式，一般有两种，一种是对边界外完全不填充，只对输入像素执行卷积操作，这样会使输出特征图的尺寸小于输入特征图尺寸；另一种是对边界外进行填充（一般填充为0），再执行卷积操作，这样可使输出特征图的尺寸与输入特征图的尺寸一致；
	注：全 padding 对应输入每一个元素都可以取到*
扩张值/dilation
通道/Channel 卷积层的通道数（层数）
更多了解，见[[各种类型的卷积]]
![[Pasted image 20230712210744.png]]




- 卷积的类型：
	- 一维卷积
	- 二维卷积
	- 三位卷积
	- 反卷积（转置）

- 卷积出现的意义：权值共享，用卷积层代替全连接层，参数数量减少，但神经元的个数没有变少，照样可以提取特征值！
![16891668399501689166839315.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16891668399501689166839315.png)

- 维度计算：卷积后的维度 = （卷积前的维度 - 卷积核边长 + 2 * padding）/ 步长 + 1  （理解上图）

```python
"""
# 0.预设环境import torchvision,from torch import  nn
# 1.构建卷积模型，继承nn.Module，包含conv2d()
# 2.加载图像数据，SummaryWriter写入到log文件中，图像PIL-%3Etensor(transforms.Totensor)
# from PIL import Image img = Image.open(img_path)
# trans_tensor = torchvision.transforms.ToTensor()
# 转换成四维trans_tensor(img).unsqueeze(0)
# 3.调用模型计算
# 4.得到输出结果，SummaryWriter写入到log文件中
tensor->PIL(transforms.ToPIL)>)
"""
from torch import nn
from PIL import Image
import torchvision
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
# input -> (Batch_size,Channels,height,weight) ->(1,1,5,5)
img_path = r"/root/..Tutorial/audio_data/boochi.jfif"
img = Image.open(img_path)
trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)
input = img_tensor.unsqueeze(0)
# 图像数据从三维张量转换为四维张量，可以满足卷积层对输入数据维度的要求。（批操作）

class MyConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=2,stride=1,padding=1)
        # 卷积后的长宽  = (卷积前的长宽 - 卷积核边长 + 2* padding)/步长 +1
    def forward(self,data):
        output = self.conv(data)
        return output

file = SummaryWriter("log_conv")
mynn = MyConv()
output = mynn(input)
print(output.shape)
trans_PIL = torchvision.transforms.ToPILImage()
img1 = trans_PIL(output.squeeze(0))
img1.show()
file.add_graph(mynn,input)
file.close()

"""
总结：
mynn(input)中的input是作为参数传递给了forward方法中的data。
data是forward方法的参数，用于接收输入数据。
input实际上就是被视为data参数的输入数据。
"""
```

- 卷积的结果：改变个像素的数值 RGB 值，（常用于边缘检测、图像锐化、方块模糊等）

#### 池化层
也叫汇聚层，用于压缩数据和参数的量减小过拟合，将数据尺寸变少，（类比信号处理中的去采样）
![16891735529521689173552490.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16891735529521689173552490.png)

```python
"""池化的网络模型
# 0.预设环境from torch import nn,from PIL import Image
# 1.构建池化的网络模型，继承nn.Module，包含MaxPool2d()
maxpooling中的stride =  stride if (stride is not None) else kernel_size
# 2.加载图像数据，SummaryWriter写入到log文件中，图像PIL->tensor(transforms.Totensor)
# from PIL import Image img = Image.open(img_path)
# trans_tensor = torchvision.transforms.ToTensor()
# 转换成四维trans_tensor(img).unsqueeze(0)
# 3.调用模型计算
# 4.得到输出结果，SummaryWriter写入到log文件中，tensor->PIL(transforms.ToPIL)
# tensorboard --logdir=D:\WZY\Class\Pytorch_Audio\Chapter3\log_pooling --port=1000
"""
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter

img_path = r"E:\Code\audio_data\mob.jpg"
img = Image.open(img_path)
trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)
# 1.构建池化的网络模型，继承nn.Module，包含MaxPool2d()
class MyPooling(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.pooling1 = nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=False)
        # MaxPool 最大池化
        self.pooling2 = nn.AvgPool2d(kernel_size=2,stride=2,ceil_mode=False)
        # AvgPool 平均池化
        # 维度与池化方式没有关系，二者区别：MaxPool取最大特征值作为特征后结果，方便突出特征，AvgPool计算特征值平均后作为特征后结果，使图像更平滑
    def forward(self,data):
        output = self.pooling2(data)
        return output

file = SummaryWriter("log_network")
# 3.调用模型计算
mynn = MyPooling()
input = img_tensor.unsqueeze(0)
output = mynn(input)
print(output.shape)
file.add_image("Image",img_tensor)
file.add_image("Pooling",output.squeeze(0))
file.close()
trans_PIL = torchvision.transforms.ToPILImage()
img1 = trans_PIL(output.squeeze(0))
img1.show()
```

#### 非线性层
非线性激活，形成复杂的函数
- ReLU（Rectified Linear Unit，修正线性单元）   $ReLU = max(x,0)$  `nn.ReLU()`
优点：解决梯度消失，计算速度快
缺点：没有输出负数，（值域有限）一些神经元不会被激活

- Tanh  $Tanhx = \frac{e^x-e^{-x}}{e^x+e^{-x}}$  `nn.Tanh()`
优点：输出正负，解决了 zero-centered 的输出问题
缺点：梯度消失和幂运算的问题依然存在

- Sigmoid $\sigma(x) = \frac{1}{1+e^{-x}}$  `nn.Sigmoid()`
输出恒大于 0，会导致模型训练的收敛速度变慢

```python
"""非线性激活ReLU的网络模型
# 0.预设环境from torch import nn,from PIL import Image
# 1.构建池化的网络模型，继承nn.Module，包含ReLU()
# 2.加载图像数据，SummaryWriter写入到log文件中，图像PIL->tensor(transforms.Totensor)
# from PIL import Image img = Image.open(img_path)
# trans_tensor = torchvision.transforms.ToTensor()
# 转换成四维trans_tensor(img).unsqueeze(0)
# 3.调用模型计算
# 4.得到输出结果，SummaryWriter写入到log文件中，tensor->PIL(transforms.ToPIL)
# tensorboard --logdir=D:\WZY\Class\Pytorch_Audio\Chapter3\log_pooling --port=1000
"""
import torchvision.transforms
from PIL import Image
from torch import nn

img_path = r"/root/..Tutorial/audio_data/boochi.jpg.jfif"
img = Image.open(img_path)
img.show()
trans_tensor = torchvision.transforms.ToTensor()
img_tensor = trans_tensor(img)
print(img_tensor.shape)
# 1.构建非线性的网络模型
class Nonliner(nn.Module):
    def __init__(self):
        super().__init__()
        self.nonliner1 = nn.ReLU()
        self.nonliner2 = nn.Tanh()
        self.nonliner3 = nn.Sigmoid()
    def forward(self,data):
        output = self.nonliner3(data)
        return output
# 3.调用模型计算
mynn = Nonliner()
input = img_tensor.unsqueeze(0)
output = mynn(input)
print(output.shape)

trans_PIL = torchvision.transforms.ToPILImage()
img1 = trans_PIL(output.squeeze(0))
img1.show()

print(f"bias = {(img_tensor - output.squeeze(0)).sum()}")
# 衡量偏差
```

#### 线性层
>线性拟合 y = xA^T + b（二维）将输出的最后维度变小，通过线性拟合，变成一维啦

具体而言，线性层将输入向量（或批量输入）映射到另一个向量（或批量输出），其中每个输出元素与输入元素的线性组合相关联。也就是所谓全连接层（FC）

```python
## 其余部分同上
## ...
self.liner = nn.Linear(in_features=1920,ou_features=20)
## ...
```

#### 整体网络构建
```python
from torch import nn ## nn:Neural Networks 神经网络
import torch
from torch.utils.tensorboard import SummaryWriter ## 写入文件
class myNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3,32,5,1,2),nn.MaxPool2d(2),
                                   nn.Conv2d(32,32,5,1,2),nn.MaxPool2d(2),
                                   nn.Conv2d(32,64,5,1,2),nn.MaxPool2d(2),
                                   nn.Flatten(),nn.Linear(64*4*4,64),nn.Linear(64,10))
        # 定义隐藏层，根据神经网络结构，设计卷积层、线性层、池化层、非线性层等等
        # 注意flatten不是严格意义上的隐藏层，他实际上是一种数据操作

    def forward(self,data): # 代表网络的前向船体
        output = self.model(data)
        return output

if __name__ == "__main__":
    mynn = myNetwork()
    input = torch.Tensor(1,3,32,32)
    output = mynn(input)
    print(output.shape)
    file = SummaryWriter("log_nn")
    file.add_graph(model=mynn,input_to_model=input)
    file.close()
```

