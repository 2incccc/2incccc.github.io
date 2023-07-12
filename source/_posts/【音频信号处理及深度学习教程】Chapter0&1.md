---
title: 【音频信号处理及深度学习教程】Chapter 0&1
date: 2023-06-25 15:00:03
tags: [音频,深度学习,声学,神经网络,信号处理]
categories: 
- 音频信号处理及深度学习教程 
math: true
---
## Chapter 0 课程介绍
- 课程内容
	- 音频信号处理
	- Pytorch 环境介绍
	- 机器学习与深度学习原理
	- Torchaudio 应用
	- 基于音频的深度学习应用
- 学习目标
	- 入门信号处理、音频信号处理
- 课程参考资料
	- 《信号与系统》
	- 《离散时间信号分析》
	- 《离散语音信号分析》
	- 《机器学习》

<!--more-->
## Chapter 1 信号的时频域分析
### 信号分析简介
- 什么是：从复杂信号分解简单信号，从信号波形提取信息
- 为什么：收集信号特征
- 怎么做：根据信号的类型选择
![16876170054651687617004672.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16876170054651687617004672.png) 
- 多个信号叠加：形成包络结构与精细结构
  低频决定包络结构，高频决定精细结构
![16876172264641687617225959.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16876172264641687617225959.png)
### 赋值包络
- 操作：将每一帧最大值连起来就是幅值包络
- 什么是分帧
	- 分帧：将信号按照时间尺度分割，每一段长度就是帧长 frame_size,分出 n 段，帧的个数 frame_num，总采样点数为 frame_size\*frame_num。
	- 分帧重叠：为了让分帧后的信号更加平滑，需要重叠分帧，也就是下一帧中包含上一帧的采样点，那么包含的点数就是重叠长度 hop_size。
	- 分帧补零：帧的个数 frame_num = 总样本数 N / 重叠数 hop_size (分帧不补零)，因为帧的个数 frame_num 是整数，为了不舍弃最后一帧不能凑成一个完整帧长的点，需要对信号补零。此时帧的个数 frame_num = (总样本数 N -帧长 frame_size)/ 重叠数 hop_size (分帧补零)
- 实验流程
  1. 加载信号 librosa.load() 
  2. 定义一个 AE 的函数，功能为取信号每一帧中幅值最值为该帧的包络最值的获取方式：max(waveform[t*(frame_size-  hop_size):t*frame_size]) 
  3. 设置参数：每一帧长 1024，以 50%的重叠率分帧，调用该函数
  4. 绘制信号的幅值包络信息
 


-  ![实验结果图像](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16876708683471687670866508.png)
### 均方根能量
- 操作：依次寻找每一帧中的 RMSE，其值为第 t 帧中每点幅值平方再取均值后开根号

$$
RMS_t = \sqrt{\frac{1}{k}* \sum\limits_{k=t·K}^{(t+1)·K-1}s(k)^2}
$$

- 对比：与时域包络相比，RMSE 体现了每一帧的包络变化，适用于不平稳的信号。尤其是对于突变信号（outlier effect），RMSE 得到的值较平稳, 因为它利用每一帧的所有点幅值的平均值，而不像 AE 利用每一帧中的最大幅值。
- 应用：RMSE 与响度有关，用于音频分段、分类 audio segmentation, music genre classification。
- 实验流程
```python
# 信号的均方根值RootMeanSquareEnergy
# 0.预设环境
# 1.加载信号
# 2.定义函数RMS，功能：计算每一帧的均方根能量，
# 公式=该帧信号的平方和，取帧长的平均值后,开根号后
# 3.设置参数：每一帧长1024，以50%的重叠率分帧，调用该函数
# 4.绘制图像
# 5.利用librosa.feature.rms绘制信号的RMS
# 6.比较两者差异
```
-  ![16876722124651687672212417.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16876722124651687672212417.png)
### 过零率
- 介绍：是一个信号符号变化的比率，即在每帧中语音信号从正变为负或从负变为正的次数。计算第 t 帧信号过零点数

$$
ZCR_t = \frac{1}{2}\cdot\sum\limits_{k=t\cdot K}^{(t+1)\cdot K-1}\lvert sgn(s(k)) - sgn(s(k+1)) \rvert
$$

- 功能：用于语音识别和音乐信息检索。通常对类似金属、摇滚等高冲击性的声音的具有更高的价值。一般情况下，过零率越大，频率 近似越高。


- 常见方法：傅里叶频谱分析、功率谱分析、倒频谱分析、共振解调技术等
- 结果：得到不同频率的幅值和相位，幅值表示了原始信号和 sin 的相似度
### 谱质心 Spectral centroid
![16876732874661687673287215.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16876732874661687673287215.png)
### 子带带宽
![16876733194771687673319164.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16876733194771687673319164.png)
### 短时傅里叶分析 STFT
- 介绍：由于声信号往往是随时间变化的，在短时间内可以近似看做平稳（对于语音来说是几十毫秒的量级），所以我们希望把长的声音切短，来观察其随时间的变化情况，由此产生 STFT 分析方式。
- 得到不同时刻，不同频率的频谱图（能量分布情况）

$$
F(w) = \intop\limits_{-\infty}^{+\infty}f(t)e^{-jwt}dt
$$

$$
F(\tau,w) = \intop\limits_{-\infty}^{+\infty}f(t)w(t-\tau)e^{-jwt}dt
$$

- 关系：
  如果窗函数带宽长，则包络中的精细结构较少，疏松，得到窄带语谱图，有较好的频域分辨率，但时域分辨率较差；如果窗函数带宽窄，则包络中的精细结构较多，密集，得到宽带语谱图，有较好的时域分辨率，但时域分辨率较差；
- 实验流程:
	- 1.加载信号
	- 2.设置参数，调用 librosa.stft 函数
	- 3.创建一个独立的画图文件，方便调用
	- 4.绘制 STFT

### 小波变换
- 介绍：利用小波作为基函数，各个小波函数按照不同比例系数展开得到 F，其中小波函数可以更改中心频率和带宽
- 操作：
  下面公式中 $\psi$ 即为小波函数也就是基函数，改变 $\tau$ 改变中心频率，改变 $s$ 改变窗的长度

$$
F(\tau,s) = \frac{1}{\sqrt {|x|}} \intop\limits_{-\infty}^{+\infty}f(t)\psi^*(\frac{t-\tau}{s})dt
$$

### Mel-Filter-Banks & MFCC


