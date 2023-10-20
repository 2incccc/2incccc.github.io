---
title: 【MIR】Signal Analysis and Reature Extraction
date: 2023-10-20 10:49:57
tags: [音乐信息检索]
categories:
- Music Information Retrieval
---
# 基本特征提取

```python
# 绘制波形
librosa.display.waveshow([x[:1000]])
# 特征提取：过零率/质心
def extract_features(signal):
	stft_signal = librosa.core.stft(signal)
	magnitude = numpy.abs(stft_signal)
	return [
	librosa.feature.zero_crossing_rate(signal)[0,0]
	librosa.feature.spectral_centroid(S=magnitude)[0,0]
	]
```

**零交叉率（Zero Crossing Rate）**：零交叉率是一个表示信号快速变化的特征。它指的是信号波形穿过零轴的次数。在这段代码中，通过 librosa.feature.zero_crossing_rate(signal)[0, 0]来计算音频信号的零交叉率，并将结果作为特征之一返回。

**频谱质心（Spectral Centroid）**：频谱质心是频谱能量的加权平均值，用于表示音频信号的频谱中心。频谱质心越高，表示频谱的能量集中在较高的频率上，反之亦然。在这段代码中，通过 librosa.feature.spectral_centroid(S=magnitude)[0, 0]来计算音频信号的频谱质心，并将结果作为特征之一返回。

## Feature Scaling 特征缩放

我们在上一个示例中使用的特征包括过零率和谱质心。这两个特征使用不同的单位来表示。这种差异可能会在稍后执行分类时带来问题。因此，我们将每个特征向量归一化到一个公共范围，并存储**归一化**参数以供以后使用。存在许多用于扩展功能的技术。现在，我们将使用[`sklearn.preprocessing.MinMaxScaler`]( http://scikit-learn.org/stable/modules/ generated/sklearn.preprocessing.MinMaxScaler.html)。 `MinMaxScaler` 返回一个缩放值数组，使得每个特征维度都在 -1 到 1 的范围内。

```python
feature_table = numpy.vstack((kick_features, snare_features))

scaler = sklearn.preprocessing.MinmaxScaler(feature_range=(-1,1)) # 定义一个范围在-1到1的预处理器
training_features = scaler.fit_transform(feature_table)

```

# Segmantation 分割

在音频处理中，通常使用恒定的帧大小和跳跃大小（即增量）一次对一帧进行操作。帧的持续时间通常选择为 10 到 100 毫秒。

## Segmentation Using Python List Comprehensions

在 Python 中，您可以使用标准的列表理解（ https://docs.python.org/2/tutorial/datastructs.html#list-com经理 ）来执行信号分割并同时计算 RMSE。

```python
# 定义帧长和间隔
frame_length = 1024
hop_length = 512
# 均方根
def RMSE(x):
	return numpy.sqrt(numpy.mean(x**2))
```

给定一个信号，[`librosa.util.frame`]( https://librosa.github.io/librosa/ generated/librosa.util.frame.html #librosa .util.frame)将生成一个统一大小的帧列表:

```python 
frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length)
```

# Energy 能量

信号的**能量**（[Wikipedia]( https://en.wikipedia.org/wiki/Energy_ (signal_processing%29); FMP, p. 66）对应于信号的总幅度。对于音频信号，大致对应于信号的响度。信号中的能量定义为 $$ \sum_n \left| x(n) \right|^2 $$

The **root-mean-square energy (RMSE)** in a signal is defined as

$$ \sqrt{ \frac{1}{N} \sum_n \left| x(n) \right|^2 } $$

```python
# 按照定义
energy = numpy.array([
    sum(abs(x[i:i+frame_length]**2))
    for i in range(0, len(x), hop_length)
])
# 利用librosa函数
rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length, center=True) # shape(1,194)
rmse = rmse[0]

```

```python
# 比较波形图和均方根能量
frames = range(len(energy))
t = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)
librosa.display.waveshow(x, sr=sr, alpha=0.4)
plt.plot(t, energy/energy.max(), 'r--')             # normalized for visualization
plt.plot(t[:len(rmse)], rmse/rmse.max(), color='g') # normalized for visualization
plt.legend(('Energy', 'RMSE'))
```

# Zero Crossing Rate 过零率

过零率指代信号波形穿过零轴的次数

```python
n0 = 6500
n1 = 7500
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1]) # zoom in

zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False) # 是否经过零点，输出结果为False和True的组合
zeor_crossings.shape # output: (1000,0)

zcrs = librosa.feature.zero_crossing_rate(x) # 过零率
print(zcrs.shape) # output: (1,97)
```

过零率的高低与信号波形的特性有关。以下是一些常见的情况：

**浊音/有谐波声音**：浊音指的是声音中含有频谱中的多个谐波分量，通常听起来比较富有音色。浊音的过零率较低，因为在谐波声音中，波形会频繁穿过零线。

**清音/无谐波声音**：清音指的是声音中几乎没有谐波成分，通常听起来比较纯净。清音的过零率较高，因为在没有谐波的声音中，波形变化相对较平缓，不会频繁穿过零线。

**静音**：静音时，信号波形处于零线附近，过零率较高，因为信号在静音状态时频繁地从正值到负值或从负值到正值。

# 傅立叶变换

傅里叶变换([维基百科](https://en.wikipedia.org/wiki/Fourier_transform))是应用数学和信号处理中最基本的运算之一。

它将时域信号转换到频域。时域将信号表示为一系列采样，而频域将信号表示为不同幅度、频率和相位偏移的正弦波的叠加。

```python
x,sr = librosa.load('filename') # 加载音频


X = scipy.fft(x) # 求傅立叶变换
X_mag = numpy.absolute(X) # 求模
f = numpy.linspace(0, sr, len(X_mag)) # frequency variable 频率范围

plt.figure(figsize=(13, 5))
plt.plot(f, X_mag) # magnitude spectrum
plt.xlabel('Frequency (Hz)')
```

# Short-Time Fourier Transform STFT 短时傅里叶变换

音乐信号是高度非平稳性的，也就是说，它们的统计数据会随着时间而变化。在一整首10分钟的歌曲中计算一次傅里叶变换是毫无意义的。

**短时傅里叶变换(STFT)**([维基百科](https://en.wikipedia.org/wiki/Short-time＿Fourier＿transform);FMP，第 53 页)是通过计算信号中连续帧的傅里叶变换得到的。 

$$ X(m, \omega) = \sum_n x(n) w(n-m) e^{-j \omega n} $$

当我们增加 $m$ 时，我们将窗口函数 $w$ 向右滑动。对于得到的坐标系，$x(n) w(n-m)$，我们计算傅里叶变换。因此，STFT $X$ 是时间 $m$ 和频率 $ω$ 的函数。

```python
hop_length = 512
n_stft = 1024 # 设定STFT参数，包括帧长度和间隔
X = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
```

## Spectrogram 谱图

在音乐处理中，我们通常只关心谱幅值而不关心相位含量。

**谱图**([维基百科](https://en.wikipedia.org/wiki/Spectrogram);FMP(第 29、55 页)显示了频率随时间的强度。谱图就是 STFT 的平方幅度:

$$ S(m, \omega) = \left| X(m, \omega) \right|^2 $$

人类对声音强度的感知是基于对数（logarithmic）的，所以我们对对数幅度更感兴趣

```python
S =  librosa.amplitude_to_db(abs(X)) # 转化为对数

plt.figure(figsize = (15,5))
librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear') # 使用specshow函数
plt.colorbar(format='%+2.0f dB')

```

## Constant -Q Transform 常数 Q 变换

与傅立叶变换不同，但类似于 MEL 比例，常量 Q 变换([Wikipedia](http://en.wikipedia.org/wiki/Constant_Q_transform))使用对数间隔的频率轴。

**Constant -Q  Transform** (CQT)是一种在频率上**使用不同的频率分辨率**来表示音频信号的方法，它模拟了人类听觉系统对不同音高的感知尺度。通过 CQT 变换，我们可以将音频信号转换为频谱表示，其中横轴表示时间，纵轴表示音高。

```python
fmin = librosa.midi_to_hz(36) # 设定最低频率
C = librosa.cqt(x, sr=sr, fmin=fmin, n_bins=72) 
logC = librosa.amplitude_to_db(abs(C)) # 转换为对数谱
```

参数解释：

`fmin` 是 CQT 变换的最低频率，表示变换时使用的最低音高。较低的 fmin 值将使 CQT 对低音更敏感。

`n_bins` 表示频率的总数量，它决定了 CQT 变换的音高范围和频率分辨率。N_bins 越大，音高范围越宽，频率分辨率越高。

## Chroma

**(Chroma Vector) 色度向量**([Wikipedia](https://en.wikipedia.org/wiki/Chroma_feature))(fmp，p.123)通常是12个元素的特征向量，指示信号中存在每个基音类别{C，C#，D，D#，E，…，B}的多少能量。

```python
chromagram01 = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length) # stft的色度向量 
chromagram02 = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length) # cqt的色度向量
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
```

输出结果：

![16907236562081690723655380.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907236562081690723655380.png)

![16907237542071690723753610.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907237542071690723753610.png)

**色度能量归一化统计量(Chroma energy normalized statistics, CENS)**。CENS 功能的主要思想是对大窗口进行统计，以平滑节奏、清晰度和音乐装饰(如颤音和弦)的局部偏差。CENS 最适合用于**音频匹配和相似性**等任务。`librosa.feature.chroma_cens()`

![16907238742061690723873877.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907238742061690723873877.png)

# Magnitude_scaling 振幅缩放（？）

通常，信号在时域或频域中的原始幅度与人类的感知相关性不如转换成其他单位的幅度，例如使用对数标度。

即使振幅呈指数增长，对我们来说，响度的增加似乎是渐进的。这种现象是 Weber-Fechner 定律([维基百科](https://en.wikipedia.org/wiki/Weber%E2%80%93Fechner_law))的一个例子，该定律指出刺激和人类感知之间的关系是对数的。

# Spectral Features 频谱特征

对于分类问题，我们将使用新的统计量矩（Moment）（包括质心、带宽、偏度、峰度）和其他谱统计数据。

矩（Moment）是物理学和统计学中出现的术语。矩的两个示例：均值和方差，第一个是原点矩，第二个是中心矩。

## 频谱质心

**频谱质心**（[维基百科](https://en.wikipedia.org/wiki/Spectral_centroid)）指示频谱能量集中在哪个频率。这就像加权平均值： $$ f_c = \frac{\sum_k S(k) f(k)}{\sum_k S(k)} $$ 其中 $S(k)$ 是频率 bin $ 处的频谱幅度 k$, $f(k)$ 是 bin $k$ 处的频率。

[`librosa.Feature.Spectral_centroid`]( https://librosa.github.io/librosa/ generated/librosa.Feature.Spectral_centroid.Html #librosa .feature.Spectral_centroid) 计算信号中每个帧的光谱质心.

输出图像：

![16907725302121690772529334.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907725302121690772529334.png)

与过零率类似，信号开始处的频谱质心存在虚假上升。这是因为开始时的静默幅度很小，高频成分有机会占主导地位。解决这个问题的一种方法是在计算光谱质心之前添加一个小常数，从而在安静部分将质心移向零。

## 频谱带宽

[`librosa.feature.spectral_bandwidth`]( https://librosa.github.io/librosa/ generated/librosa.feature.spectral_bandwidth.html #librosa .feature.spectral_bandwidth) 计算 $p$ 阶光谱带宽：

$$ \left( \sum_k S(k) \left(f(k) - f_c \right)^p \right)^{\frac{1}{p}} $$ 其中 $S(k)$ 是在频率 $k$ 处的幅度，$f(k)$ 是 $k$ 处的频率，$f_c$ 是频谱质心。当 $p = 2$ 时，这就像加权标准差。

## 频谱对比度

考虑频谱峰值、谷值以及他们在每个频率子带中的差异。

[`librosa.feature.spectral_contrast`]( https://librosa.github.io/librosa/ generated/librosa.feature.spectral_contrast.html) 计算每个时间帧的六个子带的光谱对比度：

## 频谱滚降

是指低于总频谱能量指定百分比的频率

`lirosa.feature.spectral_rolloff`

# Autocorrelation 自相关

指代自身和时移后自身的相关性。对于信号 $x$，它的自相关信号 $r(k)$ 为

$$ r(k) = \sum_n x(n) x(n-k) $$

在此等式中，$k$ 通常称为 **lag** 参数。 $r(k)$ 在 $k = 0$ 处最大化，并且关于 $k$ 对称。

自相关对于查找信号中的重复模式很有用。例如，在短滞后时，自相关可以告诉我们有关信号基频的信息。对于较长的滞后，自相关可以告诉我们一些有关音乐信号节奏的信息。

两种计算 autororrelation 的方法：`numpy.correlate` 和 `librosa.autocorrelation`

## 音高估计

自相关用于查找信号内的重复模式。对于音乐信号，重复模式可以对应于音高周期。因此，我们可以使用自相关函数，通过**寻找最值点**来估计音乐信号中的音高。

# Pitch Transcription Exercise 声调转录

在音频信号处理中，声调转录是指将音频中的**音高信息**转录成对应的**音符或音高表示**的过程。这个过程通常涉及到分析音频信号中的频率变化和音高轮廓，从而识别出其中的音符和音高变化。

## 准备工作：

- 导入库函数
```python
%matpltlib inline # 将图像输出在notebook中而不是在新窗口
import numpy, IPython.display as ipd, matpltlib.pyplot as plt
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 5)
```

- 加载音频并播放
```python
filename = '../audio/simple_piano.wav'
x,sr = librosa.load(filename)
ipd.Audio(x,rate=sr)
```

- 计算 CQT 并输出频谱
```python
bins_per_octave = 36 # 设置每个八度的频率间隔数目，表示频率轴分辨率
cqt = librosa.cqt(x,sr=sr, n_bins=300, bins_per_octave = bins_per_octave) # 使用cqt函数
log_cqt = librosa.amplitude_to_db(numpy.abs(cqt)) #转化对数谱
librosa.display.specshow(log_cqt, sr=sr, x_axis='time',
						 y_axis='cqt_note', bins_per_octave=36)
```

![16907872882141690787288035.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907872882141690787288035.png)

简单观察声谱图可知，整个音频大概包含八个相同或不同的音符，但是混杂在每一个音符的还有各种其他频率的分量。

## 任务目标

识别每个音符的音高，并将每个音符用相同音调的纯音（Pure Tone）组合起来代替音频

## 任务流程

### 第一步：检测起点

在音频信号处理和音乐分析中，"onset"（起点）是指音频信号中音乐或声音的开始部分，即音频信号开始出现显著能量变化的位置。换句话说，"onset" 表示音频信号中从无声到有声或从背景噪声到音乐开始的那个时间点。

在音频信号处理中，通常使用不同的算法和特征来检测 "onset"，比如短时能量、短时过零率、梅尔频率倒谱系数（MFCC）等。这些方法可以帮助准确地找到音频信号中显著的能量变化点，从而确定 "onset" 的位置。

在这里，我们使用新颖度函数（novelty function）来寻找音频信号中的起点。

```python
hop_length = 100
onset_env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
plt.plot(onset_env)
plt.xlim(0,len(onset_env))
```

![16907876922071690787691364.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907876922071690787691364.png)

在上图可以看到，除了几个比较显著的波峰，还有更多的很小的波峰，我们需要设置参数来忽略这些很小的波峰。

接下来使用 `onset_detct` 实现对起点的检测

```python
onset_samples = librosa.onset.onset_detect(y=x,
                                           sr=sr, units='samples', 
                                           hop_length=hop_length, 
                                           backtrack=False,
                                           pre_max=20,
                                           post_max=20,
                                           pre_avg=100,
                                           post_avg=100,
                                           delta=0.2,
                                           wait=0)
print(onset_samples)
```

```
output:[5800 11300 22300 33300 44300 55300 66400]
```

为了能将整个音频按照音符数分割开来，还要在序列首尾添加 `padding`

```python
onset_boundaries = numpy.concatenate([[0], onset_samples, [len(x)]])
print(onset_boundaries)
```

```
output:[0  5800 11300 22300 33300 44300 55300 66400 84928]
```

最后将采样点数转换为时间

```python
onset_times = librosa.samples_to_time(onset_boundaries,sr=sr)
print(onset_times)
```

```bash
output:
array(array([ 0.        ,  0.26303855,  0.51247166,  1.01133787,  1.51020408,2.00907029,  2.50793651,  3.01133787,  3.85160998]))
```

最后将分割后的结果在频谱图中展示出来

```python
librosa.display.waveshow(x,sr=sr)
plt.vlines(onset_times,-1,1,color='r')
```

![16907894122071690789411700.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907894122071690789411700.png)

经过上面的操作，我们可以看懂，红线将整个将音频中八个音符，对应波形中有明显不连续的地方分割开来。

## 第二步，估计音调

我们效仿前面的学习内容，使用**自相关方法**确定音高。

> **自相关**用于查找信号内的重复模式。对于音乐信号，重复模式可以对应于音高周期。因此，我们可以使用自相关函数，通过**寻找最值点**来估计音乐信号中的音高

```python
def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):
    
    # 计算输入的自相关
    r = librosa.autocorrelate(segment)
    
    # 定义自相关最值点的范围
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0
    
    # 寻找最值，返回对应频率
    i = r.argmax()
    f0 = float(sr)/i
    return f0
```

## 第三步：生成纯音 Pure Tone

这里我们直接使用 `numpy.sin` 生成频率固定的正弦波纯音。

```python
def generate_sine(f0, sr, n_duration):
	# 生成正弦波
    n = numpy.arange(n_duration)
    return 0.2*numpy.sin(2*numpy.pi*f0*n/float(sr))
```

## 第四步：将纯音组合起来

```python
def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
	# 找到起点位置的频率，将每一音符分割开来
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    
	# 调用函数，估计每个音符的音高
    f0 = estimate_pitch(x[n0:n1], sr)

	# 返回相同音高的纯音
    return generate_sine(f0, sr, n1-n0)
```

接下来使用 `numpy.concatenate` 将合成的片段连接起来并演奏

```python
y = numpy.concatenate([
    estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr=sr)
    for i in range(len(onset_boundaries)-1)
])
ipd.Audio(y,rate=sr)
```

为可视化展现合成后音频的最终结果，绘制合成后音频的 CQT 谱图

```python
cqt=librosa.cqt(y,sr=sr)
librosa.display.specshow(abs(cqt),sr=sr,x_axis='time',y_axis='cqt_mode')
```

![16907904922071690790491649.png](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16907904922071690790491649.png)

可以清晰看到，每个音符对应的频率谱图变得纯净，其他频率的分量基本完全消失。

至此，我们完成了这段音频的声调转录工作。