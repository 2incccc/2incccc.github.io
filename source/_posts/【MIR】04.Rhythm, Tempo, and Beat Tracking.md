---
title: 【MIR】Rhythm, Tempo, and Beat Tracking
date: 2023-10-26 18:45:56
tags: [音乐信息检索]
categories:
- Music Information Retrieval
---
## Novelty Function
为了检测音符的开始，我们希望定位信号瞬态区域开始的突然变化，但是考虑到音高的变化不仅仅涉及响度的变化，也可能只涉及频率的改编（比如小提琴的演奏），所以，接下来我们介绍基于能量和基于频谱的 **Novelty Function**

### Energy-based Novelty Functions

#### 通过均方根能量计算
主要涉及到的步骤：直接计算 RMSE 能量、求取 RMSE 的变化量（体现能量变化），对 delta RMSE 做半波整流，只保留能量增加部分
```python
# 直接计算 RMSE 能量
y_rms =librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length).flatten() 

# 计算差分
rmse_diff = numpy.zeros_like(rmse)
rmse_diff[1:] = numpy.diff(rmse)  

# 半波整流
energy_novelty = numpy.max([numpy.zeros_like(rmse_diff), rmse_diff], axis=0) 

# 输出图像并比较
plt.figure(figsize=(15, 6))
plt.plot(t, rmse, 'b--', t, rmse_diff, 'g--^', t, energy_novelty, 'r-')
plt.xlim(0, t.max())
plt.xlabel('Time (sec)')
plt.legend(('RMSE', 'delta RMSE', 'energy novelty')) 
```

![](https://fastly.jsdelivr.net/gh/2incccc/MyTuTu@main/image/16983115624001698311561925.png)

#### 对数能量
人类对声音强度的感知本质上是对数的。为了解释这一性质，我们可以在进行一阶差分之前对能量应用对数函数。
```python
log_rmse = numpy.log1p(10*rmse)
```
其余过程同上

### Spectrum-based Novelty Functions
```python
spectral_novelty = librosa.onset.onset_strength(y=x, sr=sr) # 使用谱通量计算新奇函数
```
具体来说，`librosa.onset.onset_strength` 函数使用一种称为"onset strength" 的算法来计算这个信号。这个算法的目标是识别出音频信号中的突出事件，通常与音符、鼓击或其他音乐性事件的开始时刻相关

## Peak computing 峰值计算
```python
def peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait):
   '''Uses a flexible heuristic to pick peaks in a signal.

        A sample n is selected as a peak if the corresponding x[n]
        fulfills the following three conditions:

        1. `x[n] == max(x[n - pre_max:n + post_max])`
        2. `x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta`
        3. `n - previous_n > wait`

        where `previous_n` is the last sample picked as a peak (greedily).
    '''
```
这是一个用于在信号中选择峰值（peaks）的函数，它采用了一种灵活的启发式方法。函数根据给定的条件选择信号中的峰值，这些条件旨在确保所选峰值具有一定的显著性和特定的时间间隔。让我解释每个条件的含义：

1. `x[n] == max(x[n - pre_max:n + post_max])`：这表示在信号 `x` 中，一个样本 `n` 被选为峰值，如果它等于在前 `pre_max` 个样本和后 `post_max` 个样本范围内的样本中的最大值。这确保了所选的峰值是局部最大值。

2. `x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta`：这表示在信号 `x` 中，一个样本 `n` 被选为峰值，如果它的值大于或等于在前 `pre_avg` 个样本和后 `post_avg` 个样本范围内的样本的平均值再加上 `delta`。这个条件确保了所选的峰值比周围的平均值要显著。

3. `n - previous_n > wait`：这个条件用于确保两个峰值之间有足够的时间间隔。 `n` 表示当前样本，`previous_n` 表示前一个已选择的峰值的样本。这个条件要求两个峰值之间的时间间隔至少为 `wait` 个样本。

因此，这个函数根据以上三个条件来选择信号中的峰值。这种方法可以用于从信号中提取出显著的峰值，例如，用于检测音频信号中的音符开始或其他信号中的突出事件。这种方法是一种启发式方法，可以根据应用的需求进行调整。

## Onset detection
(原理和上面类似)
```python
onset_frames = librosa.onset.onset_detect(y=x, sr=sr, wait=1, pre_avg=1, post_avg=1, pre_max=1, post_max=1)
print(onset_frames) # frame numbers of estimated onsets
```

``` bash
[output]:[ 20  29  38  57  65  75  84  93 103 112 121 131 140 148 158 167 176 185 204 213 232 241 250 260 268 278 288]
```

`librosa.onset.onset_strength` 和 `librosa.onset.onset_detect` 都是 Librosa 库中用于检测音频信号中的音符开始和强度的函数，但它们的功能和使用方式有一些不同。

1. `librosa.onset.onset_strength`:
   - 功能：该函数计算音频信号的"onset strength" 或 "onset envelope"，即在时间上表示音频信号中的突出事件的信号。这个信号通常用于后续的音频事件检测。
   - 参数：通常需要传递音频信号 `y` 和采样率 `sr` 作为参数，还可以提供其他参数来调整计算过程，例如 `hop_length` 和 `aggregate`.
   - 返回值：函数返回一个代表**音频信号强度**的一维数组。

2. `librosa.onset.onset_detect`:
   - 功能：这个函数使用 `librosa.onset.onset_strength` 的输出（或其他类似的音频强度信号）来检测音符开始或音频事件的时刻。它通过分析 "onset strength" 信号来查找潜在的音符开始时刻，并返回这些时刻的帧索引或时间。
   - 参数：通常需要传递音频强度信号（如通过 `librosa.onset.onset_strength` 计算得到的）作为参数，以及一些其他参数，如 `hop_length`、`backtrack` 等，来调整检测过程。
   - 返回值：函数返回一个包含**音符开始时刻的帧索引或时间**的一维数组。

总的来说，`librosa.onset.onset_strength` 用于计算音频信号的强度信号，而 `librosa.onset.onset_detect` 用于在强度信号上检测音符开始或音频事件的时刻。通常，它们一起使用，首先计算强度信号，然后使用 `librosa.onset.onset_detect` 来找到音符开始时刻。这种分离的方式允许更大的灵活性，因为您可以尝试不同的参数和强度信号来适应不同的音频分析任务。

### Onset detection with backtracking
在很多情况中，考虑到音符改变不是瞬时完成，信号能量从平稳到峰值是有一个过程的，为避免检测 onset 时将 onset 从峰值切开，我们令参数 `backtrack = True`，实现对前面局部最值的回溯，可以确保能量变化的完整记录。
```python
onset_frames = librosa.onset.onset_detect(y=x, sr=sr, hop_length=hop_length, backtrack=True)
```