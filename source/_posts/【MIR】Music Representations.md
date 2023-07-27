---
title: 【MIR】Music Representations
date: 2023-07-27 15:47:55
tags: [音乐信息检索、乐理]
categories:
 - Music Information Retrieval
---
# Sheet Music Representations 乐谱表示
音乐可以用多种不同的方式来表现。音乐作品的印刷视觉形式称为**乐谱**或**乐谱**。

乐谱由**音符**组成。音符具有多种属性，包括音高、音色、响度和持续时间。

**音高**（[Wikipedia]( https://en.wikipedia.org/wiki/Pitch_ (music)）是一种感知属性，指示音符听起来有多“高”或“低”。音高与音符发出的基频，尽管基频是声波的物理属性。
**八度**（[维基百科](https://en.wikipedia.org/wiki/Octave)）是两个音符之间的间隔，其中较高音符是较低音符基频的两倍。例如，440 Hz 的 A 和 880 Hz 的 A 相隔一个八度。
**Pitch Class**（[Wikipedia](https://en.wikipedia.org/wiki/Pitch_class)）是相隔整数个八度音阶的所有音符的集合。例如，所有 C 的集合 {..., C1, C2, ...} 是一个音级，所有 D 的集合 {..., D1, D2, ...} 是另一个音高班级。
**平均律**（[维基百科](https://en.wikipedia.org/wiki/Equal_temperament)）是指将八度音阶划分为 12 个统一音阶的标准做法。
两个后续音阶步长之间的差异称为**半音**（[维基百科](https://en.wikipedia.org/wiki/Semitone)），是 12 音等律音阶中可能的最小音程。音乐家可能将其称为“半步”。
**Key Signature 调号**（[Wikipedia](https://en.wikipedia.org/wiki/Key_signature)）遵循五线谱上的谱号，并通过整个乐曲中存在的升号或降号来指示乐曲的调号。
**Time Signature 拍号**（[维基百科](https://en.wikipedia.org/wiki/Time_signature)）遵循五线谱上的调号，指示乐曲的节奏结构或韵律。
**Tempo 节奏**（[维基百科](https://en.wikipedia.org/wiki/Tempo)）表示一首乐曲的演奏速度，以每分钟节拍 (BPM) 来衡量。

# Symbolic Representations 符号表示
## 符号音乐表示
包括具有音符或其他音乐事件的**显式编码**的任何类型的乐谱表示。其中包括机器可读的数据格式，例如 MIDI。任何类型的数字数据格式都可以被视为符号性的，因为它**基于有限的字母或符号字母表**。
钢琴卷帘：最早用于自弹钢琴中，如今常出现于计算机编曲宿主软件中。

## MIDI 表示 
另一种符号表示基于 **MIDI** 标准（[维基百科](https://en.wikipedia.org/wiki/MIDI)）或乐器数字接口。 1981-83 年 MIDI 的出现引起了电子乐器市场的快速增长。

MIDI 消息对每个音符事件的信息进行编码，例如**音符开始**、**音符偏移**和**强度**（在 MIDI 术语中表示为“**速度**”）。在计算机上，MIDI 文件包含 MIDI 消息和其他元数据的列表。
**MIDI 音符编号** 是 0 到 127 之间的整数，用于编码音符的音高。最重要的是，C4（中 C）的 MIDI 音符编号为 60，A4（音乐会 A440）的 MIDI 音符编号为 69。以 12 分隔的 MIDI 音符编号以一个八度分隔，例如72 = C5，84 = C6，等等。
**强度 Key velocity**是 0 到 127 之间的整数，用于控制声音的强度。
**MIDI 通道** 是 0 到 15 之间的整数，提示合成器使用特定乐器。
MIDI 将四分音符细分为 **时钟脉冲** 或 **刻度**。例如，如果每四分音符 (PPQN) 的脉冲数定义为 120，则 60 个刻度代表八分音符的长度。

## Score Representations 乐谱标记*
**乐谱表示**** 对有关音乐符号的明确信息进行编码，例如谱号、拍号、调号、音符、休止符、动态等。
但是，乐谱表示（我们在此定义的方式）不包括对最终乐谱的任何描述这些符号在页面上的视觉布局和位置。

**MusicXML** 已成为存储音乐文件以供不同乐谱应用程序使用的通用格式。以下是 MusicXML 文件的摘录：
```musicxml
        <measure number="2">
            <note>
                <pitch>
                    <step>B</step>
                    <alter>-1</alter>
                    <octave>4</octave>
                </pitch>
                <duration>1</duration>
                <voice>1</voice>
                <type>16th</type>
                <stem>down</stem>
                <beam number="1">begin</beam>
                <beam number="2">begin</beam>
            </note>
            <note>
                <pitch>
                    <step>D</step>
                    <octave>5</octave>
                </pitch>
                <duration>1</duration>
                <voice>1</voice>
                <type>16th</type>
                <stem>down</stem>
                <beam number="1">continue</beam>
                <beam number="2">continue</beam>
            </note>
            ...
        </measure>
```

# Audio Representation
**音频**是指人类可以听到的声音的产生、传输或接收。**音频信号**是声音的表示，其表示由振动引起的空气压力的波动作为时间的函数。与乐谱或符号表示不同，音频表示对再现音乐的声学实现所需的一切进行编码。但是，注意参数（如起始点、持续时间和音高）并没有明确编码。这使得从音频表示转换为符号表示成为一项困难且定义不清的任务。

## 时域波形
读取、显示波形
```python
x,sr = librosa.load('audio/c_strum.wav')
plt.figure(figsize=(15, 5))
librosa.display.waveshow(x, alpha=0.8)
```

## Timbre : Temporal Indicators （时域）

音色的一个特点是它的时间演变。信号的**包络**（envelope）是一条平滑的曲线，它近似于波形随时间的振幅极值。
包络通常由**ADSR 模型**（[Wikipedia]（https:\/\/en.Wikipedia.Org/wiki\/Synthesizer #Attack_Decay_Sustain_Release_ .28 ADSR.29_envelope））建模，该模型描述了声音的四个阶段：攻击 attack、衰减 decay、维持 sustain 和释放 release。
在攻击阶段，声音会积累起来，通常在较宽的频率范围内带有类似噪音的成分。在一个声音开始时，这种类似噪音的短持续时间声音通常被称为瞬态。
在衰减阶段，声音稳定并达到稳定的周期性模式。
在维持阶段，能量保持相当恒定。
在释放阶段，声音会逐渐消失。
ADSR 模型是一种简化，不一定对所有声音的振幅包络进行建模。

## Timbre: Spectral Indicators (频域)
用于表征音色的另一个特性是分音的存在及其相对强度**声部**是音调中的主频，最低的声部是**基频**。
声音的分音是用**声谱图**来显示的。频谱图显示了频率分量随时间的强度。（有关详细信息，请参阅[傅立叶变换]（Fourier_Transform.Html）和[短时傅里叶变换]（stft.Html）。）

## Pure Tone
```python
# 合成C6高音
T = 2.0 # seconds
f0 = 1047.0
sr = 22050
t = numpy.linspace(0, T, int(T*sr), endpoint=False) # time variable
x = 0.1*numpy.sin(2*numpy.pi*f0*t)
ipd.Audio(x, rate=sr)
# 展示频谱
X = scipy.fft(x[:4096])
X_mag = numpy.absolute(X)        # spectral magnitude
f = numpy.linspace(0, sr, 4096)  # frequency variable
plt.figure(figsize=(14, 5))
plt.plot(f[:2000], X_mag[:2000]) # magnitude spectrum
plt.xlabel('Frequency (Hz)')
```

# Tuning Systems 调音系统

## Introduction
In twelve-tone **equal temperament** ([Wikipedia](https://en.wikipedia.org/wiki/Equal_temperament)), all twelve semitones within the octave have the same width. With this tuning system, expressed as a frequency ratio, the interval of one semitone is $2^{1/12}$. Expressed in **cents**, this same interval is defined to be 100 cents. Therefore, the octave has 1200 cents.
在十二音**平均律**（[维基百科](https://en.wikipedia.org/wiki/Equal_temperament)）中，八度内的所有十二个半音都具有相同的宽度。在这个调音系统中，以**频率比**表示，一个半音的音程为 $2^{1/12}$。以 ** 美分 ** 表示，该间隔定义为 100 美分。因此，八度有 1200 音分。
In **just intonation** ([Wikipedia](https://en.wikipedia.org/wiki/Just_intonation)), the frequency ratio is expressed as a fraction between two small integers, e.g. 3:2, 4:3. As a result, the higher harmonic partials between two notes will overlap, resulting in a consonant interval that is pleasing to the ear. In **5-limit just tuning**, these fractions are expressed with prime factors no larger than 5, i.e. {2, 3, 5}. In **7-limit just tuning**, these fractions are expressed with prime factors no larger than 7, i.e. {2, 3, 5, 7}. For example, 7:4 is a 7-limit interval, but it is not a 5-limit interval.
在**Just intonation**（[维基百科](https://en.wikipedia.org/wiki/Just_intonation)）中，频率比表示为两个小整数之间的分数，例如3:2、4:3。结果，两个音符之间的高次谐波分音将重叠，从而产生悦耳的辅音音程。在 **5-limit just adjustment** 中，这些分数用不大于 5 的质因数表示，即 {2, 3, 5}。在 **7-limit just adjustment** 中，这些分数用不大于 7 的质因数表示，即 {2, 3, 5, 7}。例如，7:4 是 7 限区间，但不是 5 限区间。
In **Pythagorean tuning** ([Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_tuning)), every frequency ratio is based upon the ratio 3:2. To find that ratio, from one note in the interval, step around the Circle of Fifths until you reach the other note in the interval, multiplying (if stepping forward) or dividing (if stepping backward) by 3/2 with each step. Finally, multiply or divide by 2 enough times to return to the octave of interest. Pythagorean tuning can also be considered **3-limit just tuning** since every ratio only uses prime factors no greater than 3.
在**毕达哥拉斯调谐**（[维基百科](https://en.wikipedia.org/wiki/Pythagorean_tuning)）中，每个频率比都基于比率 3:2。要找到该比率，请从音程中的一个音符开始，绕五度圈移动，直到到达音程中的另一个音符，每一步乘以（如果向前迈进）或除（如果向后迈进）3/2。最后，乘以或除以 2 足够多次以返回到感兴趣的八度音阶。毕达哥拉斯调优也可以被认为是 **3 极限调优**，因为每个比率仅使用不大于 3 的质因数。

## MIDI 音符-频率转换表
- **note**: note name
- **midi-ET**: MIDI number, equal temperament
- **Hertz-ET**: frequency in Hertz, equal temperament （平均律）
- **midi-PT**: MIDI number, Pythagorean tuning
- **Hertz-PT**: frequency in Hertz, Pythagorean tuning （毕 调谐）
```python
note_pt = dict()

# Sharps

note_pt['A4'] = 440.0
for octave in range(0, 10):
    note_pt['A{}'.format(octave)] = 440.0*2**(octave-4)

note_pt['E1'] = 1.5*note_pt['A0']
for octave in range(0, 10):
    note_pt['E{}'.format(octave)] = note_pt['E1']*2**(octave-1)
    
note_pt['B0'] = 1.5*note_pt['E0']
for octave in range(0, 10):
    note_pt['B{}'.format(octave)] = note_pt['B0']*2**(octave-0)

note_pt['F#1'] = 1.5*note_pt['B0']
for octave in range(0, 10):
    note_pt['F#{}'.format(octave)] = note_pt['F#1']*2**(octave-1)

note_pt['C#1'] = 1.5*note_pt['F#0']
for octave in range(0, 10):
    note_pt['C#{}'.format(octave)] = note_pt['C#1']*2**(octave-1)

note_pt['G#0'] = 1.5*note_pt['C#0']
for octave in range(0, 10):
    note_pt['G#{}'.format(octave)] = note_pt['G#0']*2**(octave-0)

note_pt['D#1'] = 1.5*note_pt['G#0']
for octave in range(0, 10):
    note_pt['D#{}'.format(octave)] = note_pt['D#1']*2**(octave-1)

note_pt['A#0'] = 1.5*note_pt['D#0']
for octave in range(0, 10):
    note_pt['A#{}'.format(octave)] = note_pt['A#0']*2**(octave-0)

note_pt['E#1'] = 1.5*note_pt['A#0']
for octave in range(0, 10):
    note_pt['E#{}'.format(octave)] = note_pt['E#1']*2**(octave-1)

note_pt['B#0'] = 1.5*note_pt['E#0']
for octave in range(0, 10):
    note_pt['B#{}'.format(octave)] = note_pt['B#0']*2**(octave-0)
    
# Flats

note_pt['D0'] = 2/3*note_pt['A0']
for octave in range(0, 10):
    note_pt['D{}'.format(octave)] = note_pt['D0']*2**octave

note_pt['G0'] = 2/3*note_pt['D1']
for octave in range(0, 10):
    note_pt['G{}'.format(octave)] = note_pt['G0']*2**octave

note_pt['C0'] = 2/3*note_pt['G0']
for octave in range(0, 10):
    note_pt['C{}'.format(octave)] = note_pt['C0']*2**octave

note_pt['F0'] = 2/3*note_pt['C1']
for octave in range(0, 10):
    note_pt['F{}'.format(octave)] = note_pt['F0']*2**octave

note_pt['Bb0'] = 2/3*note_pt['F1']
for octave in range(0, 10):
    note_pt['Bb{}'.format(octave)] = note_pt['Bb0']*2**octave

note_pt['Eb0'] = 2/3*note_pt['Bb0']
for octave in range(0, 10):
    note_pt['Eb{}'.format(octave)] = note_pt['Eb0']*2**octave

note_pt['Ab0'] = 2/3*note_pt['Eb1']
for octave in range(0, 10):
    note_pt['Ab{}'.format(octave)] = note_pt['Ab0']*2**octave

note_pt['Db0'] = 2/3*note_pt['Ab0']
for octave in range(0, 10):
    note_pt['Db{}'.format(octave)] = note_pt['Db0']*2**octave

note_pt['Gb0'] = 2/3*note_pt['Db1']
for octave in range(0, 10):
    note_pt['Gb{}'.format(octave)] = note_pt['Gb0']*2**octave

note_pt['Cb0'] = 2/3*note_pt['Gb0']
for octave in range(0, 10):
    note_pt['Cb{}'.format(octave)] = note_pt['Cb0']*2**octave

note_pt['Fb0'] = 2/3*note_pt['Cb1']
for octave in range(0, 10):
    note_pt['Fb{}'.format(octave)] = note_pt['Fb0']*2**octave

sorted_notes = sorted(note_pt.items(), key=lambda x:x[1])

markdown = """|note|midi-ET|Hertz-ET|midi-PT|Hertz-PT|\n"""
markdown += """|----|----|-----|----|----|-----|\n"""
    
for note, f_pt in sorted_notes:
    
    midi_et = librosa.note_to_midi(note)
    f_et = librosa.midi_to_hz(midi_et)
    
    midi_pt = librosa.hz_to_midi(f_pt)
    
    if note.startswith('A') and midi_et % 12 == 9:
        ipd.display_markdown(markdown, raw=True)
        markdown = """|note|midi-ET|Hertz-ET|midi-PT|Hertz-PT|\n"""
        markdown += """|----|----|-----|----|----|-----|\n"""
    
    markdown += """|{}|{}|{:.5g}|{:.3f}|{:.5g}|\n""".format(
        note, midi_et, f_et, midi_pt, f_pt
    )
    
ipd.display_markdown(markdown, raw=True)
```

# Understanding Audio Features through Sonification 
目的：
1. 检测音频信号中的起始点。
2. 在每次开始时对音频信号进行分段。
3. 计算每个分段的特征。
4. 通过单独聆听每个片段来直观地了解其功能。