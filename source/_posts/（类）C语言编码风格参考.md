---
title: （类）C语言编码风格参考
date: 2023-01-27 21:21:34
tags: [C语言,coding]
---
>近期类C语言码量增多，意识到有必要学习一定的代码风格，闲来无聊不断查阅资料，在下面整理一个简单的阅读笔记，（仅自用）

#### 头文件
1. 头文件放置接口声明
2. 一个.c对应一个.h，声明对外公开的接口
3. `#define`保护
   ```c
   #ifndef __XXX_H__
   #define __XXX_H__
   ...
   #endif
   ```
4. 头文件排列方式：功能块排序、文件名排序、稳定度排序
#### 函数
1. 一个函数仅完成一个功能
2. 重复代码应提炼为函数
3. 函数不变参数使用const，使代码更牢固/更安全
#### 标识符命名与定义
1. 除通用缩写，不使用单词缩写，不使用汉语拼音
   argument可缩写为arg//变量
buffer可缩写为buff//缓冲数据
clock可缩写为clk
command可缩写为cmd
compare可缩写为cmp
configuration可缩写为cfg
device可缩写为dev
error可缩写为erI
hexadecimal可缩写为hex
increment可缩写为inc//增量
initialize可缩写为init
maximum可缩写为max
message可缩写为msg
minimum可缩写为min
parameter可缩写为para
previous可缩写为prev
register可缩写为reg
semaphore可缩写为sem//信号量
statistic可缩写为stat
synchronize可缩写为sync
temp可缩写为tmp//临时
2. 文件命名统一采用小写字符，因为不同系统对文件名大小写处理不同
3. 全局变量`g_`前缀，静态变量`s_`前缀，前缀使全局变量的名字显得很丑陋，促使开发人员尽量少使用全局变量
4. 禁止单字节命名变量，允许i,j,k作为局部循环变量
5. 对于数值或者字符串等等常量的定义，建议采用全大写字母，单词之间加下划线`_`的方式命名（枚举同样建议使用此方式定义），除头文件或编译开关的特殊标识，宏定义不能采用`_`开头或结尾
#### 变量
1. 严禁使用未经初始化的变量作为右值。
   说明：坚持建议4.3（在首次使用前初始化变量，初始化的地方离使用的地方越近越好。）可以有效避免未初始化错误。
#### 宏、常量
1. 用宏定义表达式要使用完备的括号
   例：`#define RECTANGLE_AREA(a,b) ((a) * (b))`
2. 将宏所定义的多条表达式放在大括号中
    ```c
    #define F00(x) do{\
        printf("arg is %s\n", x);\
        do_something;
   } while(0)
   ```
   完全不用担心使用者如何使用宏
   [do{...}while(0)的用法](https://blog.csdn.net/dldw8816/article/details/86519575)

#### 注释
1. 注释应放在其代码上方相邻位置或右方，不可放在下面。如放于上方则需与其上面的代码用空行隔开，且与下方代码缩进相同。
2. 建议采用工具可识别的注释格式，例如doxygen格式
#### 排版与格式
1. 缩进风格编写，每级缩进4个空格
2. 一行一条语句
3. 返回类型和函数名在同一行, 参数也尽量放在同一行, 如果放不下就对形参分行, 分行方式与 函数调用 一致.
4. 条件语句
   倾向于不在圆括号内使用空格，if和else另起一行，有两种可以接受的格式，一种是圆括号与条件之间有空格，另一种没有，if和圆括号之间，圆括号大括号之间需要有空格
5. 循环和选择语句中空循环体应使用`{}`或者`continue`而并不是简单的分号
6. 句点或箭头前后不要有空格. 指针/地址操作符 (*, &) 之后不能有空格.
7. **左大括号位于行尾**
   
### 参考资料
[华为技术有限公司内部技术规范](https://rapidupload.1kbtool.com/8228deae207f990d0212fdf12e087fc7#13268772#%E5%8D%8E%E4%B8%BA%E6%8A%80%E6%9C%AF%E6%9C%89%E9%99%90%E5%85%AC%E5%8F%B8c%E8%AF%AD%E8%A8%80%E7%BC%96%E7%A8%8B%E8%A7%84%E8%8C%83_%E5%8D%8E%E4%B8%BA_zhelper-search.pdf)
[Google开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/contents/)