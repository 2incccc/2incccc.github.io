---
title: 【Machine Learning】 Week 1 Introduction to Machine Learning
date: 2023-07-15 17:48:16
tags: [机器学习,吴恩达]
categories: 
- Machine Learning Specialization
---
# Week 1    Introduction to Machine Learning

> Welcome to the Machine Learning Specialization! You're joining millions of others who have taken either this or the original course, which led to the founding of Coursera, and has helped millions of other learners, like you, take a look at the exciting world of machine learning!


## Overview of Machine Learning

In this class, one of the relatively unique things you learn is you learn a lot about the best practices for how to actually develop a practical, valuable machine learning system.

## Supervised vs. Unsupervised Machine Learning

Machine learning algorithms 
- Supervised learning  (Course 1 & 2)
- Unsupervised learning (Course 3)
- Recommender systems (Course 3)
- Reinforcement learning (Course 3)

### Supervised learning （有监督学习）
**Characteristics**: Learn from being told the right answers, and find the correct answer
**Application**: spam filtering; speech recognition; machine translation; online advertising; silf-driving car; visual inspection

Including:
**Regression（回归）**: predict a number from infinitely many possible numbers
**Classification（分类）**: predict categories, which don't have to be numbers

### Unsupervised learning （无监督学习）
**Characteristics**: Not being told the right answer, and find something intersting(patterns of structures) in unlabeled data, or to say, place the unlabeled data, into different clusters 

**Formal definition**: in unsupervised learning, the data comes only with inputs x but not output labels y, and the algorithm has to find some structure or some pattern or something interesting in the data.

Including:
**Clustering**（集群）: Group similar data points together
**Dimensionality reduction**（降维）: Compress data using fewer numbers
**Anomaly detection**（异常检测）: Find unusual data points

### Jupyter Notebooks
The most widely used tool by machine learning and data science practitioners

Optional labs: open and run one line at a time with usually no need to write any code yourself
Practice labs: give you an opportunity to write some of that code yourself

## Regression Model
Regression model predicts nubers
Classification model predicts categories

### Terminology 术语
Training set: 训练集 The dataset that you just saw and that is used to train the model is called a training set
input features 输入 output targets 输出

单变量线性回归 Univariate linear regression

### Cost function
Modle: $f_{w,b}(x)=wx+b$  Parameters: $w,b$
$\hat y^{(i)} = f_{w,b}(x^{(i)})=wx^{(i)}+b$

Square error cost function: 均方差代价函数
$$J(w,b) = \frac{1}{2m}\sum\limits_{i=1}^m(\hat y^{(i)}-y^{(i)})^2$$
Goal: $\min_{w,b} J(w,b)$

## Train the model with gradient descent 梯度下降法 
Formula: $w=w-\alpha \frac{\partial}{\partial w}J(w,b)$  (b for the same)
$\alpha$ : Learning rate 学习率
$\frac{\partial}{\partial w}J(w,b)$ : （partial）derivative 导数

Simultaneous update: 正确做法，（w,b 同时变化）
$tmp\_w=w-\alpha \frac{\partial}{\partial w}J(w,b)$
$tmp\_b-b-\alpha \frac{\partial}{\partial b}J(w,b)$
$w=tmp\_w$
$b=tmp\_b$
**Local minimum 局部极小值**
多个极小值情况下，到达非最值的局部极小值，w 将保持不变
解决方法：调整带入训练的样本数量，通过局部规律的不一致性来规避“非全域最小值但梯度为 0”的陷阱，这就是所谓的**随机梯度下降和小批量梯度下降**

#### Gradient descent for linear regression 线性回归下的梯度下降法
$w = w-\alpha \frac{1}{m}\sum\limits_{i=1}^m(f_{w,b}(x^{(i)}-y^{(i)})x^{(i)}$
$b=b-\alpha \frac{1}{m}\sum\limits_{i=1}^m(f_{w,b}(x^{(i)}-y^{(i)})$

#### Batch gradient descent 批处理
Batch: Each step of gradient descnet uses all the training examples 使用全部的训练集
other: subsets 


