# 基于LSTM的豆瓣影评情感分析
## 简介

本项目基于Bi-LSTM模型，实现对豆瓣电影短评的情感分类。

## 使用说明
如果不更改数据集，直接运行model.py进行训练即可。
数据集在data文件夹下，两个csv文件，分别为训练集和测试集。
------

1.word2sequence.py——构建将文本转换为序列的方法

2.build_vocab——使用word2sequence按词频构建词表，注意调用了dataset.py中的数据集

3.dataset.py——使用pytorch内置的方法导入数据集，并进行数据过滤、分词、转换为Long tensor等操作

4.model.py——模型构建、训练、评估以及简单可视化

5.lib.py——用于定义相关参数，单独写出便于修改

## 具体细节

待更新
