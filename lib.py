# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : lib.py
# @Project: DoubanNLP
# @CreateTime : 2022/3/3 下午8:03:56
# @Version：V 0.1
import pickle
import torch


ws = pickle.load(open("./cache/ws.pkl","rb"))#初次运行记得注释掉

max_len = 100#句子的最大长度
batch_size = 480#48000条数据，100个batch
test_batch_size = 1200#12000条数据，10个batch

hidden_size = 128#隐藏层数
num_layers = 2#网络层数
bidirectional = True#双向LSTM
dropout = 0.5

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

