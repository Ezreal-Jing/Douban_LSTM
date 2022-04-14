# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : test.py
# @Project: DoubanNLP
# @CreateTime : 2022/2/19 下午9:30:21
# @Version：V 0.1
'''
保存词表
'''

#这一步单独写出来是因为pickle包有问题

import pickle
from tqdm import tqdm#打印进度
from word2sequence import Word2Sequence
from dataset import DoubanDataset

if __name__ == '__main__':

    ws = Word2Sequence()
    douban_dataset = DoubanDataset()
    for i in tqdm(douban_dataset.comments):
        ws.fit(i)
    ws.build_vocab()
    pickle.dump(ws,open("./cache/ws.pkl","wb"))
    print(len(ws))
    print(ws.dict)