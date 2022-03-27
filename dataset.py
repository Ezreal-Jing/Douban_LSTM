# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : dataset.py
# @Project: DoubanNLP
# @CreateTime : 2022/2/22 下午8:35:26
# @Version：V 0.1

#数据集的准备
import torch
from torch.utils.data import DataLoader,Dataset
import re
from ltp import LTP
import pandas as pd
import torch
import jieba
import lib

def tokenize(content):
    filters = ['\t','\n','\x97','\x96','#','$','%','&',':','，','。','\.','“','”','"','《','》'," ","@","、","-","（","）","0","1","2","3","4","5","6","7","8","9"]
    content = re.sub("|".join(filters),"",content)
    # ltp = LTP()
    # segment, _ = ltp.seg([content]) #中文分词
    seg_list = jieba.cut(content)  # 默认是精确模式
    segment = [i for i in seg_list] # 用结巴比较快，默认是精确模式
    return segment


class DoubanDataset(Dataset):
    def __init__(self,train = True):
        self.trian_data_path = r"./data/train.csv"
        self.test_data_path = r"./data/test.csv"
        if train:
            self.df = pd.read_csv(self.trian_data_path, encoding='utf-8')
        else:
            self.df = pd.read_csv(self.test_data_path,encoding= 'utf-8')
        self.comments = self.df['comments'].tolist()
        for i in range(len(self.comments)):
            self.comments[i] = tokenize(self.comments[i])
        self.label = self.df['rating'].tolist()

    def __getitem__(self, index):#__getitem__做的事情就是返回第index个样本的具体数据:

        return self.comments[index],int(self.label[index]-1)

    def __len__(self):
        return len(self.df)#csv的行数即为数据集的数量


def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    content,label = list(zip(*batch))
    content =[lib.ws.transform(i,max_len = lib.max_len) for i in content]#注意这里返回的是list
    content = torch.LongTensor(content)#将list转换为tensor
    label = torch.LongTensor(label)
    return content,label



def get_dataloader(train=True,batch_size = lib.batch_size):
    douban_dataset = DoubanDataset(train)
    data_loader = DataLoader(douban_dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
    return data_loader



if __name__ == '__main__':
    for idx,(input,target) in enumerate(get_dataloader()):
        print(idx)
        print(input)
        print(target)
        break

