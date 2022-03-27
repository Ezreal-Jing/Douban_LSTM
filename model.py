# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : model.py
# @Project: DoubanNLP
# @CreateTime : 2022/3/3 下午8:00:59
# @Version：V 0.1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import lib
from dataset import get_dataloader
import os
import numpy as np
import matplotlib.pyplot as plt
'''
模型构建、训练、评估
'''


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.embedding = nn.Embedding(len(lib.ws),512)#词的数量，每个词用多少向量表示，太大容易过拟合

        #加入LSTM
        self.lstm = nn.LSTM(input_size=512,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
                batch_first=True,bidirectional=lib.bidirectional,dropout=lib.dropout)

        self.fc = nn.Linear(lib.hidden_size*2,5)#5分类问题

    def forward(self,input):
        x = self.embedding(input)#进行embedding操作，形状：[batch_size, max_len,100]
        x,(h_n,c_n) = self.lstm(x)
        output_fw = h_n[-2, :, :]#正向最后一次的输出
        output_bw = h_n[-1, :, :]#反向最后一次的输出
        output = torch.concat([output_fw,output_bw],dim=-1)

        out = self.fc(output)
        return F.log_softmax(out,dim=-1)

model = MyModel().to(lib.device)
optimizer = Adam(model.parameters(),0.001)#优化器
if os.path.exists("./cache/model.pkl"):
    model.load_state_dict(torch.load("./cache/model.pkl"))
    optimizer.load_state_dict(torch.load("./cache/optimizer.pkl"))

def train(epoch):
    train_loss_list = []


    for idx,(input,target) in enumerate(get_dataloader(train = True)):
       input = input.to(lib.device)
       target = target.to(lib.device)

       optimizer.zero_grad()#梯度归零
       output = model(input)#使用模型预测
       loss = F.nll_loss(output,target)#根据预测结果计算损失
       loss.backward()#求解梯度
       optimizer.step()#优化
       train_loss_list.append(loss.item())


       print("epoch:",epoch,"idx:",idx,"loss:",loss.cpu().item())
       #保存模型
       if idx % 99 == 0:
           torch.save(model.state_dict(),"./cache/model.pkl")
           torch.save(optimizer.state_dict(), "./cache/optimizer.pkl")
    train_loss = np.mean(train_loss_list)  # 每轮训练的平均损失
    return train_loss



# def plot_show():
#     x1 = range(0, 10)
#     y1 = train(10)
#     plt.plot(x1, y1)
#     plt.title("train loss vs. epoches")
#     plt.xlabel("epoches")
#     plt.ylabel("train loss")
#     plt.show()







def eval():#模型评估
    loss_list = []
    acc_list = []

    for idx, (input, target) in enumerate(get_dataloader(train=False,batch_size=lib.test_batch_size)):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad(): # 评估时不需要梯度改变 优化
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss.cpu().item())
            # 计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print("total loss:", np.mean(loss_list), "total acc:", np.mean(acc_list))
    return np.mean(loss_list),np.mean(acc_list)







if __name__ == '__main__':
    EPOCH = 10
    train_loss = []
    test_loss = []
    test_acc = []
    for i in range(EPOCH):
        train(i)
        a,b = eval()
        train_loss.append(train(i))
        test_loss.append(a)
        test_acc.append(b)


    #可视化
    x = range(0, EPOCH)

    y1 = train_loss
    plt.plot(x, y1)
    plt.title("train loss vs. epoches")
    plt.xlabel("epoches")
    plt.ylabel("train loss")
    plt.show()

    y2 = test_loss
    plt.plot(x, y2)
    plt.title("test loss vs. epoches")
    plt.xlabel("epoches")
    plt.ylabel("test loss")
    plt.show()

    y3 = test_acc
    plt.plot(x, y3)
    plt.title("test acc vs. epoches")
    plt.xlabel("epoches")
    plt.ylabel("test acc")
    plt.show()