# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : word2sequence.py
# @Project: DoubanNLP
# @CreateTime : 2022/2/27 下午8:14:41
# @Version：V 0.1
"""
文本序列化
"""
class Word2Sequence:
    UNK_TAG = "UNK"#未出现的词
    PAD_TAG = "PAD"#填充词

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG:self.UNK,
            self.PAD_TAG:self.PAD
        }
        self.count = {}#统计词频

    def fit(self, sentence):
        '''
        统计词频
        :param sentence:
        :return:
        '''
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1#将word存储到count里

    def build_vocab(self, min_count=0, max_count=None, max_features=10000):
        """
        根据条件构建 词典
        :param min_count:最小词频
        :param max_count: 最大词频
        :param max_features: 一共保留多少个词语
        :return:
        """

        #删除count中词频小于min的word
        if min_count is not None:
            self.count = {word: count for word, count in self.count.items() if count > min_count}
        if max_count is not None:
            self.count = {word: count for word, count in self.count.items() if count < max_count}
        if max_features is not None:
            # 排序
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)  # 每次word对应一个数字

            # 把dict进行翻转
            self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        '''
        把句子转化为数字序列
        :param sentence:
        :return:
        '''
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG]*(max_len-len(sentence))#填充
            if max_len < len(sentence):
                sentence = sentence[:max_len]#裁剪

        return [self.dict.get(word,self.UNK) for word in sentence]

    def inverse_transform(self, incides):
        """
        把数字序列转化为字符
        :param incides:
        :return:
        """
        return [self.inverse_dict.get(i) for i in incides]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    sentences = [["今天", "天气", "很", "好"],
                 ["今天", "去", "吃", "什么"]]

    ws = Word2Sequence()
    for sentence in sentences:
        ws.fit(sentence)
    ws.build_vocab(min_count=0)
    print(ws.dict)

    ret = ws.transform(["今天","天气","还","不错"],max_len=10)
    print(ret)