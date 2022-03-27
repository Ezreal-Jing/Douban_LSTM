# -*- coding: utf-8 -*-
# @Author : Ezreal
# @File : test.py
# @Project: DoubanNLP
# @CreateTime : 2022/2/19 下午9:30:21
# @Version：V 0.1
#代码调试区


from torch.utils.data import DataLoader,Dataset
import re
from ltp import LTP#哈工大的分词，巨慢
import pandas as pd
import os
import jieba


trian_data_path = r"train.csv"
test_data_path = r"test.csv"
df = pd.read_csv(trian_data_path, encoding='utf-8')
comments = df['comments'].tolist()
label = df['rating'].tolist()



def tokenize(content):
    filters = ['\t','\n','\x97','\x96','#','$','%','&',':','，','。','\.','“','”','"'," "]
    content = re.sub("|".join(filters),"",content)
    seg_list = jieba.cut(content)  # 默认是精确模式
    segment = [i for i in seg_list]
    return segment
pd.set_option('display.width',None)
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)
print(df)
