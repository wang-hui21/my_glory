# -*- coding: utf-8 -*-
# @Time : 2023/11/27 16:06
# @Author : Wang Hui
# @File : test.py
# @Project : GLORY

from collections import defaultdict

# 使用 defaultdict 初始化字典，值的默认工厂是列表
my_dict = defaultdict(list)
with open('news.tsv', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 20:
            words = line.split()
            my_dict[words[1]].append(words[0])
    for key, value in my_dict.items():
        print(value)