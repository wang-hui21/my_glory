# -*- coding: utf-8 -*-
# @Time : 2023/11/27 16:06
# @Author : Wang Hui
# @File : test.py
# @Project : GLORY

from collections import defaultdict

# 使用 defaultdict 初始化字典，值的默认工厂是列表
# my_dict = defaultdict(list)
# with open('news.tsv', 'r', encoding='utf-8') as f:
#     for i, line in enumerate(f):
#         if i < 20:
#             words = line.split()
#             my_dict[words[1]].append(words[0])
#     for key, value in my_dict.items():
#         print(value)

import torch

# 假设你有一个张量 tensor
tensor = torch.tensor([[0, 1, 0], [2, 0, 3], [0, 4, 0]])

# 创建一个与 tensor 形状相同的掩码矩阵，不为零的元素设为1
mask = torch.where(tensor != 0, torch.tensor(1), torch.tensor(0))

print(mask)