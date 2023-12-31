# -*- coding: utf-8 -*-
# @Time : 2023/12/12 10:39
# @Author : Wang Hui
# @File : user_lstm
# @Project : my_glory
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 假设每个embedding的维度是embedding_dim
# embedding_dim = 128
# sequence_length = 50
# batch_size = 16
#
# # 创建模拟的输入数据
# # embeddings是一个形状为(batch_size, sequence_length, embedding_dim)的张量
# # mask是一个形状为(batch_size, sequence_length)的二元掩码矩阵，标记哪些位置有有效的值
# embeddings = torch.randn(batch_size, sequence_length, embedding_dim)
# mask = torch.randint(0, 2, (batch_size, sequence_length))


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, cfg):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(cfg.model.lstm_news_dim, cfg.model.lstm_news_dim, batch_first=True)

    def forward(self, x, mask):
        # 将输入打包成可变长度序列
        lengths = mask.sum(dim=1)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # LSTM前向传播
        packed_output, _ = self.lstm(packed_input)

        # 解压缩输出序列
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # 假设 output 是你从 LSTM 模型获得的输出张量

        # 计算每个序列中最后一个非零时间步的索引
        last_non_zero_index = (output != 0).sum(dim=1) - 1
        first_element_per_batch = last_non_zero_index[:, 0]
        # 通过索引获取最后一个非零时间步的隐藏状态
        last_non_zero_hidden_states = torch.stack([output[i, index, :] for i, index in enumerate(first_element_per_batch)])

        # last_non_zero_hidden_states 就是包含每个序列最后一个非零时间步的隐藏状态的张量

        return last_non_zero_hidden_states


# 创建模型并进行前向传播
# hidden_size = 400
# lstm_model = LSTMModel(embedding_dim, hidden_size)
# output_sequence = lstm_model(embeddings, mask)
#
# # 输出的output_sequence是形状为(batch_size, sequence_length, hidden_size)的张量
# print(output_sequence.shape)
