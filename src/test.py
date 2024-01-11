import torch
import torch.nn as nn
import torch.nn.functional as F
# 假设参数
max_click_history = 10
n_filters = 128
filter_sizes = [3, 4, 5]

# 生成随机数据
clicked_embeddings = torch.rand((32, 10, 384))  # 32个样本，每个样本有10个点击历史，每个点击历史有128 * 3维度的嵌入
news_embeddings = torch.rand((32, 10,384))  # 32个样本，每个样本的新闻嵌入是128 * 3维度
x=torch.zeros(32, 10)
if news_embeddings.shape[1]==10:
    for i in range(10):
        y=news_embeddings[:,i:i+1,:]
        s=y.squeeze(1)
        s=torch.bmm(clicked_embeddings,s.unsqueeze(dim=-1))
# s=torch.bmm(clicked_embeddings,news_embeddings.unsqueeze(dim=-1))
        x[:,i]=s.squeeze(dim=-1)[:,i]


result = torch.bmm(clicked_embeddings, news_embeddings.transpose(1, 2))

# 从结果中提取对角线上的元素
t = result.diagonal(dim1=-2, dim2=-1)
# print(x.shape)
print(x[0])
print(t[0])
print(t.shape)

# clicked_embeddings = clicked_embeddings.view(-1, max_click_history, n_filters * len(filter_sizes))
#
# # Expand news embeddings
# news_embeddings_expanded = news_embeddings.unsqueeze(1)
#
# # Attention weights
# attention_weights = torch.sum(clicked_embeddings * news_embeddings_expanded, dim=-1)
# attention_weights = F.softmax(attention_weights, dim=-1)
#
# # Expand attention weights
# attention_weights_expanded = attention_weights.unsqueeze(-1)
#
# # User embeddings
# user_embeddings = torch.sum(clicked_embeddings * attention_weights_expanded, dim=1)
# print(user_embeddings.shape)