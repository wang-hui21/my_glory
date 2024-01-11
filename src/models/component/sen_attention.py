# -*- coding: utf-8 -*-
# @Time : 2024/1/4 21:11
# @Author : Wang Hui
# @File : sen_attention
# @Project : my_glory
import torch
import torch.nn as nn
import torch.nn.functional as F



# Reshape clicked embeddings
clicked_embeddings = torch.rand((16,50,400))
news_embeddings = torch.rand((16,5,400))
# Expand news embeddings
news_embeddings_expanded = news_embeddings.unsqueeze(1)

# Attention weights
attention_weights = torch.sum(clicked_embeddings * news_embeddings_expanded, dim=-1)
attention_weights = F.softmax(attention_weights, dim=-1)

# Expand attention weights
attention_weights_expanded = attention_weights.unsqueeze(-1)

# User embeddings
user_embeddings = torch.sum(clicked_embeddings * attention_weights_expanded, dim=1)

