# -*- coding: utf-8 -*-
# @Time : 2024/1/4 21:11
# @Author : Wang Hui
# @File : sen_attention
# @Project : my_glory
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, args):
        super(AttentionModel, self).__init__()
        self.kcnn = KCNN(args)  # Assuming you have a KCNN model

    def forward(self, clicked_words, clicked_entities, news_words, news_entities, args):
        # Reshape clicked words and entities
        clicked_words = clicked_words.view(-1, args.max_title_length)
        clicked_entities = clicked_entities.view(-1, args.max_title_length)

        # KCNN for user and news embeddings
        clicked_embeddings = self.kcnn(clicked_words, clicked_entities, args)
        news_embeddings = self.kcnn(news_words, news_entities, args)

        # Reshape clicked embeddings
        clicked_embeddings = clicked_embeddings.view(-1, args.max_click_history, args.n_filters * len(args.filter_sizes))

        # Expand news embeddings
        news_embeddings_expanded = news_embeddings.unsqueeze(1)

        # Attention weights
        attention_weights = torch.sum(clicked_embeddings * news_embeddings_expanded, dim=-1)
        attention_weights = F.softmax(attention_weights, dim=-1)

        # Expand attention weights
        attention_weights_expanded = attention_weights.unsqueeze(-1)

        # User embeddings
        user_embeddings = torch.sum(clicked_embeddings * attention_weights_expanded, dim=1)

        return user_embeddings, news_embeddings
