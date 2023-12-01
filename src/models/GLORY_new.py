# -*- coding: utf-8 -*-
# @Time : 2023/11/30 16:11
# @Author : Wang Hui
# @File : GLORY_new
# @Project : GLORY
import torch
import torch.nn as nn
from torch_geometric.nn import Sequential, GatedGraphConv

from models.base.layers import *
from models.component.candidate_encoder import *
from models.component.click_encoder import ClickEncoder
from models.component.entity_encoder import EntityEncoder, GlobalEntityEncoder
from models.component.nce_loss import NCELoss
from models.component.news_encoder import *
from models.component.user_encoder import *
import pickle


class GLORY(nn.Module):
    def __init__(self, cfg, glove_emb=None, entity_emb=None):
        super().__init__()

        self.cfg = cfg
        self.use_entity = cfg.model.use_entity

        self.news_dim = cfg.model.head_num * cfg.model.head_dim
        self.entity_dim = cfg.model.entity_emb_dim

        # -------------------------- Model --------------------------
        # News Encoder
        self.local_news_encoder = NewsEncoder(cfg, glove_emb)

        # GCN
        self.global_news_encoder = Sequential('x, index', [
            (GatedGraphConv(self.news_dim, num_layers=3, aggr='add'), 'x, index -> x'),
        ])
        # Entity
        if self.use_entity:
            pretrain = torch.from_numpy(entity_emb).float()
            self.entity_embedding_layer = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)

            self.local_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (EntityEncoder(cfg), 'x, mask -> x'),
            ])

            self.local_entity_embedding = Sequential('x, mask', [           # 候选实体转换为embedding
                (self.entity_embedding_layer, 'x -> x'),
            ])

            self.global_entity_encoder = Sequential('x, mask', [
                (self.entity_embedding_layer, 'x -> x'),
                (GlobalEntityEncoder(cfg), 'x, mask -> x'),
            ])
        # Click Encoder
        self.click_encoder = ClickEncoder(cfg)

        # User Encoder
        self.user_encoder = UserEncoder(cfg)

        # Candidate Encoder
        self.candidate_encoder = CandidateEncoder(cfg)

        # click prediction
        self.click_predictor = DotProduct()
        self.loss_fn = NCELoss()

        # 添加一个列表以在每次前向传递期间存储嵌入
        self.embeddings_list = []

    def forward(self, subgraph, mapping_idx, candidate_news, candidate_entity, entity_mask, label=None):
        # -------------------------------------- clicked ----------------------------------
        mask = mapping_idx != -1
        mapping_idx[mapping_idx == -1] = 0

        batch_size, num_clicked, token_dim = mapping_idx.shape[0], mapping_idx.shape[1], candidate_news.shape[-1]
        # clicked_entity = subgraph.x[mapping_idx, -8:-3]  # clicked_entity.shape = ([8,50,5])   8组，每组50个用户，每个用户5个实体编码
        clicked_entity = subgraph.x[: , -8:-3]     # 取出所有的实体，和新闻合并在一起

        # News Encoder + GCN
        x_flatten = subgraph.x.view(1, -1, token_dim)  # x_flatten.shape = [1,4737,38]     每个新闻用38个特征表示
        x_encoded = self.local_news_encoder(x_flatten).view(-1, self.news_dim)  # x_encoded = [4737,400]

        graph_emb = self.global_news_encoder(x_encoded, subgraph.edge_index)  # graph_emb = [4737,400]

        clicked_origin_emb = x_encoded[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                                self.news_dim)  # clicked_origin_emb.shape = [8,50,400]
        clicked_graph_emb = graph_emb[mapping_idx, :].masked_fill(~mask.unsqueeze(-1), 0).view(batch_size, num_clicked,
                                                                                               self.news_dim)

        # Attention pooling
        if self.use_entity:
            clicked_entity = self.local_entity_encoder(clicked_entity, None)
        else:
            clicked_entity = None

        clicked_total_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity)
        user_emb = self.user_encoder(clicked_total_emb, mask)

        # ----------------------------------------- Candidate------------------------------------

        if self.use_entity:
            origin_entity, neighbor_entity = candidate_entity.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_embedding(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

            # cand_entity_emb = self.entity_encoder(candidate_entity, entity_mask).view(batch_size, -1, self.news_dim) # [8, 5, 400]
        else:
            cand_origin_entity_emb, cand_neighbor_entity_emb = None, None

        cand_title_emb = self.local_news_encoder(candidate_news, cand_origin_entity_emb)  # [8, 5, 400]
        cand_final_emb = self.candidate_encoder(cand_title_emb, cand_origin_entity_emb, cand_neighbor_entity_emb)

        # embedding = torch.cat((clicked_total_emb, cand_final_emb), dim=1).detach().cpu().numpy()
        # self.embeddings_list.extend(embedding)
        # ----------------------------------------- Score ------------------------------------
        score = self.click_predictor(cand_final_emb, user_emb)
        loss = self.loss_fn(score, label)

        return loss, score

    def validation_process(self, subgraph, mappings, clicked_entity, candidate_emb, candidate_entity, entity_mask):

        batch_size, num_news, news_dim = 1, len(mappings), candidate_emb.shape[-1]

        title_graph_emb = self.global_news_encoder(subgraph.x, subgraph.edge_index)
        clicked_graph_emb = title_graph_emb[mappings, :].view(batch_size, num_news, news_dim)
        clicked_origin_emb = subgraph.x[mappings, :].view(batch_size, num_news, news_dim)

        # --------------------Attention Pooling
        if self.use_entity:
            clicked_entity_emb = self.local_entity_encoder(clicked_entity.unsqueeze(0), None)
        else:
            clicked_entity_emb = None

        clicked_final_emb = self.click_encoder(clicked_origin_emb, clicked_graph_emb, clicked_entity_emb)

        user_emb = self.user_encoder(clicked_final_emb)  # [1, 400]

        # ----------------------------------------- Candidate------------------------------------

        if self.use_entity:
            cand_entity_input = candidate_entity.unsqueeze(0)
            entity_mask = entity_mask.unsqueeze(0)
            origin_entity, neighbor_entity = cand_entity_input.split(
                [self.cfg.model.entity_size, self.cfg.model.entity_size * self.cfg.model.entity_neighbors], dim=-1)

            cand_origin_entity_emb = self.local_entity_encoder(origin_entity, None)
            cand_neighbor_entity_emb = self.global_entity_encoder(neighbor_entity, entity_mask)

        else:
            cand_origin_entity_emb = None
            cand_neighbor_entity_emb = None

        cand_final_emb = self.candidate_encoder(candidate_emb.unsqueeze(0), cand_origin_entity_emb,
                                                cand_neighbor_entity_emb)
        # ---------------------------------------------------------------------------------------
        # ----------------------------------------- Score ------------------------------------
        scores = self.click_predictor(cand_final_emb, user_emb).view(-1).cpu().tolist()

        return scores
