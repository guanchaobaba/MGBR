#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
import numpy as np
from .model_base import Info, Model
import dgl
from dgl.nn import GraphConv
from config import CONFIG
from dgl.nn import GINConv
from dgl.nn import GATv2Conv
from dgl.nn import GATConv
from dgl.nn import TWIRLSConv

'''
class GIN_MLP(nn.Module):

    def __init__(self, in_feats, out_feats):
        super(GIN_MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, out_feats)
        self.linear2 = nn.Linear(out_feats, out_feats)

    def forward(self, x):
        x = F.relu(self.linear1(x))

        return self.linear2(x)
'''

class MGBR_Info(Info):

    def __init__(self, embedding_size, embed_L2_norm, mess_dropout, node_dropout, num_layers, act=nn.LeakyReLU()):
        super().__init__(embedding_size, embed_L2_norm)
        self.act = act
        assert 1 > mess_dropout >= 0
        self.mess_dropout = mess_dropout
        assert 1 > node_dropout >= 0
        self.node_dropout = node_dropout
        assert isinstance(num_layers, int) and num_layers > 0
        self.num_layers = num_layers

class MGBR(Model):

    def get_infotype(self):
        return MGBR_Info

    def __init__(self, info, dataset, raw_graph, device, gprompt, pretrain=None):
        super().__init__(info, dataset, create_embeddings=True)

        # 动态检测设备
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.items_feature = nn.Parameter(
            torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)
        self.epison = 1e-8
        self.gprompt = gprompt
        self.c_temp = 0.2

        assert isinstance(raw_graph, list)

        ub_graph, ui_graph, bi_graph = raw_graph
        
        # 处理图
        self.ui_hom_graph = self._create_homogeneous_graph(ui_graph, self.num_users, self.num_items)
        self.ub_hom_graph = self._create_homogeneous_graph(ub_graph, self.num_users, self.num_bundles)
        self.bi_hom_graph = self._create_homogeneous_graph(bi_graph, self.num_bundles, self.num_items)

        # copy from info
        self.act = self.info.act
        self.num_layers = self.info.num_layers
        self.device = device

        #  Dropouts
        self.mess_dropout = nn.Dropout(self.info.mess_dropout, True)
        self.node_dropout = nn.Dropout(self.info.node_dropout, True)

        # Layers
        # GAT
        self.gatconv1 = GATv2Conv(self.embedding_size, self.embedding_size, 3).to(self.device)
        self.gatconv2 = GATv2Conv(2 * self.embedding_size, self.embedding_size, 3).to(self.device)

        # GCN
        # self.gatconv1 = GraphConv(self.embedding_size, self.embedding_size).to(self.device)
        # self.gatconv2 = GraphConv(2*self.embedding_size, self.embedding_size).to(self.device) 

        # GINConv
        # self.gatconv1 = GINConv(GIN_MLP(self.embedding_size, self.embedding_size), learn_eps=True)
        # self.gatconv2 = GINConv(GIN_MLP(2 * self.embedding_size, self.embedding_size), learn_eps=True)

        # TWIRLSConv
        # self.gatconv1 = TWIRLSConv(self.embedding_size, self.embedding_size, 64, prop_step = 2)
        # self.gatconv2 = TWIRLSConv(2 * self.embedding_size, self.embedding_size, 64 ,prop_step = 2)

        #预训练参数
        if pretrain is not None:
            self._initialize_pretrained_features(pretrain)

    #将图的构建逻辑提取到 _create_homogeneous_graph 方法中，避免代码重复。
    def _create_homogeneous_graph(self, graph, num_nodes_a, num_nodes_b):
        if graph.shape == (num_nodes_a, num_nodes_b):
            # 添加自环
            graph_with_loops = sp.bmat([[sp.identity(num_nodes_a), graph],[graph.T, sp.identity(num_nodes_b)]])  
            return dgl.from_scipy(graph_with_loops).to(self.device)
        else:
            raise ValueError("raw_graph's shape is wrong")
        
    #预训练参数的初始化提取到 _initialize_pretrained_features 方法。    
    def _initialize_pretrained_features(self, pretrain):
        self.users_feature.data = F.normalize(pretrain['users_feature'])
        self.items_feature.data = F.normalize(pretrain['items_feature'])
        self.bundles_feature.data = F.normalize(pretrain['bundles_feature'])
    
    def one_propagate(self, graph, A_feature, B_feature):

        # propagate
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]
                     # 拼接输出
        features = self.mess_dropout(torch.cat([torch.mean(self.gatconv1(graph, features), dim=1), features], 1))
        
        # gat_output = self.gatconv1(graph, features)  # 保持原始 GAT 输出
        # features = self.mess_dropout(torch.cat([gat_output, features], dim=1))

        all_features.append(F.normalize(features))

        features = self.mess_dropout(torch.cat([torch.mean(self.gatconv2(graph, features), dim=1), features], 1))
        
        # gat_output = self.gatconv2(graph, features)  # 保持原始 GAT 输出
        # features = self.mess_dropout(torch.cat([gat_output, features], dim=1))

        all_features = torch.cat(all_features, 1)
        A_feature, B_feature = torch.split(
            all_features, (A_feature.shape[0], B_feature.shape[0]), 0)
        
        return A_feature, B_feature
    

    def propagate(self):
        #  =============================  u-i level propagation  =============================
        gat1_users_feature, gat1_items_feature = self.one_propagate(
            self.ui_hom_graph, self.users_feature, self.items_feature)

        #  ============================= b-i level propagation =============================
        gat1_bundles_feature, gat2_items_feature = self.one_propagate(
            self.bi_hom_graph, self.bundles_feature, self.items_feature)

        #  =============================  u-b level propagation  =============================
        gat2_users_feature, gat2_bundles_feature = self.one_propagate(
            self.ub_hom_graph, self.users_feature, self.bundles_feature)

        users_feature = [gat1_users_feature, gat2_users_feature]
        bundles_feature = [gat1_bundles_feature, gat2_bundles_feature]
        items_feature = [gat1_items_feature, gat2_items_feature]

        return users_feature, bundles_feature, items_feature
    

    def predict(self, users_feature, bundles_feature, items_feature):

        users_feature_atom, users_feature_non_atom = users_feature # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # batch_n_f
        items_feature_atom, items_feature_non_atom = items_feature
        pred1 = torch.sum(users_feature_atom * bundles_feature_atom, 2) \
            + torch.sum(users_feature_non_atom * bundles_feature_non_atom, 2)
        pred2 = torch.sum(users_feature_atom * items_feature_atom, 2) \
            + torch.sum(users_feature_non_atom * items_feature_non_atom, 2)
        
        return pred1, pred2
    
    # def cal_c_loss(self, features1, features2):
    #     """
    #     计算 InfoNCE 对比损失
    #     :param features1: 特征视图 1 (batch_size, embedding_dim)
    #     :param features2: 特征视图 2 (batch_size, embedding_dim)
    #     :return: 对比损失 (scalar)
    #     """
    #     # 特征归一化 (L2 norm)
    #     features1 = F.normalize(features1, p=2, dim=1)
    #     features2 = F.normalize(features2, p=2, dim=1)

    #     # 计算正样本相似性分数 (点积)
    #     pos_score = torch.sum(features1 * features2, dim=1)  # [batch_size]

    #     # 计算全样本相似性分数
    #     ttl_score = torch.matmul(features1, features2.T)  # [batch_size, batch_size]

    #     # 使用温度参数 softmax 归一化
    #     pos_score = torch.exp(pos_score / self.c_temp)  # [batch_size]
    #     ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), dim=1)  # [batch_size]

    #     # 对比损失公式 (InfoNCE)
    #     c_loss = -torch.mean(torch.log(pos_score / ttl_score))  # scalar

    #     return c_loss
    
    def forward(self, users, bundles, items):

        users_feature, bundles_feature, items_feature = self.propagate()

        users_embedding = [i[users].expand(-1, bundles.shape[1], -1) for i in users_feature]  # u_f --> batch_f --> batch_n_f
        bundles_embedding = [i[bundles] for i in bundles_feature] # b_f --> batch_n_f
        items_embedding = [i[items] for i in items_feature]

        pred1, pred2 = self.predict(users_embedding, bundles_embedding, items_embedding)
        loss1, loss2 = self.regularize(users_embedding, bundles_embedding, items_embedding)
        loss3, loss4, loss5, loss6, loss7, loss8 = self.structureloss(self.ub_hom_graph,self.ui_hom_graph,self.bi_hom_graph)

        # # 对用户特征的不同视图计算对比损失
        # u_contrastive_loss = self.cal_c_loss(users_feature[0], users_feature[1])
        # # 对捆绑特征的不同视图计算对比损失
        # b_contrastive_loss = self.cal_c_loss(bundles_feature[0], bundles_feature[1])
        # # 对物品特征的不同视图计算对比损失
        # i_contrastive_loss = self.cal_c_loss(items_feature[0], items_feature[1])

        # # 对比损失总和
        # c_loss = (u_contrastive_loss + b_contrastive_loss + i_contrastive_loss) / 3

        return pred1, pred2, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8
    
    def structureloss(self,ub_graph, ui_graph, bi_graph):

        # 图卷积和线性层
        degreeslinear = nn.Linear(self.embedding_size,1).to(self.device)
        conv = GraphConv(self.embedding_size, self.embedding_size).to(self.device)

        # =================== 用户-捆绑包图 (ub_graph) ===================
        
        # 1. 计算节点自身度数的预测损失
        ub_result = next(result for result in self.gprompt if result['graph'] == 'ub_graph')

        x = torch.cat((self.users_feature , self.bundles_feature),dim=0) # 合并用户和捆绑包特征
        mse_x = conv(ub_graph,x) # 图卷积
        mse_degrees = degreeslinear(mse_x) # 节点自身度预测
        ub_degree_lable = ub_result['node_degree'].view(-1, 1)
        ub_degree_lable = ub_degree_lable.to(torch.float) # 转换为二维张量
        mse_loss1 = F.mse_loss(mse_degrees, ub_degree_lable)

        # 2. 计算邻居度数之和的预测损失

        ub_neigh_sum = ub_result['neigh_degree_sum']  # 邻居度数之和
        mse_neigh_degrees = degreeslinear(mse_x)  # 预测邻居度数的和
        ub_neigh_label = ub_neigh_sum.view(-1, 1).to(torch.float)  # 转换为二维张量
        mse_loss1_neigh = F.mse_loss(mse_neigh_degrees, ub_neigh_label)  # 邻居度数之和的 MSE 损失

        # =================== 用户-物品图 (ui_graph) ===================

        ui_result = next(result for result in self.gprompt if result['graph'] == 'ui_graph')

        x = torch.cat((self.users_feature , self.items_feature),dim=0)
        mse_x = conv(ui_graph,x)
        mse_degrees = degreeslinear(mse_x)
        ui_degree_lable = ui_result['node_degree'].view(-1, 1)
        ui_degree_lable = ui_degree_lable.to(torch.float)
        mse_loss2 = F.mse_loss(mse_degrees, ui_degree_lable)

        ui_neigh_sum = ui_result['neigh_degree_sum']
        mse_neigh_degrees = degreeslinear(mse_x)
        ui_neigh_label = ui_neigh_sum.view(-1, 1).to(torch.float)
        mse_loss2_neigh = F.mse_loss(mse_neigh_degrees, ui_neigh_label)

        # =================== 捆绑包-物品图 (bi_graph) ===================

        bi_result = next(result for result in self.gprompt if result['graph'] == 'bi_graph')

        x = torch.cat((self.bundles_feature , self.items_feature),dim=0)
        mse_x = conv(bi_graph,x)
        mse_degrees = degreeslinear(mse_x)

        bi_degree_lable = bi_result['node_degree'].view(-1, 1)
        bi_degree_lable = bi_degree_lable.to(torch.float)
        mse_loss3 = F.mse_loss(mse_degrees, bi_degree_lable)

        bi_neigh_sum = bi_result['neigh_degree_sum']
        mse_neigh_degrees = degreeslinear(mse_x)
        bi_neigh_label = bi_neigh_sum.view(-1, 1).to(torch.float)
        mse_loss3_neigh = F.mse_loss(mse_neigh_degrees, bi_neigh_label)

        return mse_loss1, mse_loss2, mse_loss3, mse_loss1_neigh, mse_loss2_neigh, mse_loss3_neigh

    def regularize(self, users_feature, bundles_feature, items_feature):  #items_feature

        users_feature_atom, users_feature_non_atom = users_feature # batch_n_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # batch_n_f
        items_feature_atom, items_feature_non_atom = items_feature

        loss1 = self.embed_L2_norm * \
            ((users_feature_atom ** 2).sum() + (bundles_feature_atom ** 2).sum() +
            (users_feature_non_atom ** 2).sum() + (bundles_feature_non_atom ** 2).sum())
        loss2 = self.embed_L2_norm * \
            ((users_feature_atom ** 2).sum() + (items_feature_atom ** 2).sum() +
            (users_feature_non_atom ** 2).sum() + (items_feature_non_atom ** 2).sum())
        
        return loss1, loss2

    def evaluate(self, propagate_result, users):
        '''
        just for testing, compute scores of all bundles for `users` by `propagate_result`
        '''
        users_feature, bundles_feature, items_feature = propagate_result
        users_feature_atom, users_feature_non_atom = [i[users] for i in users_feature] # batch_f
        bundles_feature_atom, bundles_feature_non_atom = bundles_feature # b_f
        scores = torch.mm(users_feature_atom, bundles_feature_atom.t()) \
            + torch.mm(users_feature_non_atom, bundles_feature_non_atom.t()) # batch_b
        
        return scores