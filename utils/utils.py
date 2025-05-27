#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import dgl
import scipy.sparse as sp 

def show(metrics_log):
    x = range(0, len(list(metrics_log.values())[0]))
    i = 1
    columns = 2
    rows = ceil(len(metrics_log)/columns)
    for k, v in metrics_log.items():
        plt.subplot(rows, columns, i)
        plt.plot(x, v, '.-')
        plt.title('{} vs epochs'.format(k))
        i += 1
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()


def get_perf(metrics_log, window_size, target, show=True):
    # max
    maxs = {title: 0 for title in metrics_log.keys()}
    assert target in maxs
    length = len(metrics_log[target])
    for v in metrics_log.values():
        assert length == len(v)
    if window_size >= length:
        for k, v in metrics_log.items():
            maxs[k] = np.mean(v)
    else:
        for i in range(length-window_size):
            now = np.mean(metrics_log[target][i:i+window_size])
            if now > maxs[target]:
                for k, v in metrics_log.items():
                    maxs[k] = np.mean(v[i:i+window_size])
    if show:
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
    return maxs


def check_overfitting(metrics_log, target, threshold=0.02, show=False):
    maxs = get_perf(metrics_log, 1, target, False)
    assert target in maxs
    overfit = (maxs[target]-metrics_log[target][-1]) > threshold
    if overfit and show:
        print('***********overfit*************')
        print('best:', end=' ')
        for k, v in maxs.items():
            print('{}:{:.5f}'.format(k, v), end=' ')
        print('')
        print('now:', end=' ')
        for k, v in metrics_log.items():
            print('{}:{:.5f}'.format(k, v[-1]), end=' ')
        print('')
        print('***********overfit*************')
    return overfit

def create_homogeneous_graph(graph, num_nodes_a, num_nodes_b, device):
    if graph.shape == (num_nodes_a, num_nodes_b):
        # 添加自环
        graph_with_loops = sp.bmat([[sp.identity(num_nodes_a), graph],[graph.T, sp.identity(num_nodes_b)]])  
        return dgl.from_scipy(graph_with_loops).to(device)
    else:
        raise ValueError("raw_graph's shape is wrong")

def early_stop(metric_log, early, threshold=0.01):
    #print("early_stop, ", metric_log, ",", early, ",", threshold)
    if len(metric_log) >= 2 and metric_log[-1] < metric_log[-2] and metric_log[-1] > threshold:
        return early-1
    else:
        return early
    
def get_node_degrees(graph,edge_type='repeat'):
        """
        计算每个节点的度数
        :param graph: DGL 图对象
        :return: 包含每个节点度数的张量
        """
        # 计算节点的入度和出度

        in_degrees = graph.in_degrees()
        out_degrees = graph.out_degrees()
        
        # 计算总度数
        if edge_type == 'repeat':
            degrees = out_degrees
        else:
            degrees = out_degrees  + in_degrees
        
        # 进行对数归一化
        degree_log = torch.log1p(degrees.float()) / torch.log1p(degrees.max().float())
        
        # 将结果作为图的属性保存
        graph.ndata['degree'] = degrees.float()
        graph.ndata['degree_log'] = degree_log
        
        return graph

def get_degree_dict( graph: dgl.DGLGraph, edge_type='repeat'):

    degrees = graph.ndata['degree']

    # 确保每个度数都是整数
    degree_dict = {node: int(degree.item()) for node, degree in enumerate(degrees)}

    return degree_dict

def get_node_neighs(graph: dgl.DGLGraph, edge_type='repeat'):

    neighs_dict = {}

    if edge_type == 'repeat':
        for node in range(graph.num_nodes()):
            neighbors = graph.successors(node).tolist()
            if neighbors:
                neighs_dict[node] = neighbors
    else:
        for node in range(graph.num_nodes()):
            succ = graph.successors(node).tolist()
            pred = graph.predecessors(node).tolist()
            neighbors = list(set(succ + pred))
            if neighbors:
                neighs_dict[node] = neighbors

    return neighs_dict

def get_neighs_degree_matrix(graph: dgl.DGLGraph, device, edge_type='repeat'):

    degree_dict = get_degree_dict(graph, edge_type=edge_type)
    neighs_dict = get_node_neighs(graph, edge_type=edge_type)

    if not degree_dict:
        raise ValueError("度数字典为空，请检查图是否为空。")
    
    degree_max = max(degree_dict.values())
    degree_max = int(degree_max)  # 转换为整数
    num_nodes = graph.num_nodes()
    degree_exist_matrix = np.zeros((num_nodes, degree_max), dtype=np.float32)
    
    for node, neighbors in neighs_dict.items():
        for neighbor in neighbors:
            neighbor_degree = degree_dict.get(neighbor, 0)
            if neighbor_degree > 0:
                degree_exist_matrix[node][neighbor_degree - 1] += 1
    
    # 对 degree_exist_matrix 进行归一化
    x_min = degree_exist_matrix.min(axis=0)
    x_max = degree_exist_matrix.max(axis=0)
    denom = x_max - x_min
    denom[denom == 0] = 1  # 避免除以零
    x = (degree_exist_matrix - x_min) / denom
    x = np.nan_to_num(x)
    
    degree_exist_matrix_tensor = torch.tensor(x, dtype=torch.float32).to(device)
    
    graph.ndata['degree_exist_matrix'] = degree_exist_matrix_tensor

    return graph

def get_degree_sum_neigh(graph: dgl.DGLGraph, device, edge_type='repeat'):

    """
    获取每个节点的邻居节点的度数之和，并将其作为新的特征添加到图数据中
    :param graph: DGL 图对象
    :param edge_type: 边的类型 ('repeat' 或其他)
    :return: 更新后的 DGL 图对象，包含邻居度数之和的属性
    """
    # 获取节点度数字典
    degree_dict = get_degree_dict(graph, edge_type=edge_type)
    
    # 获取邻居节点字典
    neighs_dict = get_node_neighs(graph, edge_type=edge_type)
    
    # 初始化邻居度数之和列表，预设为0
    degree_sum_list = [0] * graph.num_nodes()
    
    # 计算每个节点的邻居度数之和
    for node, neighbors in neighs_dict.items():
        degree_sum = sum(degree_dict[neighbor] for neighbor in neighbors)
        degree_sum_list[node] = degree_sum
    
    # 转换为张量
    degree_sum = torch.tensor(degree_sum_list, dtype=torch.float32).to(device)
    
    # 添加到图的节点数据中
    graph.ndata['degree_neighs_sum'] = degree_sum
    
    # 记录最大值
    graph.degree_sum_max = degree_sum.max()
    
    # 对度数之和进行对数归一化
    degree_sum_log = torch.log(degree_sum + 1) / torch.log(torch.tensor(degree_sum.max()).float())
    degree_sum_log = torch.nan_to_num(degree_sum_log)  # 将 NaN 替换为 0
    graph.ndata['degree_neighs_sum_log'] = degree_sum_log
    
    # 计算邻居度数之和的不同类别数
    unique_classes = len(torch.unique(degree_sum))
    graph.degree_neighs_sum_classes = unique_classes
    
    return graph

def structure(ub_graph, ui_graph, bi_graph, device):

    results = []  # 存储所有图的度数和邻居度数之和信息

    # =================== 用户-捆绑包图 (ub_graph) ===================
    # 1. 计算节点自身度数的预测损失
    ub_graph = get_node_degrees(ub_graph)
    ub_degree = ub_graph.ndata['degree'] # 获得节点的度
   
    # 2. 计算邻居度数之和的预测损失
    ub_graph = get_neighs_degree_matrix(ub_graph, device) # 获取邻居节点的度矩阵
    ub_graph = get_degree_sum_neigh(ub_graph, device)  # 计算邻居度数之和
    ub_neigh_sum = ub_graph.ndata['degree_neighs_sum']  # 邻居度数之和

    # 将用户-捆绑包图的信息存储为字典
    results.append({
        "graph": "ub_graph",
        "node_degree": ub_degree,  # 节点度数
        "neigh_degree_sum": ub_neigh_sum  # 邻居度数之和
    })
    
    # =================== 用户-物品图 (ui_graph) ===================
    ui_graph = get_node_degrees(ui_graph)
    ui_degree = ui_graph.ndata['degree']
    
    ui_graph = get_neighs_degree_matrix(ui_graph, device)
    ui_graph = get_degree_sum_neigh(ui_graph, device)
    ui_neigh_sum = ui_graph.ndata['degree_neighs_sum']

    # 将用户-物品图的信息存储为字典
    results.append({
        "graph": "ui_graph",
        "node_degree": ui_degree,  # 节点度数
        "neigh_degree_sum": ui_neigh_sum  # 邻居度数之和
    })
    
    # =================== 捆绑包-物品图 (bi_graph) ===================
    bi_graph = get_node_degrees(bi_graph)
    bi_degree = bi_graph.ndata['degree']
    

    bi_graph = get_neighs_degree_matrix(bi_graph, device)
    bi_graph = get_degree_sum_neigh(bi_graph, device)
    bi_neigh_sum = bi_graph.ndata['degree_neighs_sum']

    # 将捆绑包-物品图的信息存储为字典
    results.append({
        "graph": "bi_graph",
        "node_degree": bi_degree,  # 节点度数
        "neigh_degree_sum": bi_neigh_sum  # 邻居度数之和
    })

    return results