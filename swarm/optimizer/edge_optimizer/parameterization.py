#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Tuple
import random

from swarm.graph.node import Node
from swarm.graph.graph import Graph
from swarm.graph.composite_graph import CompositeGraph


class ConnectDistribution(nn.Module):
    def __init__(self, potential_connections):
        super().__init__()
        self.potential_connections = potential_connections

    def realize(self, graph):
        raise NotImplemented


class MRFDist(ConnectDistribution):
    pass


class EdgeWiseDistribution(ConnectDistribution):
    def __init__(self,
                 potential_connections,
                 initial_probability: float = 0.5,
                 ):
        super().__init__(potential_connections)
        init_logit = torch.log(torch.tensor(initial_probability / (1 - initial_probability)))
        init_tensor = torch.ones(
            len(potential_connections),
            requires_grad=True) * init_logit
        self.edge_logits = torch.nn.Parameter(init_tensor)
        node_ids = set([x for pair in potential_connections for x in pair])
        self.node_idx2id = {i: node_id for i, node_id in enumerate(node_ids)}
        self.node_id2idx = {node_id: i for i, node_id in enumerate(node_ids)}
        order_tensor = torch.randn(len(node_ids))
        self.order_params = torch.nn.Parameter(order_tensor)    # 学习节点的拓扑排序偏好

    def random_sample_num_edges(self, graph: CompositeGraph, num_edges: int) -> CompositeGraph:
        '''
        用于快速生成满足边数约束的随机图
        '''
        _graph = deepcopy(graph)
        while True:
            if _graph.num_edges >= num_edges:
                break
            potential_connection = random.sample(self.potential_connections, 1)[0]
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_ranks(self, graph, use_max: bool = False):
        '''
        拓扑排序实现:
        1. 通过贪心算法生成拓扑排序
        2. 使用 order_params 决定节点选择偏好
        3. 支持确定性（argmax）和随机采样两种模式
        '''
        log_probs = []
        ranks = {}
        in_degrees = {node.id: len(node.predecessors) for node in graph.nodes.values()}
        for i in range(len(self.order_params)):
            avaliable_nodes = [node for node in graph.nodes if in_degrees[node] == 0]
            logits = []
            for node in avaliable_nodes:
                logits.append(self.order_params[self.node_id2idx[node]])
            logits = torch.stack(logits).reshape(-1)
            if use_max:
                idx = torch.argmax(logits)
            else:
                idx = torch.distributions.Categorical(logits=logits).sample()
            log_probs.append(torch.log_softmax(logits, dim=0)[idx])

            ranks[avaliable_nodes[idx]] = i
            in_degrees[avaliable_nodes[idx]] = -1
            for successor in graph.nodes[avaliable_nodes[idx]].successors:
                in_degrees[successor.id] -= 1
        return ranks, torch.sum(torch.stack(log_probs))

    def realize(self,
                graph: CompositeGraph,
                temperature: float = 1.0, # must be >= 1.0
                threshold: float = None,
                use_learned_order: bool = False,    # 是否使用学习到的节点拓扑顺序
                ) -> Tuple[CompositeGraph, torch.Tensor]:
        if use_learned_order:    # 如果启用则生成基于学习参数的节点拓扑排序
            ranks, log_prob = self.realize_ranks(graph, threshold is not None)
            log_probs = [log_prob]
        else:    # 否则初始化一个零值的可微分log概率
            log_probs = [torch.tensor(0.0, requires_grad=True)]
        _graph = deepcopy(graph)
        for potential_connection, edge_logit in zip(
                self.potential_connections, self.edge_logits):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue
            
            addable_if_use_learned_order = use_learned_order and (ranks[out_node.id] < ranks[in_node.id])    # 源节点排序必须早于目标节点（保证非循环）
            addable_if_not_used_learned_order = (not use_learned_order) and (not _graph.check_cycle(in_node, {out_node}, set()))    # 如果未使用学习到的节点拓扑顺序，则检查是否存在循环
            if addable_if_not_used_learned_order or addable_if_use_learned_order:
                edge_prob = torch.sigmoid(edge_logit / temperature)
                if threshold:
                    edge_prob = torch.tensor(1 if edge_prob > threshold else 0)
                if torch.rand(1) < edge_prob:    # 按随机概率决定潜在边是否存在，有向边
                    out_node.add_successor(in_node)
                    # in_node.add_predecessor(out_node)
                    log_probs.append(torch.log(edge_prob))
                else:
                    log_probs.append(torch.log(1 - edge_prob))

        log_prob = torch.sum(torch.stack(log_probs))    # 所有边采样结果的联合概率的对数表示，可用于梯度反向传播。
        return _graph, log_prob

    def realize_full(self, graph: CompositeGraph) -> CompositeGraph:
        '''
        生成最大可能连接图（不形成环的情况下）
        '''
        _graph = deepcopy(graph)
        for i, potential_connection in enumerate(self.potential_connections):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                out_node.add_successor(in_node)
                in_node.add_predecessor(out_node)
        return _graph

    def realize_mask(self, graph: CompositeGraph, edge_mask: torch.Tensor) -> CompositeGraph:
        '''
        根据给定的边掩码生成图
        '''
        _graph = deepcopy(graph)
        for i, (potential_connection, is_edge) in enumerate(zip(self.potential_connections, edge_mask)):
            out_node = _graph.find_node(potential_connection[0])
            in_node = _graph.find_node(potential_connection[1])

            if not out_node or not in_node:
                continue

            if not _graph.check_cycle(in_node, {out_node}, set()):
                if is_edge:
                    out_node.add_successor(in_node)
                    in_node.add_predecessor(out_node)
        return _graph
