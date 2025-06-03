import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

def temporal_split(
    edges_df: pd.DataFrame,
    split_year: int = 2002,
    test_years: Tuple[int, int] = (2003, 2004)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    按时间切分数据集为训练集和测试集
    
    Args:
        edges_df: 包含边信息的DataFrame，至少包含 source, target, year 列
        split_year: 训练集截止年份
        test_years: 测试集年份范围元组 (start_year, end_year)
    
    Returns:
        train_edges: 训练集边列表 shape=(N, 2)
        test_pos_edges: 测试集正样本边列表
        test_neg_edges: 测试集负样本边列表
    """
    # 训练集：截止到split_year的所有边
    train_mask = edges_df['year'] <= split_year
    train_edges = edges_df[train_mask][['source', 'target']].values
    
    # 测试集正样本：test_years范围内的新增边
    test_mask = (edges_df['year'] >= test_years[0]) & (edges_df['year'] <= test_years[1])
    test_pos_edges = edges_df[test_mask][['source', 'target']].values
    
    # 生成负样本
    all_nodes = np.unique(edges_df[['source', 'target']].values)
    train_edges_set = set(map(tuple, train_edges))
    test_edges_set = set(map(tuple, test_pos_edges))
    
    # 随机采样负样本，确保不在训练集和测试集中
    n_neg_samples = len(test_pos_edges)
    neg_edges = []
    while len(neg_edges) < n_neg_samples:
        source = np.random.choice(all_nodes)
        target = np.random.choice(all_nodes)
        if source != target and (source, target) not in train_edges_set and (source, target) not in test_edges_set:
            neg_edges.append((source, target))
    
    test_neg_edges = np.array(neg_edges)
    
    return train_edges, test_pos_edges, test_neg_edges

def get_directed_neighbors(edges: np.ndarray, node: int) -> Tuple[set, set]:
    """
    获取节点的入边和出边邻居集合
    
    Args:
        edges: 边列表 shape=(N, 2)
        node: 目标节点ID
    
    Returns:
        in_neighbors: 入边邻居集合
        out_neighbors: 出边邻居集合
    """
    in_neighbors = set(edges[edges[:, 1] == node, 0])
    out_neighbors = set(edges[edges[:, 0] == node, 1])
    return in_neighbors, out_neighbors 