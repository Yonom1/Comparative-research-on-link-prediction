import networkx as nx
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Tuple, List, Set, Dict
import os

def load_edge_list(file_path: str) -> List[Tuple[int, int]]:
    """从文件加载边列表，并确保节点ID是连续的"""
    # 首先读取所有边
    edges = []
    node_set = set()
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过注释行
            if line.startswith('#'):
                continue
            # 解析边
            u, v = map(int, line.strip().split())
            edges.append((u, v))
            node_set.add(u)
            node_set.add(v)
    
    # 创建节点ID映射
    node_list = sorted(list(node_set))
    node_map = {old_id: new_id for new_id, old_id in enumerate(node_list)}
    
    # 重新映射边的节点ID
    mapped_edges = [(node_map[u], node_map[v]) for u, v in edges]
    
    print(f"节点ID重映射完成:")
    print(f"- 原始节点ID范围: [{min(node_set)}, {max(node_set)}]")
    print(f"- 映射后节点ID范围: [0, {len(node_set)-1}]")
    
    return mapped_edges

def build_graph(edges: List[Tuple[int, int]]) -> nx.Graph:
    """从边列表构建无向图"""
    G = nx.Graph()
    G.add_edges_from(edges)
    return G

def split_edges(G: nx.Graph, test_size: float, random_state: int) -> Tuple[nx.Graph, List[Tuple[str, str]]]:
    """划分训练集与测试集"""
    edges = list(G.edges())
    train_edges, test_edges = train_test_split(edges, test_size=test_size, random_state=random_state)
    
    # 构建训练图（保留所有节点）
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)
    
    return G_train, test_edges

def sample_non_edges(G: nx.Graph, sample_size: int, random_state: int) -> List[Tuple[int, int]]:
    """随机采样非边（不存在的边）

    Args:
        G: NetworkX图对象
        sample_size: 要采样的非边数量
        random_state: 随机种子

    Returns:
        采样得到的非边列表
    """
    np.random.seed(random_state)
    nodes = list(G.nodes())  # 获取实际的节点ID列表
    n_nodes = len(nodes)
    existing_edges = set(G.edges())

    # 使用集合来存储采样的非边，避免重复
    sampled_non_edges = set()

    # 采样直到达到目标数量或尝试次数上限
    max_attempts = sample_size * 10  # 设置最大尝试次数
    attempts = 0

    while len(sampled_non_edges) < sample_size and attempts < max_attempts:
        # 从实际的节点列表中随机选择两个节点
        i = np.random.randint(0, n_nodes)
        j = np.random.randint(0, n_nodes)
        u, v = nodes[i], nodes[j]  # 使用实际的节点ID

        # 确保u < v，保持一致性（如果是数字ID的话）
        if isinstance(u, (int, float)) and u > v:
            u, v = v, u

        # 如果两个节点相同或者边已存在或者已经被采样过，则跳过
        if u == v or (u, v) in existing_edges or (v, u) in existing_edges or (u, v) in sampled_non_edges:
            attempts += 1
            continue

        sampled_non_edges.add((u, v))

    # 如果达到尝试次数上限但样本不足，发出警告
    if len(sampled_non_edges) < sample_size:
        import warnings
        warnings.warn(f"只采样到 {len(sampled_non_edges)} 个非边，少于请求的 {sample_size} 个")

    return list(sampled_non_edges)

def build_adjacency_dict(G: nx.Graph) -> Dict[str, Set[str]]:
    """构建邻接表字典"""
    return {node: set(G.neighbors(node)) for node in G.nodes()}

