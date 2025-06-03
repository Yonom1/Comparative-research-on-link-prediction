import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, List, Dict, Set
import networkx as nx
from sklearn.model_selection import train_test_split

def create_pyg_data(adj_dict: Dict[str, Set[str]], 
                   node_features: torch.Tensor = None) -> Data:
    """将邻接字典转换为PyG的Data对象"""
    # 创建节点索引映射
    nodes = list(adj_dict.keys())
    node_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # 构建边索引
    edge_index = []
    for u in adj_dict:
        for v in adj_dict[u]:
            edge_index.append([node_idx[u], node_idx[v]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # 如果没有提供节点特征，创建默认特征
    if node_features is None:
        node_features = torch.eye(len(nodes))
    
    return Data(x=node_features, edge_index=edge_index)

def train_test_split_edges(data: Data, 
                          test_ratio: float = 0.1, 
                          val_ratio: float = 0.05) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
    """划分训练集、验证集和测试集的边"""
    # 获取无向图的边（确保每条边只出现一次）
    edge_index = data.edge_index.t().numpy()
    edges = set(map(tuple, edge_index))
    edges = [list(edge) for edge in edges]
    
    # 划分训练集、验证集和测试集
    train_edges, test_edges = train_test_split(edges, test_size=test_ratio, random_state=42)
    train_edges, val_edges = train_test_split(train_edges, 
                                            test_size=val_ratio/(1-test_ratio), 
                                            random_state=42)
    
    # 转换为tensor
    train_edge_index = torch.tensor(train_edges, dtype=torch.long).t()
    val_edge_index = torch.tensor(val_edges, dtype=torch.long).t()
    test_edge_index = torch.tensor(test_edges, dtype=torch.long).t()
    
    # 更新data对象
    data.edge_index = train_edge_index
    
    return data, train_edge_index, val_edge_index, test_edge_index

def sample_negative_edges(edge_index: torch.Tensor, 
                        num_nodes: int,
                        num_samples: int) -> torch.Tensor:
    """采样负边（不存在的边）"""
    # 创建现有边的集合
    existing_edges = set(map(tuple, edge_index.cpu().t().numpy()))

    # 采样负边
    neg_edges = []
    while len(neg_edges) < num_samples:
        i = np.random.randint(0, num_nodes)
        j = np.random.randint(0, num_nodes)
        if i != j and (i, j) not in existing_edges and (j, i) not in existing_edges:
            neg_edges.append([i, j])
    
    return torch.tensor(neg_edges, dtype=torch.long).t()

def create_node_features(G: nx.Graph, feature_type: str = 'degree') -> torch.Tensor:
    """创建节点特征
    
    Args:
        G: NetworkX图对象
        feature_type: 特征类型，可选：
            - 'degree': 使用节点度作为特征
            - 'onehot': 使用独热编码
            - 'spectral': 使用谱嵌入
    """
    num_nodes = G.number_of_nodes()

    if feature_type == 'degree':
        # 使用节点度作为特征
        degrees = torch.tensor([[G.degree(node)] for node in range(num_nodes)], 
                             dtype=torch.float)
        # 归一化
        degrees = degrees / degrees.max()
        return degrees
    
    elif feature_type == 'onehot':
        # 使用独热编码
        return torch.eye(num_nodes)

    elif feature_type == 'spectral':
        # 使用谱嵌入（取前16个特征）
        try:
            from sklearn.manifold import SpectralEmbedding
            embedder = SpectralEmbedding(n_components=16, random_state=42)
            features = embedder.fit_transform(nx.adjacency_matrix(G).todense())
            return torch.tensor(features, dtype=torch.float)
        except:
            print("警告：谱嵌入计算失败，使用节点度作为特征")
            return create_node_features(G, 'degree')

    else:
        raise ValueError(f"未知的特征类型：{feature_type}")

def prepare_data_for_gnn(G: nx.Graph,
                        test_ratio: float = 0.1,
                        val_ratio: float = 0.05,
                        feature_type: str = 'spectral') -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor]:
    """准备GNN模型的训练数据

    Args:
        G: NetworkX图对象
        test_ratio: 测试集比例
        val_ratio: 验证集比例
        feature_type: 特征类型，建议使用'spectral'以匹配GNN模型的输入维度
    """
    # 创建节点特征
    print(f"[数据处理] 使用{feature_type}特征...")
    node_features = create_node_features(G, feature_type)
    print(f"[数据处理] 特征维度: {node_features.shape}")

    # 创建PyG的Data对象
    edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t()
    data = Data(x=node_features, edge_index=edge_index)
    
    # 划分数据集
    data, train_edges, val_edges, test_edges = train_test_split_edges(
        data, test_ratio, val_ratio)
    
    return data, train_edges, val_edges, test_edges

def split_edges(G: nx.Graph, test_size: float = 0.1, random_state: int = 42) -> Tuple[nx.Graph, List[Tuple[int, int]]]:
    """为传统方法划分训练集和测试集

    Args:
        G: 原始图
        test_size: 测试集比例
        random_state: 随机种子

    Returns:
        G_train: 训练图（移除测试集边的图）
        test_edges: 测试集边列表
    """
    edges = list(G.edges())
    nodes = list(G.nodes())

    # 划分边
    train_edges, test_edges = train_test_split(
        edges, test_size=test_size, random_state=random_state
    )

    # 创建训练图
    G_train = nx.Graph()
    G_train.add_nodes_from(nodes)
    G_train.add_edges_from(train_edges)

    return G_train, test_edges

def sample_non_edges(G: nx.Graph, num_samples: int, random_state: int = 42) -> List[Tuple[int, int]]:
    """采样不存在的边作为负样本

    Args:
        G: 图
        num_samples: 需要采样的边数量
        random_state: 随机种子

    Returns:
        non_edges: 采样的不存在的边列表
    """
    np.random.seed(random_state)

    # 获取所有可能的不存在的边
    non_edges = list(nx.non_edges(G))

    # 如果不存在的边数量少于需要采样的数量，返回所有不存在的边
    if len(non_edges) <= num_samples:
        return non_edges

    # 随机采样指定数量的不存在的边
    sampled_indices = np.random.choice(len(non_edges), num_samples, replace=False)
    return [non_edges[i] for i in sampled_indices]


