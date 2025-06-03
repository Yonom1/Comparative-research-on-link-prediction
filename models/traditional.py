from typing import List, Tuple, Dict, Set
import networkx as nx
from abc import ABC, abstractmethod
import math
import numpy as np
from utils.temporal_split import get_directed_neighbors

class LinkPredictor(ABC):
    """链接预测器基类"""
    
    @abstractmethod
    def predict(self, node_pairs: List[Tuple[str, str]]) -> List[float]:
        """预测给定节点对之间的链接分数"""
        pass

    def predict_single(self, u: str, v: str) -> float:
        """预测单个节点对之间的链接分数"""
        return self.predict([(u, v)])[0]

class CommonNeighbors:
    """基于共同邻居的链接预测"""
    def predict(self, G: nx.Graph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        return np.array([len(list(nx.common_neighbors(G, u, v))) for u, v in node_pairs])

class JaccardCoefficient:
    """基于Jaccard系数的链接预测"""
    def predict(self, G: nx.Graph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        return np.array([preds[2] for preds in nx.jaccard_coefficient(G, node_pairs)])

class AdamicAdar:
    """基于Adamic-Adar指数的链接预测"""
    def predict(self, G: nx.Graph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        def safe_log(x):
            """安全的对数计算，处理x≤1的情况"""
            return math.log(max(x, 1.01))  # 确保分母不为0

        scores = []
        for u, v in node_pairs:
            score = sum(1.0 / safe_log(G.degree(w)) for w in nx.common_neighbors(G, u, v))
            scores.append(score)
        return np.array(scores)

class ResourceAllocation:
    """基于资源分配指数的链接预测"""
    def predict(self, G: nx.Graph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        return np.array([preds[2] for preds in nx.resource_allocation_index(G, node_pairs)])

class PreferentialAttachment:
    """基于优先连接的链接预测"""
    def predict(self, G: nx.Graph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        return np.array([preds[2] for preds in nx.preferential_attachment(G, node_pairs)])

class DirectedLinkPredictor(ABC):
    """有向图链接预测器基类"""
    
    @abstractmethod
    def predict(self, G: nx.DiGraph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """预测给定节点对之间的链接分数"""
        pass

class DirectedCommonNeighbors(DirectedLinkPredictor):
    """有向图版本的共同邻居"""
    def predict(self, G: nx.DiGraph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        edges = np.array(G.edges())
        scores = []
        for u, v in node_pairs:
            # u的出边邻居与v的入边邻居的交集
            u_out, _ = get_directed_neighbors(edges, u)
            _, v_in = get_directed_neighbors(edges, v)
            scores.append(len(u_out.intersection(v_in)))
        return np.array(scores)

class DirectedJaccardCoefficient(DirectedLinkPredictor):
    """有向图版本的Jaccard系数"""
    def predict(self, G: nx.DiGraph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        edges = np.array(G.edges())
        scores = []
        for u, v in node_pairs:
            u_out, _ = get_directed_neighbors(edges, u)
            _, v_in = get_directed_neighbors(edges, v)
            intersection = len(u_out.intersection(v_in))
            union = len(u_out.union(v_in))
            scores.append(intersection / union if union > 0 else 0)
        return np.array(scores)

class DirectedAdamicAdar(DirectedLinkPredictor):
    """有向图版本的Adamic-Adar指数"""
    def predict(self, G: nx.DiGraph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        edges = np.array(G.edges())
        scores = []
        for u, v in node_pairs:
            u_out, _ = get_directed_neighbors(edges, u)
            _, v_in = get_directed_neighbors(edges, v)
            common = u_out.intersection(v_in)
            
            score = 0
            for w in common:
                # 使用入度和出度的调和平均
                w_in, w_out = get_directed_neighbors(edges, w)
                if len(w_in) > 0 and len(w_out) > 0:
                    degree = 2 / (1/len(w_in) + 1/len(w_out))
                    score += 1 / math.log(degree + 1)
            scores.append(score)
        return np.array(scores)

class DirectedResourceAllocation(DirectedLinkPredictor):
    """有向图版本的资源分配指数"""
    def predict(self, G: nx.DiGraph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        edges = np.array(G.edges())
        scores = []
        for u, v in node_pairs:
            u_out, _ = get_directed_neighbors(edges, u)
            _, v_in = get_directed_neighbors(edges, v)
            common = u_out.intersection(v_in)
            
            score = 0
            for w in common:
                w_in, w_out = get_directed_neighbors(edges, w)
                # 使用入度和出度的调和平均
                if len(w_in) > 0 and len(w_out) > 0:
                    degree = 2 / (1/len(w_in) + 1/len(w_out))
                    score += 1 / degree
            scores.append(score)
        return np.array(scores)

class DirectedPreferentialAttachment(DirectedLinkPredictor):
    """有向图版本的优先连接"""
    def predict(self, G: nx.DiGraph, node_pairs: List[Tuple[int, int]]) -> np.ndarray:
        edges = np.array(G.edges())
        scores = []
        for u, v in node_pairs:
            # 使用u的出度和v的入度的乘积
            u_out, _ = get_directed_neighbors(edges, u)
            _, v_in = get_directed_neighbors(edges, v)
            scores.append(len(u_out) * len(v_in))
        return np.array(scores)

