import torch
import numpy as np
from node2vec import Node2Vec
import networkx as nx
from typing import List, Tuple

class Node2VecPredictor:
    """使用Node2Vec进行链接预测的模型"""
    def __init__(self, dimensions=128, walk_length=30, num_walks=200, workers=4):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.embeddings = None
        
    def fit(self, G: nx.Graph):
        """训练Node2Vec模型获取节点嵌入"""
        # 初始化Node2Vec模型
        node2vec = Node2Vec(
            graph=G,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=self.workers
        )
        
        # 训练模型
        model = node2vec.fit(window=10, min_count=1)
        
        # 获取所有节点的嵌入
        self.embeddings = {}
        for node in G.nodes():
            self.embeddings[node] = model.wv[str(node)]
            
    def predict(self, edge_list: List[Tuple[int, int]]) -> np.ndarray:
        """预测边的存在概率"""
        if self.embeddings is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
            
        scores = []
        for u, v in edge_list:
            # 使用点积作为相似度度量
            score = np.dot(self.embeddings[u], self.embeddings[v])
            scores.append(score)
            
        return np.array(scores) 