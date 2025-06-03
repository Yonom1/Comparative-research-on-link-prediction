import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from typing import List, Tuple, Optional, Union
import numpy as np
import os

class GNNLinkPredictor(nn.Module):
    """GNN链接预测模型的基类"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

    def create_node_features(self, num_nodes: int) -> torch.Tensor:
        """如果没有节点特征，创建默认特征（度特征或独热编码）"""
        return torch.eye(num_nodes, dtype=torch.float)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """前向传播，需要在子类中实现"""
        raise NotImplementedError

    def predict_link_score(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """计算边的预测分数"""
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=-1)

class GCN(GNNLinkPredictor):
    """图卷积网络模型"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.5):
        super().__init__(in_channels, hidden_channels, num_layers, dropout)

        self.convs = nn.ModuleList()
        # 第一层
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # 最后一层
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GraphSAGE(GNNLinkPredictor):
    """GraphSAGE模型"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.5):
        super().__init__(in_channels, hidden_channels, num_layers, dropout)

        self.convs = nn.ModuleList()
        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        # 最后一层
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GAT(GNNLinkPredictor):
    """图注意力网络模型"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int,
                 heads: int = 4, dropout: float = 0.5):
        super().__init__(in_channels, hidden_channels, num_layers, dropout)

        self.heads = heads
        self.convs = nn.ModuleList()
        # 第一层
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // heads, heads=heads))
        # 最后一层 (单头注意力)
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                # 如果不是最后一层，需要reshape多头注意力的输出
                if i != self.num_layers - 2:
                    x = x.view(-1, self.heads * (self.hidden_channels // self.heads))
        return x

class GNNTrainer:
    """GNN模型训练器"""
    def __init__(self, model, patience: int = 10):
        self.model = model
        self.patience = patience
        self.best_val_loss = float('inf')
        self.counter = 0
        self.best_metrics = None
        # 初始化数据相关属性
        self.data = None
        self.train_edges = None
        self.val_edges = None
        self.test_edges = None

    def train_epoch(self, data, optimizer, train_edges, neg_edges):
        """训练一个epoch"""
        self.model.train()
        optimizer.zero_grad()

        # 前向传播
        z = self.model(data.x, data.edge_index)

        # 计算正样本和负样本的得分
        pos_score = self.model.predict_link_score(z, train_edges)
        neg_score = self.model.predict_link_score(z, neg_edges)

        # 计算损失
        loss = self.compute_loss(pos_score, neg_score)

        # 反向传播
        loss.backward()
        optimizer.step()

        return loss.item()

    def compute_loss(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> torch.Tensor:
        """计算二元交叉熵损失"""
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
        return pos_loss + neg_loss

    def validate(self, data, val_edges, neg_val_edges):
        """验证模型性能"""
        self.model.eval()
        with torch.no_grad():
            z = self.model(data.x, data.edge_index)
            pos_score = self.model.predict_link_score(z, val_edges)
            neg_score = self.model.predict_link_score(z, neg_val_edges)
            val_loss = self.compute_loss(pos_score, neg_score)
            auc, ap = self.compute_metrics(pos_score, neg_score)
        return val_loss, auc, ap

    def compute_metrics(self, pos_score: torch.Tensor, neg_score: torch.Tensor) -> Tuple[float, float]:
        """计算AUC和AP评估指标"""
        from sklearn.metrics import roc_auc_score, average_precision_score

        scores = torch.cat([pos_score, neg_score]).cpu().numpy()
        labels = torch.cat([torch.ones(pos_score.size(0)),
                          torch.zeros(neg_score.size(0))]).cpu().numpy()

        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)

        return auc, ap

    def should_stop(self, val_loss: float, val_metrics: Tuple[float, float]) -> bool:
        """检查是否应该早停"""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_metrics = val_metrics
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    def save_checkpoint(self, path: str, epoch: int, optimizer, val_metrics: Tuple[float, float]):
        """保存模型检查点"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss
        }, path)

    def load_checkpoint(self, path: str, optimizer) -> Tuple[int, Tuple[float, float]]:
        """加载模型检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch'], checkpoint['val_metrics']

    def train(self, data, train_edges, val_edges, test_edges,
              epochs=100, lr=0.01, neg_sampling_ratio=1.0):
        """
        训练GNN模型
        Args:
            data: 图数据
            train_edges: 训练边
            val_edges: 验证边
            test_edges: 测试边
            epochs: 训练轮数
            lr: 学习率
            neg_sampling_ratio: 负采样比例
        Returns:
            train_losses: 训练损失列表
            val_aucs: 验证集AUC列表
            test_auc: 测试集AUC
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_losses = []
        val_aucs = []

        # 为验证集和测试集生成负样本边
        neg_val_edges = negative_sampling(data.edge_index, num_nodes=data.num_nodes,
                                        num_neg_samples=val_edges.size(1))
        neg_test_edges = negative_sampling(data.edge_index, num_nodes=data.num_nodes,
                                         num_neg_samples=test_edges.size(1))

        for epoch in range(epochs):
            # 为训练集生成负样本边
            neg_train_edges = negative_sampling(data.edge_index, num_nodes=data.num_nodes,
                                              num_neg_samples=int(train_edges.size(1) * neg_sampling_ratio))

            # 训练一个epoch
            loss = self.train_epoch(data, optimizer, train_edges, neg_train_edges)
            train_losses.append(loss)

            # 在验证集上评估
            val_loss, val_auc, val_ap = self.validate(data, val_edges, neg_val_edges)
            val_aucs.append(val_auc)

            # 提前停止检查
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                self.best_metrics = self.test(data, test_edges, neg_test_edges)
                # 保存最佳模型状态
                self.best_model_state = {key: value.clone() for key, value
                                       in self.model.state_dict().items()}
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f'提前停止训练! Epoch: {epoch}')
                    break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return train_losses, val_aucs, self.best_metrics['auc']

    def test(self, data, test_edges, neg_test_edges):
        """测试模型性能"""
        self.model.eval()
        with torch.no_grad():
            z = self.model(data.x, data.edge_index)
            pos_score = self.model.predict_link_score(z, test_edges)
            neg_score = self.model.predict_link_score(z, neg_test_edges)
            auc, ap = self.compute_metrics(pos_score, neg_score)
        return {'auc': auc, 'ap': ap}

    def save_model(self, path: str):
        """保存模型到文件"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics
        }, path)

    def load_model(self, path: str):
        """从文件加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_metrics = checkpoint.get('best_metrics', None)

    def evaluate(self, test_edges, data=None):
        """
        评估模型性能
        Args:
            test_edges: 测试边
            data: 可选的图数据，如果为None则使用self.data
        """
        self.model.eval()
        if data is None and self.data is None:
            raise ValueError("请提供图数据或先设置self.data!")
        data = data if data is not None else self.data

        with torch.no_grad():
            # 生成负样本边
            neg_test_edges = negative_sampling(
                data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=test_edges.size(1)
            )
            # 获取节点嵌入
            z = self.model(data.x, data.edge_index)
            # 计算正样本和负样本的分数
            pos_score = self.model.predict_link_score(z, test_edges)
            neg_score = self.model.predict_link_score(z, neg_test_edges)
            # 计算评估指标
            auc, ap = self.compute_metrics(pos_score, neg_score)
            return auc, ap
