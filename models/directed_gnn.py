import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from typing import Tuple, Optional

class DGCN(nn.Module):
    """有向图卷积网络"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 正向和反向卷积层
        self.forward_convs = nn.ModuleList()
        self.backward_convs = nn.ModuleList()
        
        # 第一层
        self.forward_convs.append(GCNConv(in_channels, hidden_channels))
        self.backward_convs.append(GCNConv(in_channels, hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.forward_convs.append(GCNConv(2 * hidden_channels, hidden_channels))
            self.backward_convs.append(GCNConv(2 * hidden_channels, hidden_channels))
            
        # 最后一层
        self.forward_convs.append(GCNConv(2 * hidden_channels, hidden_channels))
        self.backward_convs.append(GCNConv(2 * hidden_channels, hidden_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 分离正向边和反向边
        row, col = edge_index
        forward_edge_index = edge_index
        backward_edge_index = torch.stack([col, row], dim=0)
        
        for i in range(self.num_layers):
            # 正向和反向传播
            x_forward = self.forward_convs[i](x, forward_edge_index)
            x_backward = self.backward_convs[i](x, backward_edge_index)
            
            # 连接正向和反向特征
            x = torch.cat([x_forward, x_backward], dim=1)
            
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class DirectedGraphSAGE(nn.Module):
    """有向GraphSAGE"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 入边和出边卷积层
        self.in_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        
        # 第一层
        self.in_convs.append(SAGEConv(in_channels, hidden_channels))
        self.out_convs.append(SAGEConv(in_channels, hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.in_convs.append(SAGEConv(2 * hidden_channels, hidden_channels))
            self.out_convs.append(SAGEConv(2 * hidden_channels, hidden_channels))
            
        # 最后一层
        self.in_convs.append(SAGEConv(2 * hidden_channels, hidden_channels))
        self.out_convs.append(SAGEConv(2 * hidden_channels, hidden_channels))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 分离入边和出边
        row, col = edge_index
        in_edge_index = edge_index
        out_edge_index = torch.stack([col, row], dim=0)
        
        for i in range(self.num_layers):
            # 分别聚合入边和出边邻居
            x_in = self.in_convs[i](x, in_edge_index)
            x_out = self.out_convs[i](x, out_edge_index)
            
            # 连接入边和出边特征
            x = torch.cat([x_in, x_out], dim=1)
            
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x

class DirectedGAT(nn.Module):
    """有向图注意力网络"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, 
                 heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.heads = heads
        
        # 入边和出边注意力层
        self.in_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        
        # 第一层
        self.in_convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        self.out_convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.in_convs.append(
                GATConv(2 * hidden_channels, hidden_channels // heads, heads=heads))
            self.out_convs.append(
                GATConv(2 * hidden_channels, hidden_channels // heads, heads=heads))
            
        # 最后一层 (单头注意力)
        self.in_convs.append(GATConv(2 * hidden_channels, hidden_channels, heads=1))
        self.out_convs.append(GATConv(2 * hidden_channels, hidden_channels, heads=1))
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 分离入边和出边
        row, col = edge_index
        in_edge_index = edge_index
        out_edge_index = torch.stack([col, row], dim=0)
        
        for i in range(self.num_layers):
            # 分别计算入边和出边注意力
            x_in = self.in_convs[i](x, in_edge_index)
            x_out = self.out_convs[i](x, out_edge_index)
            
            # 连接入边和出边特征
            x = torch.cat([x_in, x_out], dim=1)
            
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if i != self.num_layers - 2:
                    x = x.view(-1, 2 * self.heads * (self.hidden_channels // self.heads))
        
        return x

class DirectedGNNTrainer:
    """有向图GNN模型训练器"""
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def train_step(self, 
                  x: torch.Tensor,
                  edge_index: torch.Tensor,
                  pos_edge_index: torch.Tensor,
                  neg_edge_index: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """单步训练"""
        self.model.train()
        optimizer.zero_grad()
        
        # 前向传播
        z = self.model(x.to(self.device), edge_index.to(self.device))
        
        # 计算正样本和负样本的分数
        pos_score = self.predict_link_score(z, pos_edge_index.to(self.device))
        neg_score = self.predict_link_score(z, neg_edge_index.to(self.device))
        
        # 构建标签
        pos_label = torch.ones(pos_score.size(0), device=self.device)
        neg_label = torch.zeros(neg_score.size(0), device=self.device)
        
        # 计算损失
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([pos_label, neg_label])
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def predict_link_score(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """计算边的预测分数"""
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=-1)
    
    @torch.no_grad()
    def evaluate(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                pos_edge_index: torch.Tensor,
                neg_edge_index: torch.Tensor) -> Tuple[float, float, float, float]:
        """模型评估"""
        self.model.eval()
        
        z = self.model(x.to(self.device), edge_index.to(self.device))
        pos_score = self.predict_link_score(z, pos_edge_index.to(self.device))
        neg_score = self.predict_link_score(z, neg_edge_index.to(self.device))
        
        # 计算评估指标
        pos_score = pos_score.cpu().numpy()
        neg_score = neg_score.cpu().numpy()
        
        scores = np.concatenate([pos_score, neg_score])
        labels = np.concatenate([np.ones(pos_score.shape[0]), np.zeros(neg_score.shape[0])])
        
        from sklearn.metrics import roc_auc_score, average_precision_score
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        # 计算Precision@K和Recall@K
        k = min(100, len(pos_score))  # 取前100个或所有正样本
        all_scores = np.concatenate([pos_score, neg_score])
        all_labels = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        
        # 按分数排序
        indices = np.argsort(all_scores)[::-1]
        labels_at_k = all_labels[indices[:k]]
        
        precision_at_k = np.sum(labels_at_k) / k
        recall_at_k = np.sum(labels_at_k) / np.sum(all_labels)
        
        return auc, ap, precision_at_k, recall_at_k 