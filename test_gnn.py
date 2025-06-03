import torch
import networkx as nx
from utils.data_loader import load_edge_list, build_graph
from utils.data_processor import prepare_data_for_gnn, sample_negative_edges
from models.gnn import GCN, GraphSAGE, GAT, GNNTrainer
import numpy as np
from tqdm import tqdm
import time
import os
import pickle

def prepare_or_load_data(cache_file='data/processed/arxiv_processed.pkl'):
    """加载或预处理数据，支持缓存"""
    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        print("\n发现缓存数据，正在加载...")
        start_time = time.time()
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"缓存数据加载完成! 耗时: {time.time() - start_time:.2f}秒")
        return cache_data
    
    print("\n未发现缓存数据，开始预处理...")
    
    # 加载数据
    print("\n1. 数据加载阶段")
    print("-" * 20)
    start_time = time.time()
    edges = load_edge_list('data/raw/arxiv/CA-AstroPh.txt')
    G = build_graph(edges)
    print(f"数据加载完成! 耗时: {time.time() - start_time:.2f}秒")
    print(f"图的统计信息:")
    print(f"- 节点数: {G.number_of_nodes():,}")
    print(f"- 边数: {G.number_of_edges():,}")
    
    # 准备数据
    print("\n2. 数据预处理阶段")
    print("-" * 20)
    start_time = time.time()
    data, train_edges, val_edges, test_edges = prepare_data_for_gnn(
        G, 
        test_ratio=0.1,
        val_ratio=0.05,
        feature_type='spectral'  # 使用谱特征
    )
    print(f"数据预处理完成! 耗时: {time.time() - start_time:.2f}秒")
    
    # 保存缓存
    print("\n保存预处理结果到缓存...")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump((data, train_edges, val_edges, test_edges), f)
    print("缓存保存完成!")
    
    return data, train_edges, val_edges, test_edges

def train_model(model_name: str, data, train_edges, val_edges, test_edges):
    """在Arxiv数据集上训练指定的GNN模型"""
    print(f"\n{'='*50}")
    print(f"开始训练 {model_name}")
    print(f"{'='*50}")
    
    # 创建模型
    print("\n1. 模型初始化阶段")
    print("-" * 20)
    if model_name == 'gcn':
        model = GCN(
            in_channels=data.x.size(1),
            hidden_channels=64,
            num_layers=2,
            dropout=0.5
        )
    elif model_name == 'graphsage':
        model = GraphSAGE(
            in_channels=data.x.size(1),
            hidden_channels=64,
            num_layers=2,
            dropout=0.5
        )
    elif model_name == 'gat':
        model = GAT(
            in_channels=data.x.size(1),
            hidden_channels=64,
            num_layers=2,
            heads=4,
            dropout=0.5
        )
    
    # 创建训练器
    trainer = GNNTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练循环
    print("\n2. 模型训练阶段")
    print("-" * 20)
    print("训练配置:")
    print("- 学习率: 0.01")
    print("- Weight Decay: 5e-4")
    print("- 最大Epoch数: 10")  # 减少epoch数用于测试
    
    best_val_auc = 0
    best_epoch = 0
    
    print("\n开始训练...")
    start_time = time.time()
    for epoch in tqdm(range(10), desc="训练进度"):  # 只训练10个epoch用于测试
        # 采样负边
        neg_train_edges = sample_negative_edges(
            train_edges, 
            data.x.size(0), 
            train_edges.size(1)
        )
        
        # 训练一个epoch
        loss = trainer.train_step(data, optimizer, train_edges, neg_train_edges)
        
        # 验证
        neg_val_edges = sample_negative_edges(
            val_edges, 
            data.x.size(0), 
            val_edges.size(1)
        )
        val_auc, val_ap = trainer.test(data, val_edges, neg_val_edges)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
        
        print(f"\nEpoch {epoch+1:03d}:")
        print(f"Loss: {loss:.4f}")
        print(f"验证集 AUC: {val_auc:.4f}")
        print(f"验证集 AP: {val_ap:.4f}")
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {training_time:.2f}秒")
    print(f"最佳验证集性能 (Epoch {best_epoch+1}):")
    print(f"AUC: {best_val_auc:.4f}")
    
    # 测试
    print("\n3. 模型评估阶段")
    print("-" * 20)
    print("在测试集上评估...")
    neg_test_edges = sample_negative_edges(
        test_edges, 
        data.x.size(0), 
        test_edges.size(1)
    )
    test_auc, test_ap = trainer.test(data, test_edges, neg_test_edges)
    print(f"\n测试集结果:")
    print(f"AUC: {test_auc:.4f}")
    print(f"AP: {test_ap:.4f}")
    
    return test_auc, test_ap

def main():
    """测试所有GNN模型"""
    # 准备或加载数据
    data, train_edges, val_edges, test_edges = prepare_or_load_data()
    
    results = {}
    # 测试所有模型
    for model_name in ['gcn', 'graphsage', 'gat']:
        test_auc, test_ap = train_model(model_name, data, train_edges, val_edges, test_edges)
        results[model_name] = {
            'auc': test_auc,
            'ap': test_ap
        }
    
    # 打印结果表格
    print("\n最终测试结果汇总")
    print("=" * 40)
    print(f"{'模型':<15}{'AUC':<15}{'AP':<15}")
    print("-" * 40)
    for model_name in results:
        auc = results[model_name]['auc']
        ap = results[model_name]['ap']
        print(f"{model_name:<15}{auc:.4f}{ap:>15.4f}")

if __name__ == "__main__":
    main() 