import torch
import networkx as nx
from utils.data_loader import load_edge_list, build_graph
from utils.data_processor import prepare_data_for_gnn, sample_negative_edges
from models.gnn import GCN, GraphSAGE, GAT, GNNTrainer
from config import DATASETS, GNN_CONFIG, TRAIN_CONFIG
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
import os
import pickle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

def setup_chinese_font():
    """设置matplotlib的中文字体"""
    # 设置字体路径
    font_paths = [
        'C:/Windows/Fonts/SimHei.ttf',  # Windows
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
        '/System/Library/Fonts/PingFang.ttc'  # macOS
    ]
    
    # 尝试设置字体
    font_set = False
    for font_path in font_paths:
        try:
            mpl.font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
            plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号
            font_set = True
            break
        except:
            continue
    
    if not font_set:
        print("警告：未能成功设置中文字体，图表中的中文可能无法正常显示")

def train_model(name, data, train_edges, val_edges, test_edges, model_class, only_train=False):
    """训练或测试指定的GNN模型"""
    from config import GNN_CONFIG, TRAIN_CONFIG
    from models.gnn import GNNTrainer
    import time
    import os

    print(f"[train_model] 初始化{name.upper()}模型...")

    # 获取输入特征维度并调整模型配置
    in_channels = data.x.size(1)
    print(f"[train_model] 检测到输入特征维度: {in_channels}")

    # 根据实际输入维度调整模型配置
    model_config = GNN_CONFIG[name].copy()
    if in_channels == 1:
        # 如果是1维输入（度特征），调整模型架构
        print("[train_model] 检测到1维特征，调整模型架构...")
        model_config.update({
            'in_channels': 1,
            'hidden_channels': 16,  # 减小隐藏层维度
        })
        if name == 'gat':
            model_config['heads'] = 2  # 对GAT减少注意力头数
    else:
        # 使用原始配置（默认是为16维谱嵌入设计的）
        print("[train_model] 使用默认模型架构（适用于高维特征）...")
        model_config['in_channels'] = in_channels

    print(f"[train_model] 最终模型配置: in_channels={model_config['in_channels']}, "
          f"hidden_channels={model_config['hidden_channels']}")

    model = model_class(**model_config)
    trainer = GNNTrainer(model, patience=TRAIN_CONFIG['patience'])

    # 设置trainer的数据相关属性
    trainer.data = data
    trainer.train_edges = train_edges
    trainer.val_edges = val_edges
    trainer.test_edges = test_edges

    model_path = f"results/{name}_best_model.pt"
    if only_train:
        print(f"[train_model] 只训练{name.upper()}，不做测试...")
        start_time = time.time()
        train_losses, val_aucs, test_auc = trainer.train(
            data=data,
            train_edges=train_edges,
            val_edges=val_edges,
            test_edges=test_edges,
            epochs=TRAIN_CONFIG['epochs'],
            lr=TRAIN_CONFIG['learning_rate'],
            neg_sampling_ratio=TRAIN_CONFIG['neg_sampling_ratio']
        )
        print(f"[train_model] 训练完成! 耗时: {time.time() - start_time:.2f}秒")
        trainer.save_model(model_path)
        print(f"[train_model] 模型已保存到: {model_path}")
        return None
    else:
        if os.path.exists(model_path):
            trainer.load_model(model_path)
            print(f"[train_model] 已加载模型: {model_path}")
        else:
            print(f"[train_model] 未找到已保存模型，重新训练...")
            train_losses, val_aucs, test_auc = trainer.train(
                data=data,
                train_edges=train_edges,
                val_edges=val_edges,
                test_edges=test_edges,
                epochs=TRAIN_CONFIG['epochs'],
                lr=TRAIN_CONFIG['learning_rate'],
                neg_sampling_ratio=TRAIN_CONFIG['neg_sampling_ratio']
            )
            trainer.save_model(model_path)
            print(f"[train_model] 新模型已保存到: {model_path}")
        print(f"[train_model] 开始在测试集上评估{name.upper()}...")
        test_auc, test_ap = trainer.evaluate(test_edges)
        print(f"[train_model] 测试AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        return {'test_auc': test_auc, 'test_ap': test_ap}

def prepare_or_load_data(dataset_name: str):
    """加载或预处理数据，支持缓存"""
    # 构建缓存文件路径
    cache_dir = os.path.join('data', 'processed', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{dataset_name}_processed.pkl')
    
    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        print(f"\n发现{dataset_name}数据集的缓存，正在加载...")
        start_time = time.time()
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        print(f"缓存数据加载完成! 耗时: {time.time() - start_time:.2f}秒")
        return cache_data
    
    print(f"\n未发现{dataset_name}数据集的缓存，开始预处理...")
    
    # 加载数据
    print("\n1. 数据加载阶段")
    print("-" * 20)
    start_time = time.time()
    edges = load_edge_list(DATASETS[dataset_name]['path'])
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
        feature_type=TRAIN_CONFIG['feature_type']
    )
    print(f"数据预处理完成! 耗时: {time.time() - start_time:.2f}秒")
    
    # 保存缓存
    print(f"\n保存{dataset_name}数据集的预处理结果到缓存...")
    with open(cache_file, 'wb') as f:
        pickle.dump((data, train_edges, val_edges, test_edges), f)
    print("缓存保存完成!")
    
    return data, train_edges, val_edges, test_edges

