import os
from typing import Dict, Any
import torch

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
CACHE_DIR = os.path.join(PROCESSED_DATA_DIR, 'cache')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 数据集配置
DATASETS = {
    'arxiv': {
        'name': 'CA-AstroPh.txt',
        'path': os.path.join(RAW_DATA_DIR, 'arxiv', 'CA-AstroPh.txt'),
        'processed_path': os.path.join(CACHE_DIR, 'arxiv_processed.pkl'),
        'gnn_data_path': os.path.join(CACHE_DIR, 'arxiv_gnn_data.pkl'),
        'description': 'Arxiv Astro Physics collaboration network',
        'type': 'collaboration',
        'stats': {
            'nodes': 18772,
            'edges': 198110
        }
    },
    'dblp': {
        'name': 'com-dblp.ungraph.txt',
        'path': os.path.join(RAW_DATA_DIR, 'dblp-citation', 'com-dblp.ungraph.txt'),
        'processed_path': os.path.join(CACHE_DIR, 'dblp_processed.pkl'),
        'gnn_data_path': os.path.join(CACHE_DIR, 'dblp_gnn_data.pkl'),
        'description': 'DBLP collaboration network',
        'type': 'collaboration',
        'stats': {
            'nodes': 317080,
            'edges': 1049866
        }
    },
    'hep-ph': {
        'name': 'cit-HepPh.txt',
        'path': os.path.join(RAW_DATA_DIR, 'pubmed-diabetes', 'cit-HepPh.txt'),
        'processed_path': os.path.join(CACHE_DIR, 'hep-ph_processed.pkl'),
        'gnn_data_path': os.path.join(CACHE_DIR, 'hep-ph_gnn_data.pkl'),
        'description': 'High Energy Physics citation network',
        'type': 'citation',
        'stats': {
            'nodes': 34546,
            'edges': 421578
        }
    }
}

# 实验配置
RANDOM_SEED = 42
TEST_RATIO = 0.2
VAL_RATIO = 0.1
NON_EDGES_SAMPLE_RATIO = 1.0  # 负样本采样比例（相对于正样本数量）

# 特征类型配置
FEATURE_TYPES = {
    'degree': '使用节点度作为特征',
    'onehot': '使用独热编码作为特征',
    'spectral': '使用谱嵌入作为特征（16维）'
}

# GNN训练配置
TRAIN_CONFIG = {
    'epochs': 200,
    'learning_rate': 0.001,
    'hidden_channels': 64,
    'num_layers': 2,
    'dropout': 0.5,
    'patience': 20,
    'batch_size': 512,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 模型配置
TRADITIONAL_MODELS = {
    'common_neighbors': {
        'name': 'Common Neighbors',
        'class': 'CommonNeighbors'
    },
    'jaccard_coefficient': {
        'name': 'Jaccard Coefficient',
        'class': 'JaccardCoefficient'
    },
    'adamic_adar': {
        'name': 'Adamic Adar',
        'class': 'AdamicAdar'
    },
    'resource_allocation': {
        'name': 'Resource Allocation',
        'class': 'ResourceAllocation'
    },
    'preferential_attachment': {
        'name': 'Preferential Attachment',
        'class': 'PreferentialAttachment'
    }
}

GNN_MODELS = {
    'gcn': {
        'name': 'GCN',
        'class': 'GCN',
        'save_path': os.path.join(RESULTS_DIR, 'gcn_best_model.pt')
    },
    'graphsage': {
        'name': 'GraphSAGE',
        'class': 'GraphSAGE',
        'save_path': os.path.join(RESULTS_DIR, 'graphsage_best_model.pt')
    },
    'gat': {
        'name': 'GAT',
        'class': 'GAT',
        'save_path': os.path.join(RESULTS_DIR, 'gat_best_model.pt')
    }
}

# 评估配置
METRICS = {
    'auc_roc': True,
    'precision_at_k': True,
    'recall_at_k': True,
    'average_precision': True
}
TOP_K = 10

# 可视化配置
VISUALIZATION = {
    'font_family': 'SimHei',
    'figure_size': (10, 6),
    'dpi': 100,
    'colors': {
        'line': 'darkorange',
        'baseline': 'navy'
    }
}

# GNN模型配置
GNN_CONFIG = {
    'gcn': {
        'in_channels': 16,      # 修改为16，匹配谱嵌入特征维度
        'hidden_channels': 64,  # 隐藏层维度
        'num_layers': 2,       # 层数
        'dropout': 0.5         # dropout率
    },
    'graphsage': {
        'in_channels': 16,
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.5
    },
    'gat': {
        'in_channels': 16,
        'hidden_channels': 64,
        'num_layers': 2,
        'dropout': 0.5,
        'heads': 4            # GAT特有的注意力头数
    }
}
