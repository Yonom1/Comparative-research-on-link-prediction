import os
import pickle
from config import DATASETS
from models.traditional import CommonNeighbors, JaccardCoefficient, AdamicAdar, ResourceAllocation, PreferentialAttachment
from models.gnn import GCN, GraphSAGE, GAT
from utils.data_loader import load_edge_list, build_graph
from utils.data_processor import prepare_data_for_gnn
from test_traditional import evaluate_predictor
from train_gnn import train_model
import matplotlib.pyplot as plt
import pandas as pd

# 1. 数据预处理（带缓存）
def load_or_cache_data(dataset_key):
    cache_dir = os.path.join('data', 'processed', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{dataset_key}_gnn_data.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            loaded = pickle.load(f)
        if len(loaded) == 5:
            G, data, train_edges, val_edges, test_edges = loaded
        elif len(loaded) == 4:
            data, train_edges, val_edges, test_edges = loaded
            # 兼容旧缓存，重新生成G
            print(f"[警告] 旧缓存格式，自动重建 networkx Graph ...")
            if hasattr(data, 'edge_index'):
                import torch
                import networkx as nx
                edge_index = data.edge_index.cpu().numpy()
                edges = [(int(u), int(v)) for u, v in zip(edge_index[0], edge_index[1])]
                G = nx.Graph()
                G.add_edges_from(edges)
            else:
                raise ValueError("data对象不包含edge_index，无法重建Graph！请清理缓存后重试。")
        else:
            raise ValueError("缓存文件格式错误！")
        print(f"[缓存] 加载 {dataset_key} 数据集")
    else:
        print(f"[原始] 处理 {dataset_key} 数据集")
        edges = load_edge_list(DATASETS[dataset_key]['path'])
        G = build_graph(edges)
        data, train_edges, val_edges, test_edges = prepare_data_for_gnn(G)
        with open(cache_file, 'wb') as f:
            pickle.dump((G, data, train_edges, val_edges, test_edges), f)
    return G, data, train_edges, val_edges, test_edges

# 2. 评估所有传统方法
def run_traditional_methods(G, test_edges):
    from utils.data_loader import sample_non_edges
    import torch
    import networkx as nx
    methods = {
        'common_neighbors': CommonNeighbors,
        'jaccard_coefficient': JaccardCoefficient,
        'adamic_adar': AdamicAdar,
        'resource_allocation': ResourceAllocation,
        'preferential_attachment': PreferentialAttachment
    }

    # 转换测试边为二元组列表格式
    if isinstance(test_edges, torch.Tensor):
        test_edges = test_edges.t().numpy()
        test_edges = [(int(u), int(v)) for u, v in test_edges]
    else:
        test_edges = list(test_edges)

    # 检查节点是否都在图中
    nodes_in_graph = set(G.nodes())
    valid_test_edges = []
    invalid_edges_count = 0
    for u, v in test_edges:
        if u in nodes_in_graph and v in nodes_in_graph:
            valid_test_edges.append((u, v))
        else:
            invalid_edges_count += 1

    print(f"[信息] 原始测试边数量: {len(test_edges)}")
    print(f"[信息] 有效测试边数量: {len(valid_test_edges)}")
    print(f"[信息] 节点不在图中的边数量: {invalid_edges_count}")
    print(f"[信息] 图中节点数量: {G.number_of_nodes()}")
    print(f"[信息] 图中边数量: {G.number_of_edges()}")

    # 采样负样本（确保节点在图中）
    sampled_non_edges = sample_non_edges(G, sample_size=10000, random_state=42)
    sampled_non_edges = list(sampled_non_edges)

    results = {}
    for name, predictor_class in methods.items():
        print(f"[传统方法] 正在评估: {name}")
        predictor = predictor_class()
        auc, ap = evaluate_predictor(predictor, G, valid_test_edges, sampled_non_edges)
        results[name] = {'auc': auc, 'ap': ap}
        print(f"[传统方法] {name} 评估完成: AUC={auc:.4f}, AP={ap:.4f}")
    return results

# 3. 训练和测试GNN模型
def run_gnn_models(data, train_edges, val_edges, test_edges):
    gnn_models = {
        'gcn': GCN,
        'graphsage': GraphSAGE,
        'gat': GAT
    }
    gnn_results = {}
    for name, model_class in gnn_models.items():
        print(f"\n训练 {name.upper()} ...")
        train_model(name, data, train_edges, val_edges, test_edges, model_class, only_train=True)
        print(f"\n测试 {name.upper()} ...")
        res = train_model(name, data, train_edges, val_edges, test_edges, model_class, only_train=False)
        gnn_results[name] = res
    return gnn_results

# 4. 结果输出（图片、表格、日志）
def save_results(traditional_results, gnn_results, dataset_key):
    # 合并结果
    all_results = {}
    for method, res in traditional_results.items():
        all_results[method] = {'AUC': res['auc'], 'AP': res['ap']}
    for model, res in gnn_results.items():
        all_results[model] = {'AUC': res['test_auc'], 'AP': res['test_ap']}
    df = pd.DataFrame(all_results).T
    df.to_csv(f'results/{dataset_key}_results.csv')
    print(f"结果表格已保存: results/{dataset_key}_results.csv")
    # 绘图
    plt.figure(figsize=(8,4))
    df[['AUC','AP']].plot(kind='bar')
    plt.title(f'{dataset_key} 各模型AUC/AP对比')
    plt.ylabel('Score')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f'results/{dataset_key}_results.png', dpi=200)
    plt.close()
    print(f"结果图片已保存: results/{dataset_key}_results.png")
    # 日志
    print(f"\n==== {dataset_key} 结果 ====")
    print(df)

if __name__ == "__main__":
    for dataset_key in DATASETS:
        print(f"\n===== 处理数据集: {dataset_key} =====")
        print("[主流程] 开始加载或缓存数据...")
        G, data, train_edges, val_edges, test_edges = load_or_cache_data(dataset_key)
        print("[主流程] 数据加载完成。\n")
        print("[主流程] 评估所有传统方法...")
        traditional_results = run_traditional_methods(G, test_edges)
        print("[主流程] 传统方法评估完成。\n")
        print("[主流程] 训练和测试所有GNN模型...")
        gnn_results = run_gnn_models(data, train_edges, val_edges, test_edges)
        print("[主流程] GNN模型训练与测试完成。\n")
        print("[主流程] 输出所有结果...")
        save_results(traditional_results, gnn_results, dataset_key)
        print(f"[主流程] {dataset_key} 数据集流程结束。\n")
    print("\n全部实验流程已顺利完成！")
