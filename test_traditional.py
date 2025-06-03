import networkx as nx
import numpy as np
from utils.data_loader import load_edge_list, build_graph, split_edges, sample_non_edges
from models.traditional import (
    CommonNeighbors,
    JaccardCoefficient,
    AdamicAdar,
    ResourceAllocation,
    PreferentialAttachment
)
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def evaluate_predictor(predictor, G_train, test_edges, sampled_non_edges):
    """评估预测器的性能"""
    # 预测分数
    pos_scores = predictor.predict(G_train, test_edges)
    neg_scores = predictor.predict(G_train, sampled_non_edges)
    
    # 计算评估指标
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_score = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    return auc, ap

def plot_results(results):
    """绘制结果对比图"""
    plt.figure(figsize=(15, 6))
    
    # 准备数据
    methods = list(results.keys())
    method_names = {  # 中文名称映射
        'Common Neighbors': '共同邻居',
        'Jaccard': 'Jaccard系数',
        'Adamic-Adar': 'Adamic-Adar',
        'Resource Allocation': '资源分配',
        'Preferential Attachment': '优先连接'
    }
    methods_zh = [method_names[m] for m in methods]
    aucs = [results[m]['auc'] for m in methods]
    aps = [results[m]['ap'] for m in methods]
    
    # 设置x轴位置
    x = np.arange(len(methods))
    width = 0.35
    
    # 绘制AUC柱状图
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(x, aucs, width, label='AUC')
    plt.xlabel('方法')
    plt.ylabel('AUC值')
    plt.title('各方法的AUC比较')
    plt.xticks(x, methods_zh, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # 绘制AP柱状图
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(x, aps, width, label='AP')
    plt.xlabel('方法')
    plt.ylabel('AP值')
    plt.title('各方法的AP比较')
    plt.xticks(x, methods_zh, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 在柱状图上添加数值标签
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('traditional_methods_comparison.png', dpi=300, bbox_inches='tight')
    print("\n结果对比图已保存为: traditional_methods_comparison.png")
    plt.close()

def main():
    """测试所有传统方法"""
    # 设置中文字体
    setup_chinese_font()
    
    # 加载数据
    print("1. 数据加载阶段")
    print("-" * 20)
    start_time = time.time()
    edges = load_edge_list('data/raw/arxiv/CA-AstroPh.txt')
    G = build_graph(edges)
    print(f"数据加载完成! 耗时: {time.time() - start_time:.2f}秒")
    print(f"图的统计信息:")
    print(f"- 节点数: {G.number_of_nodes():,}")
    print(f"- 边数: {G.number_of_edges():,}")
    
    # 划分数据集
    print("\n2. 数据集划分")
    print("-" * 20)
    G_train, test_edges = split_edges(G, test_size=0.1, random_state=42)
    print(f"训练集边数: {G_train.number_of_edges():,}")
    print(f"测试集边数: {len(test_edges):,}")
    
    # 采样负边
    print("\n3. 负采样")
    print("-" * 20)
    sampled_non_edges = sample_non_edges(G_train, len(test_edges), 42)
    print(f"采样负边数量: {len(sampled_non_edges):,}")
    
    # 初始化所有预测器
    predictors = {
        'Common Neighbors': CommonNeighbors(),
        'Jaccard': JaccardCoefficient(),
        'Adamic-Adar': AdamicAdar(),
        'Resource Allocation': ResourceAllocation(),
        'Preferential Attachment': PreferentialAttachment()
    }
    
    # 评估所有方法
    print("\n4. 模型评估")
    print("-" * 20)
    results = {}
    
    for name, predictor in predictors.items():
        print(f"\n评估 {name}...")
        start_time = time.time()
        auc, ap = evaluate_predictor(predictor, G_train, test_edges, sampled_non_edges)
        time_taken = time.time() - start_time
        
        results[name] = {
            'auc': auc,
            'ap': ap,
            'time': time_taken
        }
        
        print(f"- AUC: {auc:.4f}")
        print(f"- AP: {ap:.4f}")
        print(f"- 耗时: {time_taken:.2f}秒")
    
    # 打印结果表格
    print("\n最终结果汇总")
    print("=" * 60)
    print(f"{'方法':<25}{'AUC':<15}{'AP':<15}{'耗时(秒)':<10}")
    print("-" * 60)
    for name in results:
        print(f"{name:<25}{results[name]['auc']:<15.4f}{results[name]['ap']:<15.4f}{results[name]['time']:<10.2f}")
    
    # 绘制结果对比图
    plot_results(results)

if __name__ == "__main__":
    main()