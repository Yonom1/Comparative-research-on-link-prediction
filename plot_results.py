import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 算法名称映射
algorithm_names = {
    'common_neighbors': 'Common\nNeighbors',
    'jaccard_coefficient': 'Jaccard\nCoefficient',
    'adamic_adar': 'Adamic\nAdar',
    'resource_allocation': 'Resource\nAllocation',
    'preferential_attachment': 'Preferential\nAttachment',
    'gcn': 'GCN',
    'graphsage': 'GraphSAGE',
    'gat': 'GAT'
}

# 读取数据
arxiv_data = pd.read_csv('results/arxiv_results.csv', index_col=0)
dblp_data = pd.read_csv('results/dblp_results.csv', index_col=0)
hepph_data = pd.read_csv('results/hep-ph_results.csv', index_col=0)

# 将索引替换为更友好的显示名称
for df in [arxiv_data, dblp_data, hepph_data]:
    df.index = df.index.map(algorithm_names)

# 定义颜色方案
traditional_colors = sns.color_palette("Reds_d", n_colors=5)
gnn_colors = sns.color_palette("Blues_d", n_colors=3)

def create_plot(data, title):
    plt.figure(figsize=(14, 6))

    # 设置传统方法和GNN方法的索引
    traditional_idx = range(5)
    gnn_idx = range(5, 8)

    # 绘制柱状图
    # 传统方法
    bars1 = plt.bar(traditional_idx, data['AUC'].iloc[:5], width=0.35, label='AUC (传统方法)',
                    color=traditional_colors, alpha=0.8)
    bars2 = plt.bar([x + 0.35 for x in traditional_idx], data['AP'].iloc[:5], width=0.35,
                    label='AP (传统方法)', color=traditional_colors, alpha=0.4)

    # GNN方法
    bars3 = plt.bar(gnn_idx, data['AUC'].iloc[5:], width=0.35, label='AUC (GNN方法)',
                    color=gnn_colors, alpha=0.8)
    bars4 = plt.bar([x + 0.35 for x in gnn_idx], data['AP'].iloc[5:], width=0.35,
                    label='AP (GNN方法)', color=gnn_colors, alpha=0.4)

    # 设置标签和标题
    plt.xlabel('算法', fontsize=12)
    plt.ylabel('评分', fontsize=12)
    plt.title(f'{title}数据集结果对比', fontsize=14, pad=20)

    # 设置x轴标签
    plt.xticks([i + 0.35/2 for i in range(8)], data.index, rotation=0, ha='center')

    # 添加图例
    plt.legend(loc='lower left', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'results/{title}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 创建每个数据集的图表
create_plot(arxiv_data, 'arxiv')
create_plot(dblp_data, 'dblp')
create_plot(hepph_data, 'hep-ph')
