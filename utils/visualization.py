import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import os

def setup_chinese_font():
    """设置matplotlib的中文字体"""
    font_paths = [
        'C:/Windows/Fonts/SimHei.ttf',  # Windows
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Linux
        '/System/Library/Fonts/PingFang.ttc'  # macOS
    ]

    font_set = False
    for font_path in font_paths:
        try:
            mpl.font_manager.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            font_set = True
            break
        except:
            continue

    if not font_set:
        print("警告：未能成功设置中文字体，图表中的中文可能无法正常显示")

def plot_combined_results(traditional_results, gnn_results, dataset_name, save_path='results'):
    """绘制传统方法和GNN方法的对比图"""
    setup_chinese_font()
    plt.figure(figsize=(15, 10))

    # 准备数据
    methods = []
    aucs = []
    aps = []
    colors = []

    # 传统方法数据
    trad_methods = {
        'Common Neighbors': '共同邻居',
        'Jaccard': 'Jaccard系数',
        'Adamic-Adar': 'Adamic-Adar',
        'Resource Allocation': '资源分配',
        'Preferential Attachment': '优先连接'
    }

    for name in trad_methods:
        if name in traditional_results:
            methods.append(trad_methods[name])
            aucs.append(traditional_results[name]['auc'])
            aps.append(traditional_results[name]['ap'])
            colors.append('lightblue')

    # GNN方法数据
    gnn_name_map = {
        'gcn': 'GCN',
        'graphsage': 'GraphSAGE',
        'gat': 'GAT'
    }

    for name, results in gnn_results.items():
        methods.append(gnn_name_map.get(name, name))
        aucs.append(results['test_auc'])
        aps.append(results['test_ap'])
        colors.append('lightcoral')

    # 绘制图表
    x = np.arange(len(methods))
    width = 0.35

    # AUC对比
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(x, aucs, width, color=colors)
    plt.ylabel('AUC值')
    plt.title(f'{dataset_name}数据集上各方法的AUC比较')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    # AP对比
    plt.subplot(2, 1, 2)
    bars2 = plt.bar(x, aps, width, color=colors)
    plt.ylabel('AP值')
    plt.title(f'{dataset_name}数据集上各方法的AP比较')
    plt.xticks(x, methods, rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')

    # 添加图例
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightblue', label='传统方法'),
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', label='GNN方法')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    save_name = f'combined_results_{dataset_name}.png'
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n结果对比图已保存为: {save_path}")
    plt.close()
