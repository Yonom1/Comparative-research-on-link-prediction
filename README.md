# 链接预测算法对比研究

本项目实现并比较了多种链接预测算法在不同数据集上的表现，包括传统启发式算法和基于图神经网络(GNN)的方法。

## 项目结构

```
.
├── config.py                 # 配置文件
├── download_datasets.py      # 数据集下载脚本
├── extract_datasets.py       # 数据集提取处理脚本
├── main.py                  # 主程序入口
├── train_gnn.py             # GNN模型训练脚本
├── test_gnn.py              # GNN模型测试脚本
├── test_traditional.py      # 传统算法测试脚本
├── plot_results.py          # 结果可视化脚本
├── data/                    # 数据目录
│   ├── processed/           # 处理后的数据
│   └── raw/                 # 原始数据
├── models/                  # 模型实现
│   ├── traditional.py       # 传统链接预测算法
│   ├── gnn.py              # GNN模型实现
│   └── directed_gnn.py     # 有向图GNN模型
├── results/                 # 实验结果
└── utils/                   # 工具函数
    ├── data_loader.py      # 数据加载
    ├── data_processor.py   # 数据预处理
    ├── evaluation.py       # 评估指标
    └── visualization.py    # 可视化工具
```

## 实现的算法

### 传统启发式算法
- Common Neighbors
- Jaccard Coefficient
- Adamic Adar
- Resource Allocation
- Preferential Attachment

### 图神经网络方法
- GCN (Graph Convolutional Network)
- GraphSAGE
- GAT (Graph Attention Network)

## 数据集

本项目使用了以下三个真实世界的引文网络数据集：
- arXiv Citation Network
- DBLP Citation Network
- HEP-PH Citation Network

## 环境要求

```bash
# 创建环境
conda create -n network python=3.10
conda activate network

# 安装依赖
pip install -r requirements.txt
```

## 使用方法

1. 下载并处理数据集：
```bash
python download_datasets.py
python extract_datasets.py
```

2. 运行传统算法测试：
```bash
python test_traditional.py
```

3. 训练和测试GNN模型：
```bash
python train_gnn.py
python test_gnn.py
```

4. 可视化结果对比：
```bash
python plot_results.py
```

## 实验结果

实验结果保存在 `results/` 目录下：
- `*_results.csv`：各算法的具体评估指标
- `*_comparison.png`：可视化对比图表
- `*_best_model.pt`：最佳GNN模型参数

主要评估指标包括：
- AUC (Area Under the ROC Curve)
- AP (Average Precision)

## 主要发现

1. 在所有三个数据集上，传统启发式算法整体表现优于GNN方法
2. Resource Allocation和Adamic Adar算法展现出最佳性能
3. 在GNN方法中，GCN和GraphSAGE的表现相对较好
4. 数据集特性对算法性能有显著影响

## Future Work

1. 时序特性探索
   - 引入边的产生时间信息，研究时序链接预测
   - 设计基于时间序列的特征提取方法
   - 分析网络演化模式对链接预测的影响
   - 研究节点关系随时间变化的规律

2. 模型改进
   - 设计适用于有向图的GNN架构
   - 整合节点属性和边属性信息
   - 改进现有模型的聚合函数和更新函数
   - 探索新的注意力机制以提升性能

3. 可扩展性研究
   - 设计分布式训练框架支持大规模图计算
   - 研究高效的图采样和批处理策略
   - 优化内存使用和计算效率
   - 实现增量学习以支持动态图更新

4. 应用场景拓展
   - 扩展到异质图网络的链接预测
   - 探索社交网络好友推荐应用
   - 研究蛋白质相互作用网络预测
   - 应用于学术合作关系预测

## 参考文献

1. Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
2. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive representation learning on large graphs. In Advances in neural information processing systems.
3. Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.
