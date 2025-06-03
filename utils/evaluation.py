from sklearn.metrics import roc_curve, auc
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> Dict[str, float]:
    """
    计算各种评估指标
    
    Args:
        y_true: 真实标签
        y_score: 预测分数
        k: top-k的k值
    
    Returns:
        包含各种指标的字典
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # 计算AUC和AP
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    
    # 计算Precision@K和Recall@K
    k = min(k, len(y_true))
    indices = np.argsort(y_score)[::-1]
    top_k_labels = y_true[indices[:k]]
    
    precision_at_k = np.sum(top_k_labels) / k
    recall_at_k = np.sum(top_k_labels) / np.sum(y_true)
    
    # 计算MRR (Mean Reciprocal Rank)
    pos_indices = np.where(y_true == 1)[0]
    pos_ranks = []
    for idx in pos_indices:
        rank = np.sum(y_score >= y_score[idx])
        pos_ranks.append(rank)
    mrr = np.mean(1.0 / np.array(pos_ranks))
    
    return {
        'AUC': auc,
        'AP': ap,
        f'Precision@{k}': precision_at_k,
        f'Recall@{k}': recall_at_k,
        'MRR': mrr
    }

def plot_roc_curves(results: Dict[str, Dict[str, np.ndarray]], save_path: Optional[str] = None):
    """
    绘制ROC曲线对比图
    
    Args:
        results: 包含各个模型预测结果的字典
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, data in results.items():
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_score'])
        auc = data['metrics']['AUC']
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title('ROC曲线对比')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_pr_curves(results: Dict[str, Dict[str, np.ndarray]], save_path: Optional[str] = None):
    """
    绘制PR曲线对比图
    
    Args:
        results: 包含各个模型预测结果的字典
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, data in results.items():
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_score'])
        ap = data['metrics']['AP']
        plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('PR曲线对比')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(results: Dict[str, Dict[str, float]], 
                          metrics: List[str],
                          save_path: Optional[str] = None):
    """
    绘制不同模型在各个指标上的对比柱状图
    
    Args:
        results: 包含各个模型评估指标的字典
        metrics: 要对比的指标列表
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    
    for i, (model_name, model_results) in enumerate(results.items()):
        values = [model_results[metric] for metric in metrics]
        plt.bar(x + i * width, values, width, label=model_name)
    
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    plt.title('模型性能对比')
    plt.xticks(x + width * (len(results) - 1) / 2, metrics)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_ablation_study(ablation_results: Dict[str, Dict[str, float]],
                       metrics: List[str],
                       save_path: Optional[str] = None):
    """
    绘制消融实验结果的热力图
    
    Args:
        ablation_results: 包含消融实验结果的字典
        metrics: 要展示的指标列表
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    
    # 构建热力图数据
    data = []
    for model_name in ablation_results:
        row = [ablation_results[model_name][metric] for metric in metrics]
        data.append(row)
    
    data = np.array(data)
    
    # 绘制热力图
    sns.heatmap(data, 
                annot=True, 
                fmt='.3f',
                xticklabels=metrics,
                yticklabels=list(ablation_results.keys()),
                cmap='YlOrRd')
    
    plt.title('消融实验结果')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def print_top_predictions(predictions: List[Tuple[str, str, float]], k: int) -> None:
    """打印前K个预测结果"""
    print(f"\n预测最可能合作的前{k}组学者：")
    for u, v, score in predictions[:k]:
        print(f"作者 {u} 和 {v} | 共同合作者数量: {int(score)}") 