#!/usr/bin/env python3
"""
基于PyTorch Lightning的深度学习LTR模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score
import pickle
import concurrent.futures
import multiprocessing
from sklearn.impute import SimpleImputer
import random

# Feature groups (used for feature ablations / analysis)
# NOTE: values are raw feature keys before DictVectorizer one-hot expansion.
time_feature = [
    'month_sin', 'month_cos',
    'day_of_week_sin', 'day_of_week_cos',
    'is_weekend',
]
host_feature = [
    'host_model',
    'gpu_model',
    'cpu_model',
    'quota_group',
    'manufacturer',
    'main_board',
]
fault_feature = [
    'fault_type_counts',
    'fault_level_counts',
    'fault_class_counts',
    'fault_subclass_counts',
]
frequency_feature = [
    'history_fault_count',
    'days_since_last_fault',
    'recent_fault_count',
    'recent_to_total_ratio',
    'fault_frequency',
    'mean_fault_interval',
    'min_fault_interval',
    'max_fault_interval',
    'std_fault_interval',
    'max_fault_duration',
    'min_fault_duration',
    'median_fault_duration',
    'unresolved_fault_count',
]

def flatten_features_worker(sample_data, features_to_exclude=None):
    """工作函数：展平单个样本的特征"""
    sample, features_to_exclude = sample_data
    flattened = {}
    
    for key, value in sample.features.items():
        # 检查是否应该排除该特征（作为前缀匹配）
        if features_to_exclude:
            should_exclude = any(key.startswith(prefix) for prefix in features_to_exclude)
            if should_exclude:
                continue
                
        if isinstance(value, list):
            for i, v in enumerate(value):
                flattened[f"{key}_{i}"] = v
        else:
            flattened[key] = value
    
    return flattened


class QueryBatchSampler(Sampler):
    def __init__(self, dataset, queries_per_batch=5, shuffle=True, seed=42):
        self.dataset = dataset
        self.queries_per_batch = queries_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0  # 用于跟踪epoch，配合PyTorch Lightning的随机状态
        
        # 获取所有query ID及其对应的样本索引
        self.query_to_indices = {}
        for idx, qid in enumerate(dataset.query_ids):
            if qid not in self.query_to_indices:
                self.query_to_indices[qid] = []
            self.query_to_indices[qid].append(idx)
        
        # 获取所有query ID列表
        self.query_ids = list(self.query_to_indices.keys())
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        
    def __iter__(self):
        query_ids = self.query_ids.copy()
        
        if self.shuffle:
            generator = random.Random(self.seed + self.epoch)
            generator.shuffle(query_ids)
        
        for i in range(0, len(query_ids), self.queries_per_batch):
            batch_query_ids = query_ids[i:i + self.queries_per_batch]
            
            batch_indices = []
            for qid in batch_query_ids:
                batch_indices.extend(self.query_to_indices[qid])
            
            yield batch_indices
    
    def __len__(self):
        return (len(self.query_ids) + self.queries_per_batch - 1) // self.queries_per_batch


class LTRDataset(Dataset):
    
    def __init__(
        self,
        samples,
        vectorizer=None,
        scaler=None,
        imputer=None,
        is_training=True,
        n_workers=None,
        use_multiprocessing=True,
        features_to_exclude=None,
    ):
        self.samples = samples
        self.is_training = is_training
        del samples
        self.labels = torch.tensor([sample.label for sample in self.samples], dtype=torch.float32)
        self.query_ids = [sample.features['query_id'] for sample in self.samples]

        self.features_to_exclude = ['metadata', 'query_id']
        if features_to_exclude:
            self.features_to_exclude.extend(list(features_to_exclude))
        
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 8)
        
        if use_multiprocessing :  # 只有大数据集才使用多线程
            self.features_raw = self._flatten_features_parallel(self.samples, n_workers)
        else:
            self.features_raw = self._flatten_features_sequential(self.samples)
        if vectorizer is None:
            self.vectorizer = DictVectorizer(sparse=False)
            features_vectorized = self.vectorizer.fit_transform(self.features_raw)
        else:
            self.vectorizer = vectorizer
            features_vectorized = self.vectorizer.transform(self.features_raw)
        
        if imputer is None:
            self.imputer = SimpleImputer(strategy='median')
            features_imputed = self.imputer.fit_transform(features_vectorized)
        else:
            self.imputer = imputer
            features_imputed = self.imputer.transform(features_vectorized)

        if scaler is None:
            self.scaler = RobustScaler()
            features_scaled = self.scaler.fit_transform(features_imputed)
        else:
            self.scaler = scaler
            features_scaled = self.scaler.transform(features_imputed)
        
        self.features = torch.tensor(features_scaled, dtype=torch.float32)
        
        # 计算查询组信息
        self.query_groups = self._compute_query_groups()
    
    def get_feature_indices(self, feature_names):
        """获取指定特征名称对应的索引位置"""
        feature_names_out = self.vectorizer.get_feature_names_out()
        indices = []
        for name in feature_names:
            # 处理特征名称匹配（可能需要处理前缀匹配）
            matches = [i for i, fname in enumerate(feature_names_out) 
                    if fname.startswith(name) or name in fname]
            indices.extend(matches)
        return list(set(indices))  # 去重

    def _flatten_features_parallel(self, samples, n_workers):
        """并行处理特征展平"""
        # 准备数据
        sample_data = [(sample, self.features_to_exclude) for sample in samples]
        
        # 使用线程池执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            features_raw = list(executor.map(flatten_features_worker, sample_data))
        
        return features_raw
    
    def _flatten_features_sequential(self, samples):
        """顺序处理特征展平（备用方案）"""
        return [flatten_features_worker((sample, self.features_to_exclude)) for sample in samples]
  

    def _compute_query_groups(self):
        """优化后的查询组计算 (O(N) 复杂度)"""
        query_groups = {}
        
        # 使用一次遍历进行分组
        for idx, qid in enumerate(self.query_ids):
            if qid not in query_groups:
                query_groups[qid] = {
                    'indices': [],
                    'size': 0
                }
            query_groups[qid]['indices'].append(idx)
            query_groups[qid]['size'] += 1
            
        return query_groups

    

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'query_id': self.query_ids[idx]
        }

class DeepLTRModel(pl.LightningModule):
    """深度学习LTR模型"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3,
                 learning_rate=1e-3, weight_decay=1e-4, loss_type='listwise', description="",
                 neg_sample_k=30, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        self.neg_sample_k = neg_sample_k

        # 构建MLP网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # 用于存储验证和测试结果
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict(self, x):
        """推理接口：进行常规的 pointwise 预测，输出风险得分（0-1之间）
        
        Args:
            x: 特征张量 [batch_size, input_dim]
        Returns:
            torch.Tensor: 风险得分 [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def listwise_loss(self, scores, labels, query_groups):
        """Listwise损失函数 (ListNet)"""
        total_loss = 0
        num_queries = 0
        
        for qid, group_info in query_groups.items():
            indices = group_info['indices']
            
            if len(indices) < 2:  # 跳过只有一个样本的查询
                continue
            
            query_scores = scores[indices]
            query_labels = labels[indices]
            
            # 计算概率分布
            score_probs = F.softmax(query_scores, dim=0)
            label_probs = F.softmax(query_labels, dim=0)
            
            # KL散度损失
            loss = F.kl_div(score_probs.log(), label_probs, reduction='sum')
            total_loss += loss
            num_queries += 1
        
        return total_loss / max(num_queries, 1)
    
    def pairwise_loss(self, logits, labels, query_groups):
        """
        按照用户流程实现的 LambdaRank Pairwise Loss
        1. 获取 Sigmoid 置信度得分
        2. 按 Query 遍历，计算正负样本笛卡尔积
        3. 计算 |delta NDCG| 作为权重
        4. 计算 w * log(1 + exp(-(s_i - s_j)))
        """
        # 模型输出置信度
        scores = torch.sigmoid(logits)
        
        total_batch_loss = 0.0
        num_queries = 0

        for qid, group_info in query_groups.items():
            indices = group_info['indices']
            if len(indices) < 2:
                continue
            
            q_logits = logits[indices]
            q_scores = scores[indices]
            q_labels = labels[indices]
            
            # 找出正负样本索引
            pos_mask = (q_labels == 1)
            neg_mask = (q_labels == 0)
            
            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]
            
            # 如果没有正样本或没有负样本，无法构建对
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            
            # 1. 计算当前 Query 的 IDCG (用于归一化 delta NDCG)
            num_pos = len(pos_idx)
            # 对于二元标签，IDCG = sum_{i=1}^{num_pos} 1/log2(i+1)
            ideal_ranks = torch.arange(1, num_pos + 1, device=logits.device).float()
            idcg = torch.sum(1.0 / torch.log2(ideal_ranks + 1.0))
            if idcg == 0: continue

            # 2. 计算当前样本的真实 Rank (按 score 降序)
            # 使用 stable sort 或者稍微抖动一下分数以处理相同分数的情况
            _, sorted_idx = torch.sort(q_scores + torch.randn_like(q_scores) * 1e-6, descending=True)
            ranks = torch.zeros_like(q_scores)
            ranks[sorted_idx] = torch.arange(1, len(q_scores) + 1, device=logits.device).float()

            # 3. 构建笛卡尔积 Pair (i, j) where i is pos, j is neg
            # i_grid: [num_pos, num_neg], j_grid: [num_pos, num_neg]
            i_grid, j_grid = torch.meshgrid(pos_idx, neg_idx, indexing='ij')
            i_flat = i_grid.reshape(-1)
            j_flat = j_grid.reshape(-1)
            
            # 4. 计算分数差 diff = s_i - s_j (使用用户指定的 sigmoid score)
            s_i = q_scores[i_flat]
            s_j = q_scores[j_flat]
            diff = s_i - s_j
            
            # 5. 计算权重 w_ij = |delta NDCG|
            # 对于二元标签，交换 i 和 j 的 delta DCG 为 |1/log2(rank_i+1) - 1/log2(rank_j+1)|
            rank_i = ranks[i_flat]
            rank_j = ranks[j_flat]
            delta_ndcg = torch.abs(1.0 / torch.log2(rank_i + 1.0) - 1.0 / torch.log2(rank_j + 1.0)) / idcg
            
            # 6. 计算 Query Loss: sum(w_ij * log(1 + exp(-diff)))
            # 使用 softplus(x) = log(1 + exp(x)) 保证数值稳定
            query_loss = torch.sum(delta_ndcg * F.softplus(-diff))
            
            total_batch_loss += query_loss
            num_queries += 1

        # 7. Batch 聚合：求所有 Query Loss 的平均值
        return total_batch_loss / max(num_queries, 1)

    def pointwise_loss(self, scores, labels, query_groups):
        """Pointwise损失函数 (MSE + 查询归一化)"""
        total_loss = 0
        num_queries = 0
        
        for qid, group_info in query_groups.items():
            indices = group_info['indices']
            
            if len(indices) < 1:  # 跳过空查询
                continue
            
            query_scores = scores[indices]
            query_labels = labels[indices]
            
            bce_loss = F.binary_cross_entropy_with_logits(query_scores, query_labels)
            
            combined_loss = bce_loss
            total_loss += combined_loss
            num_queries += 1
        
        return total_loss / max(num_queries, 1)

    def training_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['label']
        query_ids = batch['query_id']

        scores = self(features)

        # 构建查询组信息（使用indices而不是start_idx/end_idx，因为batch中query的样本可能不连续）
        query_groups = {}
        unique_qids = torch.unique(query_ids)

        for qid in unique_qids:
            mask = query_ids == qid
            indices = torch.where(mask)[0]
            query_groups[qid.item()] = {
                'indices': indices,  # 直接存储索引列表
                'size': len(indices)
            }

        # 计算损失
        if self.loss_type == 'pairwise':
            loss = self.pairwise_loss(scores, labels, query_groups)
        elif self.loss_type == 'listwise':
            loss = self.listwise_loss(scores, labels, query_groups)
        elif self.loss_type == 'pointwise':
            loss = self.pointwise_loss(scores, labels, query_groups)
        else:  # 默认使用简单的二元交叉熵
            loss = F.binary_cross_entropy_with_logits(scores, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['label']
        query_ids = batch['query_id']
        
        scores = self(features)
        
        # 存储结果用于epoch结束时计算指标
        self.validation_outputs.append({
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'query_ids': query_ids.cpu()
        })
        
        return {'scores': scores, 'labels': labels, 'query_ids': query_ids}
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        # 合并所有批次的结果
        all_scores = torch.cat([x['scores'] for x in self.validation_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_outputs])
        all_query_ids = torch.cat([x['query_ids'] for x in self.validation_outputs])
        
        # 计算指标
        metrics = self.compute_ranking_metrics(all_scores, all_labels, all_query_ids)
        
        # 记录指标
        for metric_name, value in metrics.items():
            self.log(f'val_{metric_name}', value, prog_bar=True)
        
        self.validation_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['label']
        query_ids = batch['query_id']
        
        scores = self(features)
        
        self.test_outputs.append({
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'query_ids': query_ids.cpu()
        })
        
        return {'scores': scores, 'labels': labels, 'query_ids': query_ids}
    
    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        
        # 合并所有批次的结果
        all_scores = torch.cat([x['scores'] for x in self.test_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_outputs])
        all_query_ids = torch.cat([x['query_ids'] for x in self.test_outputs])
        
        # 计算指标
        metrics = self.compute_ranking_metrics(all_scores, all_labels, all_query_ids)
        
        # 记录指标
        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value)
        
        # 保存详细排序结果
        detailed_results = {}
        unique_qids = torch.unique(all_query_ids)
        
        for qid in unique_qids:
            mask = all_query_ids == qid
            query_scores = all_scores[mask].float().cpu().numpy()
            query_labels = all_labels[mask].float().cpu().numpy()
            
            # 获取排序索引（从高到低）
            sorted_indices = np.argsort(-query_scores)
            
            # 按排序顺序保存结果
            sorted_results = []
            for rank, idx in enumerate(sorted_indices, start=1):
                sorted_results.append({
                    'score': float(query_scores[idx]),
                    'label': int(query_labels[idx]),
                    'ranking ratio': rank/len(query_labels) if len(query_labels) > 0 else 0
                })
                
            detailed_results[qid.item()] = {
                'sorted_items': sorted_results,
                'positive_count': int(np.sum(query_labels)),
                'total_count': len(query_labels)
            }
        
        import os
        import json
        from datetime import datetime
        
        os.makedirs('test_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = f'test_results/ranking_results_{timestamp}.json'
        
        with open(result_path, 'w') as f:
            json.dump(detailed_results, f, indent=4)
        with open("test_results/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n详细排序结果已保存至: {result_path}")
        
        self.test_outputs.clear()
        return metrics
    
    def compute_ranking_metrics(self, scores, labels, query_ids):
        """计算排序指标"""
        metrics = {}
        
        # 按查询分组计算指标
        unique_qids = torch.unique(query_ids)
        ndcg_scores = {'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': [], 'ndcg@30': []}
        auc_scores = []
        map_scores = []
        
        for qid in unique_qids:
            mask = query_ids == qid
            query_scores = scores[mask].float().cpu().numpy()
            query_labels = labels[mask].float().cpu().numpy()
            
            if len(set(query_labels)) < 2:  # 跳过只有一种标签的查询
                continue
            
            # 计算NDCG@k
            for k in [5, 10, 20, 30]:
                if len(query_scores) >= k:
                    ndcg_k = ndcg_score([query_labels], [query_scores], k=k)
                    ndcg_scores[f'ndcg@{k}'].append(ndcg_k)

            
            # 计算AUC
            try:
                auc = roc_auc_score(query_labels, query_scores)
                auc_scores.append(auc)
            except:
                pass
        
        # 计算 Top-K（按百分比）Precision/Recall/HitRate
        pct_list = [0.01, 0.02, 0.05, 0.10]
        precision_at_pct = {f'precision@{int(p*100)}pct': [] for p in pct_list}
        recall_at_pct = {f'recall@{int(p*100)}pct': [] for p in pct_list}
        hitrate_at_pct = {f'hitrate@{int(p*100)}pct': [] for p in pct_list}
        
        # 计算"理想情况"下的指标：如果K=总正例数时的precision
        precision_at_all_pos = []  # 如果只看所有正例，它们能排到多靠前
        recall_at_all_pos = []     # 所有正例都在top K中的最小K值（归一化）

        unique_qids = torch.unique(query_ids)
        for qid in unique_qids:
            mask = query_ids == qid
            query_scores = scores[mask].float().cpu().numpy()
            query_labels = labels[mask].float().cpu().numpy()
            n = len(query_scores)
            if n == 0:
                continue
            total_pos = int(np.sum(query_labels))
            if total_pos == 0:
                continue  # 跳过没有正例的query
            
            # 按分数从高到低排序
            sorted_idx = np.lexsort((np.arange(len(query_scores)), -query_scores))

            sorted_labels = query_labels[sorted_idx]

            # 计算按百分比的指标
            for p in pct_list:
                k = max(1, int(np.ceil(p * n)))
                top_idx = sorted_idx[:k]
                pos_in_top = int(np.sum(query_labels[top_idx]))
                
                # Precision@K = pos_in_top / K
                precision_k = pos_in_top / k
                precision_at_pct[f'precision@{int(p*100)}pct'].append(precision_k)
                
                # Recall@K = pos_in_top / total_pos
                recall_k = pos_in_top / total_pos
                recall_at_pct[f'recall@{int(p*100)}pct'].append(recall_k)
                
                # HitRate@K: top K中是否至少有一个正例
                hitrate_k = 1.0 if pos_in_top > 0 else 0.0
                hitrate_at_pct[f'hitrate@{int(p*100)}pct'].append(hitrate_k)
            
            # 计算"理想情况"：如果K=总正例数时的precision
            # 这表示"如果只看所有正例，它们能排到多靠前"
            k_ideal = min(total_pos, n)  # 取总正例数和总样本数的较小值
            top_k_ideal = sorted_idx[:k_ideal]
            pos_in_top_ideal = int(np.sum(query_labels[top_k_ideal]))
            precision_ideal = pos_in_top_ideal / k_ideal if k_ideal > 0 else 0.0
            precision_at_all_pos.append(precision_ideal)
            
            # 计算"所有正例都在top K中"的最小K值（归一化到0-1）
            # 找到最后一个正例的位置
            last_pos_idx = np.where(sorted_labels == 1)[0]
            if len(last_pos_idx) > 0:
                last_pos_rank = last_pos_idx[-1] + 1  # 排名（从1开始）
                # 归一化：如果所有正例都在前10%，则值为0.1；如果都在前50%，则值为0.5
                normalized_k = last_pos_rank / n
                recall_at_all_pos.append(normalized_k)

        # 计算平均指标（NDCG）
        for metric_name, scores_list in ndcg_scores.items():
            if scores_list:
                score_std = np.std(scores_list)
                metrics[metric_name + '_mean'] = np.mean(scores_list)
                metrics[metric_name + '_std'] = score_std
                metrics[metric_name + '_max'] = np.max(scores_list)
                metrics[metric_name + '_min'] = np.min(scores_list)

        # 计算平均指标（AUC/MAP）
        if auc_scores:
            metrics['mean_auc'] = np.mean(auc_scores)
            metrics['std_auc'] = np.std(auc_scores)
            metrics['auc_max'] = np.max(auc_scores)
            metrics['auc_min'] = np.min(auc_scores)

        # 计算平均指标（Precision/Recall@pct）
        for metric_name, values in precision_at_pct.items():
            if values:
                metrics[metric_name + '_mean'] = float(np.mean(values))
                metrics[metric_name + '_std'] = float(np.std(values))
                metrics[metric_name + '_max'] = float(np.max(values))
                metrics[metric_name + '_min'] = float(np.min(values))
        for metric_name, values in recall_at_pct.items():
            if values:
                metrics[metric_name + '_mean'] = float(np.mean(values))
                metrics[metric_name + '_std'] = float(np.std(values))
                metrics[metric_name + '_max'] = float(np.max(values))
                metrics[metric_name + '_min'] = float(np.min(values))
        for metric_name, values in hitrate_at_pct.items():
            if values:
                metrics[metric_name + '_mean'] = float(np.mean(values))
                metrics[metric_name + '_std'] = float(np.std(values))
                metrics[metric_name + '_max'] = float(np.max(values))
                metrics[metric_name + '_min'] = float(np.min(values))
        # 返回理想情况下的指标
        metrics['precision@all_positives_mean'] = float(np.mean(precision_at_all_pos))
        metrics['precision@all_positives_std'] = float(np.std(precision_at_all_pos))
        metrics['precision@all_positives_max'] = float(np.max(precision_at_all_pos))
        metrics['precision@all_positives_min'] = float(np.min(precision_at_all_pos))
        metrics['recall@all_positives_mean'] = float(np.mean(recall_at_all_pos))
        metrics['recall@all_positives_std'] = float(np.std(recall_at_all_pos))
        metrics['recall@all_positives_max'] = float(np.max(recall_at_all_pos))
        metrics['recall@all_positives_min'] = float(np.min(recall_at_all_pos))
        return metrics
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=5, 
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_recall@5pct_mean',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    data_path = "data/adaptive_ltr_samples.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = LTRDataset(data)
    print(dataset.features_raw[0])
