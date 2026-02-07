#!/usr/bin/env python3
"""
Core model, dataset, and metrics for the ranking pipeline.
"""

import concurrent.futures
import json
import multiprocessing
import os
import pickle
import random
from datetime import datetime

import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ndcg_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset, Sampler

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
    """Flatten one sample feature dictionary for vectorization."""
    sample, features_to_exclude = sample_data
    flattened = {}
    
    for key, value in sample.features.items():

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
        self.epoch = 0
        

        self.query_to_indices = {}
        for idx, qid in enumerate(dataset.query_ids):
            if qid not in self.query_to_indices:
                self.query_to_indices[qid] = []
            self.query_to_indices[qid].append(idx)
        

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
        
        if use_multiprocessing :
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
        

        self.query_groups = self._compute_query_groups()
    
    def get_feature_indices(self, feature_names):
        """Return vectorized feature indices matching prefix names."""
        feature_names_out = self.vectorizer.get_feature_names_out()
        indices = []
        for name in feature_names:

            matches = [i for i, fname in enumerate(feature_names_out) 
                    if fname.startswith(name) or name in fname]
            indices.extend(matches)
        return list(set(indices))

    def _flatten_features_parallel(self, samples, n_workers):
        """Flatten features with a thread pool."""

        sample_data = [(sample, self.features_to_exclude) for sample in samples]
        

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            features_raw = list(executor.map(flatten_features_worker, sample_data))
        
        return features_raw
    
    def _flatten_features_sequential(self, samples):
        """Flatten features sequentially."""
        return [flatten_features_worker((sample, self.features_to_exclude)) for sample in samples]
  

    def _compute_query_groups(self):
        """Build query to sample-index mapping in O(N)."""
        query_groups = {}
        

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
    """PyTorch Lightning model for listwise/pairwise/pointwise ranking."""
    
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


        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)


        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def predict(self, x):
        """Return probability scores in [0, 1] for each sample."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
    
    def listwise_loss(self, scores, labels, query_groups):
        """Compute ListNet-style listwise loss per query."""
        total_loss = 0
        num_queries = 0
        
        for qid, group_info in query_groups.items():
            indices = group_info['indices']
            
            if len(indices) < 2:
                continue
            
            query_scores = scores[indices]
            query_labels = labels[indices]
            

            score_probs = F.softmax(query_scores, dim=0)
            label_probs = F.softmax(query_labels, dim=0)
            

            loss = F.kl_div(score_probs.log(), label_probs, reduction='sum')
            total_loss += loss
            num_queries += 1
        
        return total_loss / max(num_queries, 1)
    
    def pairwise_loss(self, logits, labels, query_groups):
        """
        Compute LambdaRank-like pairwise loss weighted by Delta-NDCG.
        """

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
            

            pos_mask = (q_labels == 1)
            neg_mask = (q_labels == 0)
            
            pos_idx = torch.where(pos_mask)[0]
            neg_idx = torch.where(neg_mask)[0]
            

            if len(pos_idx) == 0 or len(neg_idx) == 0:
                continue
            

            num_pos = len(pos_idx)

            ideal_ranks = torch.arange(1, num_pos + 1, device=logits.device).float()
            idcg = torch.sum(1.0 / torch.log2(ideal_ranks + 1.0))
            if idcg == 0: continue



            _, sorted_idx = torch.sort(q_scores + torch.randn_like(q_scores) * 1e-6, descending=True)
            ranks = torch.zeros_like(q_scores)
            ranks[sorted_idx] = torch.arange(1, len(q_scores) + 1, device=logits.device).float()



            i_grid, j_grid = torch.meshgrid(pos_idx, neg_idx, indexing='ij')
            i_flat = i_grid.reshape(-1)
            j_flat = j_grid.reshape(-1)
            

            s_i = q_scores[i_flat]
            s_j = q_scores[j_flat]
            diff = s_i - s_j
            


            rank_i = ranks[i_flat]
            rank_j = ranks[j_flat]
            delta_ndcg = torch.abs(1.0 / torch.log2(rank_i + 1.0) - 1.0 / torch.log2(rank_j + 1.0)) / idcg
            


            query_loss = torch.sum(delta_ndcg * F.softplus(-diff))
            
            total_batch_loss += query_loss
            num_queries += 1


        return total_batch_loss / max(num_queries, 1)

    def pointwise_loss(self, scores, labels, query_groups):
        """Compute binary cross-entropy loss per query."""
        total_loss = 0
        num_queries = 0
        
        for qid, group_info in query_groups.items():
            indices = group_info['indices']
            
            if len(indices) < 1:
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


        query_groups = {}
        unique_qids = torch.unique(query_ids)

        for qid in unique_qids:
            mask = query_ids == qid
            indices = torch.where(mask)[0]
            query_groups[qid.item()] = {
                'indices': indices,
                'size': len(indices)
            }


        if self.loss_type == 'pairwise':
            loss = self.pairwise_loss(scores, labels, query_groups)
        elif self.loss_type == 'listwise':
            loss = self.listwise_loss(scores, labels, query_groups)
        elif self.loss_type == 'pointwise':
            loss = self.pointwise_loss(scores, labels, query_groups)
        else:
            loss = F.binary_cross_entropy_with_logits(scores, labels)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['label']
        query_ids = batch['query_id']
        
        scores = self(features)
        

        self.validation_outputs.append({
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'query_ids': query_ids.cpu()
        })
        
        return {'scores': scores, 'labels': labels, 'query_ids': query_ids}
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        

        all_scores = torch.cat([x['scores'] for x in self.validation_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_outputs])
        all_query_ids = torch.cat([x['query_ids'] for x in self.validation_outputs])
        

        metrics = self.compute_ranking_metrics(all_scores, all_labels, all_query_ids)
        

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
        

        all_scores = torch.cat([x['scores'] for x in self.test_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_outputs])
        all_query_ids = torch.cat([x['query_ids'] for x in self.test_outputs])
        

        metrics = self.compute_ranking_metrics(all_scores, all_labels, all_query_ids)
        

        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value)
        

        detailed_results = {}
        unique_qids = torch.unique(all_query_ids)
        
        for qid in unique_qids:
            mask = all_query_ids == qid
            query_scores = all_scores[mask].float().cpu().numpy()
            query_labels = all_labels[mask].float().cpu().numpy()
            

            sorted_indices = np.argsort(-query_scores)
            

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
        
        os.makedirs('test_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = f'test_results/ranking_results_{timestamp}.json'
        
        with open(result_path, 'w') as f:
            json.dump(detailed_results, f, indent=4)
        with open("test_results/metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nSaved ranking details to: {result_path}")
        
        self.test_outputs.clear()
        return metrics
    
    def compute_ranking_metrics(self, scores, labels, query_ids):
        """Compute ranking and retrieval metrics grouped by query."""
        metrics = {}
        

        unique_qids = torch.unique(query_ids)
        ndcg_scores = {'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': [], 'ndcg@30': []}
        auc_scores = []
        for qid in unique_qids:
            mask = query_ids == qid
            query_scores = scores[mask].float().cpu().numpy()
            query_labels = labels[mask].float().cpu().numpy()
            
            if len(set(query_labels)) < 2:
                continue
            

            for k in [5, 10, 20, 30]:
                if len(query_scores) >= k:
                    ndcg_k = ndcg_score([query_labels], [query_scores], k=k)
                    ndcg_scores[f'ndcg@{k}'].append(ndcg_k)

            

            try:
                auc = roc_auc_score(query_labels, query_scores)
                auc_scores.append(auc)
            except Exception:
                pass
        

        pct_list = [0.01, 0.02, 0.05, 0.10]
        precision_at_pct = {f'precision@{int(p*100)}pct': [] for p in pct_list}
        recall_at_pct = {f'recall@{int(p*100)}pct': [] for p in pct_list}
        hitrate_at_pct = {f'hitrate@{int(p*100)}pct': [] for p in pct_list}
        

        precision_at_all_pos = []
        recall_at_all_pos = []

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
                continue
            

            sorted_idx = np.lexsort((np.arange(len(query_scores)), -query_scores))

            sorted_labels = query_labels[sorted_idx]


            for p in pct_list:
                k = max(1, int(np.ceil(p * n)))
                top_idx = sorted_idx[:k]
                pos_in_top = int(np.sum(query_labels[top_idx]))
                

                precision_k = pos_in_top / k
                precision_at_pct[f'precision@{int(p*100)}pct'].append(precision_k)
                

                recall_k = pos_in_top / total_pos
                recall_at_pct[f'recall@{int(p*100)}pct'].append(recall_k)
                

                hitrate_k = 1.0 if pos_in_top > 0 else 0.0
                hitrate_at_pct[f'hitrate@{int(p*100)}pct'].append(hitrate_k)
            


            k_ideal = min(total_pos, n)
            top_k_ideal = sorted_idx[:k_ideal]
            pos_in_top_ideal = int(np.sum(query_labels[top_k_ideal]))
            precision_ideal = pos_in_top_ideal / k_ideal if k_ideal > 0 else 0.0
            precision_at_all_pos.append(precision_ideal)
            


            last_pos_idx = np.where(sorted_labels == 1)[0]
            if len(last_pos_idx) > 0:
                last_pos_rank = last_pos_idx[-1] + 1

                normalized_k = last_pos_rank / n
                recall_at_all_pos.append(normalized_k)


        for metric_name, scores_list in ndcg_scores.items():
            if scores_list:
                score_std = np.std(scores_list)
                metrics[metric_name + '_mean'] = np.mean(scores_list)
                metrics[metric_name + '_std'] = score_std
                metrics[metric_name + '_max'] = np.max(scores_list)
                metrics[metric_name + '_min'] = np.min(scores_list)


        if auc_scores:
            metrics['mean_auc'] = np.mean(auc_scores)
            metrics['std_auc'] = np.std(auc_scores)
            metrics['auc_max'] = np.max(auc_scores)
            metrics['auc_min'] = np.min(auc_scores)


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

        if precision_at_all_pos:
            metrics['precision@all_positives_mean'] = float(np.mean(precision_at_all_pos))
            metrics['precision@all_positives_std'] = float(np.std(precision_at_all_pos))
            metrics['precision@all_positives_max'] = float(np.max(precision_at_all_pos))
            metrics['precision@all_positives_min'] = float(np.min(precision_at_all_pos))
        else:
            metrics['precision@all_positives_mean'] = 0.0
            metrics['precision@all_positives_std'] = 0.0
            metrics['precision@all_positives_max'] = 0.0
            metrics['precision@all_positives_min'] = 0.0

        if recall_at_all_pos:
            metrics['recall@all_positives_mean'] = float(np.mean(recall_at_all_pos))
            metrics['recall@all_positives_std'] = float(np.std(recall_at_all_pos))
            metrics['recall@all_positives_max'] = float(np.max(recall_at_all_pos))
            metrics['recall@all_positives_min'] = float(np.min(recall_at_all_pos))
        else:
            metrics['recall@all_positives_mean'] = 0.0
            metrics['recall@all_positives_std'] = 0.0
            metrics['recall@all_positives_max'] = 0.0
            metrics['recall@all_positives_min'] = 0.0
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
    data_path = "data/ltr_samples.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = LTRDataset(data)
    print(dataset.features_raw[0])
