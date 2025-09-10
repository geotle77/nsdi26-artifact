#!/usr/bin/env python3
"""
Deep Learning LTR Model based on PyTorch Lightning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score, roc_auc_score, average_precision_score
import pickle
from collections import defaultdict
import seaborn as sns
import concurrent.futures
import multiprocessing
from functools import partial
import time

time_feature = ['is_weekend','month_of_year','week_of_month','day_of_week','day_of_month']
host_feature = ['host_model','host_id','datacenter']
fault_feature = ['fault_level_counts','fault_class_counts','fault_subclass_counts']
frequency_feature =['history_fault_count','days_since_last_fault','fault_frequency',
                    'mean_fault_interval','recent_fault_count','recent_to_total_ratio',"mean_duration"]

def flatten_features_worker(sample_data, features_to_exclude=None):
    """Worker function: flatten features of a single sample"""
    sample, features_to_exclude = sample_data
    flattened = {}
    
    for key, value in sample.features.items():
        # Check if this feature should be excluded (prefix matching)
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


class LTRDataset(Dataset):
    """LTR Dataset class"""
    
    def __init__(self, samples, vectorizer=None, scaler=None, is_training=True, n_workers=None,use_multiprocessing=True):
        self.samples = samples
        self.is_training = is_training
        
        # Extract features and labels (filter during flattening)
        self.labels = torch.tensor([sample.label for sample in samples], dtype=torch.float32)
        self.query_ids = [sample.features['query_id'] for sample in samples]
        for sample in samples:
            del sample.features['query_id']

        self.features_to_exclude = None
        
        # Determine number of threads
        if n_workers is None:
            n_workers = min(multiprocessing.cpu_count(), 8)  # Limit maximum threads
        
        # Create thread pool
        # Multi-threaded feature flattening
        if use_multiprocessing :  # Only use multi-threading for large datasets
            self.features_raw = self._flatten_features_parallel(samples, n_workers)
        else:
            self.features_raw = self._flatten_features_sequential(samples)
        
        # Feature vectorization
        if vectorizer is None:
            self.vectorizer = DictVectorizer(sparse=False)
            features_vectorized = self.vectorizer.fit_transform(self.features_raw)
        else:
            self.vectorizer = vectorizer
            features_vectorized = self.vectorizer.transform(self.features_raw)
        
        # Feature standardization
        if scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features_vectorized)
        else:
            self.scaler = scaler
            features_scaled = self.scaler.transform(features_vectorized)
        
        self.features = torch.tensor(features_scaled, dtype=torch.float32)
        
        # Calculate query group information
        self.query_groups = self._compute_query_groups()
    
        
    def _flatten_features_parallel(self, samples, n_workers):
        """Parallel feature flattening processing"""
        # Prepare data
        sample_data = [(sample, self.features_to_exclude) for sample in samples]
        
        # Execute using thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            features_raw = list(executor.map(flatten_features_worker, sample_data))
        
        return features_raw
    
    def _flatten_features_sequential(self, samples):
        """Sequential feature flattening processing (fallback option)"""
        return [flatten_features_worker((sample, self.features_to_exclude)) for sample in samples]
  

    def _compute_query_groups(self):
        """Calculate query group information"""
        query_groups = {}
        current_idx = 0
        
        for qid in sorted(set(self.query_ids)):
            indices = [i for i, q in enumerate(self.query_ids) if q == qid]
            query_groups[qid] = {
                'start_idx': current_idx,
                'end_idx': current_idx + len(indices),
                'size': len(indices),
                'indices': indices
            }
            current_idx += len(indices)
        
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
    """Deep Learning LTR Model"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.3, 
                 learning_rate=1e-3, weight_decay=1e-4, loss_type='listwise',description=""):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_type = loss_type
        
        # Build MLP network
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # For storing validation and test results
        self.validation_outputs = []
        self.test_outputs = []
        
    def forward(self, x):
        return self.network(x).squeeze(-1)
    
    def listwise_loss(self, scores, labels, query_groups):
        """Listwise损失函数 (ListNet)"""
        total_loss = 0
        num_queries = 0
        
        for qid, group_info in query_groups.items():
            start_idx = group_info['start_idx']
            end_idx = group_info['end_idx']
            
            if end_idx - start_idx < 2:  # jump over queries with only one sample
                continue
            
            query_scores = scores[start_idx:end_idx]
            query_labels = labels[start_idx:end_idx]
            
            # Calculate probability distribution
            score_probs = F.softmax(query_scores, dim=0)
            label_probs = F.softmax(query_labels, dim=0)
            
            # KL divergence loss
            loss = F.kl_div(score_probs.log(), label_probs, reduction='sum')
            total_loss += loss
            num_queries += 1
        
        return total_loss / max(num_queries, 1)
    
    def pairwise_loss(self, scores, labels, query_groups):
        """Pairwise loss function (RankNet)"""
        total_loss = 0
        num_pairs = 0
        
        for qid, group_info in query_groups.items():
            start_idx = group_info['start_idx']
            end_idx = group_info['end_idx']
            
            query_scores = scores[start_idx:end_idx]
            query_labels = labels[start_idx:end_idx]
            
            # Generate all possible pairs
            for i in range(len(query_scores)):
                for j in range(i + 1, len(query_scores)):
                    if query_labels[i] != query_labels[j]:
                        # Ensure positive samples score higher than negative samples
                        if query_labels[i] > query_labels[j]:
                            diff = query_scores[i] - query_scores[j]
                        else:
                            diff = query_scores[j] - query_scores[i]
                        
                        loss = torch.log(1 + torch.exp(-diff))
                        total_loss += loss
                        num_pairs += 1
        
        return total_loss / max(num_pairs, 1)
    def pointwise_loss(self, scores, labels, query_groups):
        """Pointwise loss function (MSE + query normalization)"""
        total_loss = 0
        num_queries = 0
        
        for qid, group_info in query_groups.items():
            start_idx = group_info['start_idx']
            end_idx = group_info['end_idx']
            
            if end_idx - start_idx < 1:  # jump over empty queries
                continue
            
            query_scores = scores[start_idx:end_idx]
            query_labels = labels[start_idx:end_idx]
            
            # Query normalization scores and labels (optional)
            #norm_scores = (query_scores - query_scores.mean()) / (query_scores.std() + 1e-8)
            #norm_labels = (query_labels - query_labels.mean()) / (query_labels.std() + 1e-8)
            
            # Calculate MSE loss
            #mse_loss = F.mse_loss(query_scores, query_labels)
            
            # Add binary cross entropy loss to improve classification ability
            bce_loss = F.binary_cross_entropy_with_logits(query_scores, query_labels)
            
            # Combined loss
            combined_loss = bce_loss
            total_loss += combined_loss
            num_queries += 1
        
        return total_loss / max(num_queries, 1)
    def training_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['label']
        query_ids = batch['query_id']
        
        scores = self(features)
        
        # Build query group information
        query_groups = {}
        unique_qids = torch.unique(query_ids)
        
        for qid in unique_qids:
            mask = query_ids == qid
            indices = torch.where(mask)[0]
            query_groups[qid.item()] = {
                'start_idx': indices[0].item(),
                'end_idx': indices[-1].item() + 1,
                'size': len(indices)
            }
        
        # Calculate loss
        if self.loss_type == 'listwise':
            loss = self.listwise_loss(scores, labels, query_groups)
        elif self.loss_type == 'pairwise':
            loss = self.pairwise_loss(scores, labels, query_groups)
        elif self.loss_type == 'pointwise':
            loss = self.pointwise_loss(scores, labels, query_groups)
        else:  # Default use simple binary cross entropy
            loss = F.binary_cross_entropy_with_logits(scores, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        features = batch['features']
        labels = batch['label']
        query_ids = batch['query_id']
        
        scores = self(features)
        
        # Store results for metrics calculation at epoch end
        self.validation_outputs.append({
            'scores': scores.cpu(),
            'labels': labels.cpu(),
            'query_ids': query_ids.cpu()
        })
        
        return {'scores': scores, 'labels': labels, 'query_ids': query_ids}
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        # Merge all batch results
        all_scores = torch.cat([x['scores'] for x in self.validation_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_outputs])
        all_query_ids = torch.cat([x['query_ids'] for x in self.validation_outputs])
        
        # Calculate metrics
        metrics = self.compute_ranking_metrics(all_scores, all_labels, all_query_ids)
        
        # Record metrics
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
        
        # Merge all batch results
        all_scores = torch.cat([x['scores'] for x in self.test_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_outputs])
        all_query_ids = torch.cat([x['query_ids'] for x in self.test_outputs])
        
        # Calculate metrics
        metrics = self.compute_ranking_metrics(all_scores, all_labels, all_query_ids)
        
        # Record metrics
        for metric_name, value in metrics.items():
            self.log(f'test_{metric_name}', value)
        
        # Save detailed ranking results
        detailed_results = {}
        unique_qids = torch.unique(all_query_ids)
        
        for qid in unique_qids:
            mask = all_query_ids == qid
            query_scores = all_scores[mask].float().cpu().numpy()
            query_labels = all_labels[mask].float().cpu().numpy()
            
            # Get sorted indices (from high to low)
            sorted_indices = np.argsort(-query_scores)
            
            # Save results in sorted order
            sorted_results = []
            for idx in sorted_indices:
                sorted_results.append({
                    'score': float(query_scores[idx]),
                    'label': int(query_labels[idx]),
                    'ranking': int(idx) + 1
                })
            
            # Save sorted results for this query
            detailed_results[qid.item()] = {
                'sorted_items': sorted_results,
                'positive_count': int(np.sum(query_labels)),
                'total_count': len(query_labels)
            }
        
        # Save to file
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
        
        print(f"\nDetailed ranking results saved to: {result_path}")
        
        self.test_outputs.clear()
        return metrics
    
    def compute_ranking_metrics(self, scores, labels, query_ids):
        """Calculate ranking metrics"""
        metrics = {}
        
        # Calculate metrics by query group
        unique_qids = torch.unique(query_ids)
        ndcg_scores = {'ndcg@5': [], 'ndcg@10': [], 'ndcg@20': [], 'ndcg@30': []}
        auc_scores = []
        map_scores = []
        
        for qid in unique_qids:
            mask = query_ids == qid
            query_scores = scores[mask].float().cpu().numpy()
            query_labels = labels[mask].float().cpu().numpy()
            
            if len(set(query_labels)) < 2:  # skip queries with only one label
                continue
            
            # Calculate NDCG@k
            for k in [5, 10, 20, 30]:
                if len(query_scores) >= k:
                    ndcg_k = ndcg_score([query_labels], [query_scores], k=k)
                    ndcg_scores[f'ndcg@{k}'].append(ndcg_k)

            
            # Calculate AUC
            try:
                auc = roc_auc_score(query_labels, query_scores)
                auc_scores.append(auc)
            except:
                pass
            
            # Calculate MAP
            try:
                map_score = average_precision_score(query_labels, query_scores)
                map_scores.append(map_score)
            except:
                pass
        
        # Calculate average metrics
        for metric_name, scores_list in ndcg_scores.items():
            if scores_list:
                score_std = np.std(scores_list)
                metrics[metric_name + '_mean'] = np.mean(scores_list)
                metrics[metric_name + '_std'] = score_std
        
        if auc_scores:
            metrics['mean_auc'] = np.mean(auc_scores)
            metrics['std_auc'] = np.std(auc_scores)
        
        if map_scores:
            metrics['map'] = np.mean(map_scores)
            metrics['std_map'] = np.std(map_scores)
        
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
                'monitor': 'val_ndcg@5_mean',
                'frequency': 1
            }
        }


if __name__ == "__main__":
    data_path = "data/adaptive_ltr_samples.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = LTRDataset(data)
    print(dataset.features_raw[0])