#!/usr/bin/env python3
"""
Deep Learning LTR Training and Comparison Script
"""
import os
import pickle
import json
import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from deep_ltr_model import DeepLTRModel, LTRDataset

class DeepLTRTrainer:
    """Deep Learning LTR Trainer"""
    
    def __init__(self, config_path="deep_ltr_config.json"):
        self.config = self.load_config(config_path)
        self.results = {}
        
    def load_config(self, config_path):
        """Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                "model_params": {
                    "hidden_dims": [512, 256, 128],
                    "dropout_rate": 0.3,
                    "learning_rate": 1e-3,
                    "weight_decay": 1e-4,
                    "loss_type": "listwise"
                },
                "training_params": {
                    "max_epochs": 100,
                    "batch_size": 256,
                    "patience": 10,
                    "num_workers": 4
                },
                "data_split": {
                    "test_ratio": 0.2,
                    "validation_ratio": 0.1
                }
            }
    
    def load_data(self, data_path):
        """Load LTR data"""
        print(f"Loading data: {data_path}")
        with open(data_path, "rb") as f:
            ltr_samples = pickle.load(f)
        print(f"Total samples: {len(ltr_samples)}")
        return ltr_samples
    
    def temporal_split(self, ltr_samples, test_ratio=0.1, validation_ratio=0.1):
        """Time-based data splitting"""
        # Extract query time information
        query_info = {}
        for sample in ltr_samples:
            qid = sample.features['query_id']
            if qid not in query_info:
                # Use fault_time as timestamp
                timestamp = sample.fault_time
                query_info[qid] = {
                    'timestamp': timestamp,
                    'samples': []
                }
            query_info[qid]['samples'].append(sample)
        
        # Sort queries by time
        sorted_queries = sorted(query_info.items(), key=lambda x: x[1]['timestamp'])
        
        # Calculate split points
        total_queries = len(sorted_queries)
        test_split_point = int(total_queries * (1 - test_ratio))
        val_split_point = int(total_queries * (1 - test_ratio - validation_ratio))
        
        # Split dataset
        train_samples = []
        val_samples = []
        test_samples = []
        
        for i, (qid, info) in enumerate(sorted_queries):
            if i < val_split_point:
                train_samples.extend(info['samples'])
            elif i < test_split_point:
                val_samples.extend(info['samples'])
            else:
                test_samples.extend(info['samples'])
        
        print(f"Temporal split results:")
        print(f"  Training set: {len(train_samples)} samples")
        print(f"  Validation set: {len(val_samples)} samples")
        print(f"  Test set: {len(test_samples)} samples")
        
        return train_samples, val_samples, test_samples
    
    def random_split(self, ltr_samples, test_ratio=0.2, validation_ratio=0.1):
        """Random data splitting method"""
        import random
        
        # Group samples by query ID
        query_groups = {}
        for sample in ltr_samples:
            qid = sample.features['query_id']
            if qid not in query_groups:
                query_groups[qid] = []
            query_groups[qid].append(sample)
        
        # Get all query IDs
        all_query_ids = list(query_groups.keys())
        
        # Randomly shuffle query IDs
        random.seed(42)
        random.shuffle(all_query_ids)
        
        # Calculate split points
        total_queries = len(all_query_ids)
        test_split_point = int(total_queries * (1 - test_ratio))
        val_split_point = int(total_queries * (1 - test_ratio - validation_ratio))
        
        # Split dataset
        train_samples = []
        val_samples = []
        test_samples = []
        
        for i, qid in enumerate(all_query_ids):
            if i < val_split_point:
                train_samples.extend(query_groups[qid])
            elif i < test_split_point:
                val_samples.extend(query_groups[qid])
            else:
                test_samples.extend(query_groups[qid])
        
        print(f"Random split results:")
        print(f"  Training set: {len(train_samples)} samples")
        print(f"  Validation set: {len(val_samples)} samples")
        print(f"  Test set: {len(test_samples)} samples")
        
        return train_samples, val_samples, test_samples
    
    def create_data_loaders(self, train_samples, val_samples, test_samples):
        """Create data loaders"""
        # Create training set
        train_dataset = LTRDataset(train_samples, is_training=True)
        
        # Create validation and test sets (using training set's vectorizer and scaler)
        val_dataset = LTRDataset(
            val_samples, 
            vectorizer=train_dataset.vectorizer,
            scaler=train_dataset.scaler,
            is_training=False
        )
        
        test_dataset = LTRDataset(
            test_samples,
            vectorizer=train_dataset.vectorizer,
            scaler=train_dataset.scaler,
            is_training=False
        )
        
        # Create data loaders
        batch_size = self.config["training_params"]["batch_size"]
        num_workers = self.config["training_params"]["num_workers"]
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False
        )
        
        return train_loader, val_loader, test_loader, train_dataset.features.shape[1]
    
    def train_model(self, data_path):
        """Train model"""
        print(f"\n🚀 Starting model training")
        print("="*50)
        
        # Load data
        ltr_samples = self.load_data(data_path)
        
        # Time-based splitting
        train_samples, val_samples, test_samples = self.random_split(ltr_samples)
        
        # Create data loaders
        train_loader, val_loader, test_loader, input_dim = self.create_data_loaders(
            train_samples, val_samples, test_samples
        )
        
        # Create model
        model = DeepLTRModel(
            input_dim=input_dim,
            **self.config["model_params"]
        )
        
        # Set up callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints",
            filename=f"{{epoch:02d}}-{{val_ndcg@10_mean:.4f}}",
            monitor='val_ndcg@5_mean',
            mode='max',
            save_top_k=1,
        )
        
        early_stopping = EarlyStopping(
            monitor='val_ndcg@5_mean',
            patience=self.config["training_params"]["patience"],
            mode='max'
        )
        
        # Set up logger
        logger = TensorBoardLogger(
            save_dir="logs",
            name=f"deep_ltr",
            version=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.config["training_params"]["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping],
            logger=logger,
            accelerator='auto',
            devices='auto',
            precision=16,  # Use mixed precision training
            gradient_clip_val=1.0,
            log_every_n_steps=50
        )
        
        # Train model
        print("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        
        # Test model
        print("Starting testing...")
        test_results = trainer.test(model, test_loader)
        
        # Save results
        self.results = {
            'test_metrics': test_results[0] if test_results else {},
            'best_model_path': checkpoint_callback.best_model_path,
            'train_size': len(train_samples),
            'val_size': len(val_samples),
            'test_size': len(test_samples)
        }
        
        print(f"✅ Model training completed")
         # After training, save model for inference
        print("\n💾 Saving model for online inference...")
        save_paths = self.save_model_for_inference(
            model=model, 
            train_dataset=train_loader.dataset,
            model_dir="model"
        )
        
        print(f"✅ Model saving completed, ready for online inference")
        print(f"   Model file: {save_paths['model_path']}")
        print(f"   Metadata file: {save_paths['metadata_path']}")
        return model, test_results
    
    def save_model_for_inference(self, model, train_dataset, model_dir="model"):
        """
        Save model for online inference
        
        Args:
            model: Trained model
            train_dataset: Training dataset (contains vectorizer and scaler)
            model_dir: Model save directory
        """
        import os
        import json
        import pickle
        from datetime import datetime
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. Save model weights (.pth format)
        model_path = os.path.join(model_dir, "ltr_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model weights saved: {model_path}")
        
        # 2. Save feature processors
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(train_dataset.vectorizer, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_dataset.scaler, f)
        
        print(f"Feature processors saved: {vectorizer_path}, {scaler_path}")
        
        # 3. Save model metadata
        metadata = {
            # Model architecture parameters
            'input_dim': train_dataset.features.shape[1],
            'hidden_dims': model.hidden_dims,
            'dropout_rate': model.dropout_rate,
            'learning_rate': model.learning_rate,
            'weight_decay': model.weight_decay,
            'loss_type': model.loss_type,
            
            # Feature processor paths
            'vectorizer_path': vectorizer_path,
            'scaler_path': scaler_path,
            
            # Training information
            'training_date': datetime.now().isoformat(),
            'feature_version': 'v1.0',
            'model_version': '1.0.0',
            
            # Training statistics
            'train_size': len(train_dataset.samples),
            'feature_names': list(train_dataset.vectorizer.get_feature_names_out()),
            
            # Model performance metrics
            'best_val_ndcg@5': self.results.get('best_val_ndcg@5', 0.0),
            'test_ndcg@5': self.results.get('test_metrics', {}).get('test_ndcg@5_mean', 0.0),
            'test_auc': self.results.get('test_metrics', {}).get('test_auc_mean', 0.0),
        }
        
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"Model metadata saved: {metadata_path}")
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'vectorizer_path': vectorizer_path,
            'scaler_path': scaler_path
        }


def main():
    parser = argparse.ArgumentParser(description="Deep Learning LTR Training and Comparison")
    parser.add_argument("--config", default="deep_ltr_config.json", help="Configuration file path")
    parser.add_argument("--data",default="data/adaptive_ltr_samples.pkl", 
                       help="Data file path")
    
    args = parser.parse_args()
    
    # Create trainer and start comparison
    trainer = DeepLTRTrainer(args.config)
    trainer.train_model(args.data)

if __name__ == "__main__":
    main()
