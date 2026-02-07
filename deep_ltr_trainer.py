#!/usr/bin/env python3
import pickle
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

from deep_ltr_model import DeepLTRModel, LTRDataset, QueryBatchSampler
import yaml
from GenSample import HostSample

class SamplerEpochCallback(Callback):
    """Callback to set epoch for QueryBatchSampler"""
    def __init__(self, sampler):
        self.sampler = sampler
    
    def on_train_epoch_start(self, trainer, pl_module):
        """在每个训练epoch开始时设置sampler的epoch"""
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(trainer.current_epoch)


class DeepLTRTrainer:
    """深度学习LTR训练器"""
    
    def __init__(self, config_path="config/train.yaml"):
        self.config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        self.results = {}
    
    def load_data(self, data_path):
        """加载LTR数据"""
        print(f"加载数据: {data_path}")
        with open(data_path, "rb") as f:
            ltr_samples = pickle.load(f)
        print(f"总样本数: {len(ltr_samples)}")
        return ltr_samples
    
    def temporal_split(self, ltr_samples, test_ratio=0.2, validation_ratio=0.1):
        """基于时间的数据划分"""
        # 提取查询时间信息
        query_info = {}
        for sample in ltr_samples:
            qid = sample.features['query_id']
            if qid not in query_info:
                # 使用fault_time作为时间戳
                timestamp = sample.fault_time
                query_info[qid] = {
                    'timestamp': timestamp,
                    'samples': []
                }
            query_info[qid]['samples'].append(sample)
        
        # 按时间排序查询
        sorted_queries = sorted(query_info.items(), key=lambda x: x[1]['timestamp'])
        
        # 计算划分点
        total_queries = len(sorted_queries)
        test_split_point = int(total_queries * (1 - test_ratio))
        val_split_point = int(total_queries * (1 - test_ratio - validation_ratio))
        
        # 划分数据集
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
        
        print(f"时间划分结果:")
        print(f"  训练集: {len(train_samples)} 样本")
        print(f"  验证集: {len(val_samples)} 样本")
        print(f"  测试集: {len(test_samples)} 样本")
        
        return train_samples, val_samples, test_samples
    
    def create_data_loaders(self, train_samples, val_samples, test_samples):
        """创建数据加载器"""
        # 创建训练集
        train_dataset = LTRDataset(train_samples, is_training=True)
        
        # 创建验证集和测试集（使用训练集的处理器）
        val_dataset = LTRDataset(
            val_samples, 
            vectorizer=train_dataset.vectorizer,
            scaler=train_dataset.scaler,
            imputer=train_dataset.imputer,
            is_training=False
        )
        
        test_dataset = LTRDataset(
            test_samples,
            vectorizer=train_dataset.vectorizer,
            scaler=train_dataset.scaler,
            imputer=train_dataset.imputer,
            is_training=False
        )
        
        queries_per_batch = self.config["training"].get("queries_per_batch", 5)  # 每个batch的query数量
        num_workers = self.config["training"]["num_workers"]
        is_gpu = torch.cuda.is_available()

        # 使用QueryBatchSampler按query分组（训练集、验证集、测试集都需要）
        train_sampler = QueryBatchSampler(
            train_dataset,
            queries_per_batch=queries_per_batch,
            shuffle=True,
            seed=42
        )
        
        val_sampler = QueryBatchSampler(
            val_dataset,
            queries_per_batch=queries_per_batch,
            shuffle=False,  # 验证集不需要shuffle
            seed=42
        )
        
        test_sampler = QueryBatchSampler(
            test_dataset,
            queries_per_batch=queries_per_batch,
            shuffle=False,  # 测试集不需要shuffle
            seed=42
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,  
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=is_gpu
        )

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,  # 验证集也使用QueryBatchSampler
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=is_gpu
        )

        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,  # 测试集也使用QueryBatchSampler
            num_workers=num_workers,
            persistent_workers=True if num_workers > 0 else False,
            pin_memory=is_gpu
        )

        return train_loader, val_loader, test_loader, train_dataset.features.shape[1], train_sampler
    
    def train_model(self, data_path):
        """训练模型"""
        print(f"\n🚀 开始训练模型")
        print("="*50)
        
        # 加载数据
        ltr_samples = self.load_data(data_path)
        
        # 时间划分
        train_samples, val_samples, test_samples = self.temporal_split(ltr_samples)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader, input_dim, train_sampler = self.create_data_loaders(
            train_samples, val_samples, test_samples
        )

        model = DeepLTRModel(
            input_dim=input_dim,
            **self.config["model"]
        )

        # 设置回调函数
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints",
            filename=f"{{epoch:02d}}-{{val_ndcg@5_mean:.4f}}",
            monitor='val_ndcg@5_mean',
            mode='max',
            save_top_k=1,
        )
        
        early_stopping = EarlyStopping(
            monitor='val_ndcg@5_mean',
            patience=self.config["training"]["patience"],
            mode='max'
        )
        
        # 添加sampler epoch callback
        sampler_callback = SamplerEpochCallback(train_sampler)
        
        # 设置日志记录器
        logger = TensorBoardLogger(
            save_dir="logs",
            name=f"deep_ltr",
            version=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=self.config["training"]["max_epochs"],
            callbacks=[checkpoint_callback, early_stopping, sampler_callback],
            logger=logger,
            accelerator='auto',
            devices='auto',
            log_every_n_steps=50
        )
        
        # 训练模型
        print("开始训练...")
        trainer.fit(model, train_loader, val_loader)
        
        # 测试模型
        print("开始测试...")
        test_results = trainer.test(model, test_loader)
        
        # 保存结果
        self.results = {
            'test_metrics': test_results[0] if test_results else {},
            'best_model_path': checkpoint_callback.best_model_path,
            'train_size': len(train_samples),
            'val_size': len(val_samples),
            'test_size': len(test_samples)
        }
        
        print(f"✅ 模型训练完成")
         # 训练完成后，保存模型用于推理
        print("\n💾 保存模型用于在线推理...")
        save_paths = self.save_model_for_inference(
            model=model, 
            train_dataset=train_loader.dataset,
            model_dir="model"
        )
        
        print(f"✅ 模型保存完成，可用于在线推理")
        print(f"   模型文件: {save_paths['model_path']}")
        print(f"   元数据文件: {save_paths['metadata_path']}")
        return model, test_results
    
    def save_model_for_inference(self, model, train_dataset, model_dir="model"):
        """
        保存模型用于在线推理
        
        Args:
            model: 训练好的模型
            train_dataset: 训练数据集（包含vectorizer和scaler）
            model_dir: 模型保存目录
        """
        import os
        import json
        import pickle
        from datetime import datetime
        
        # 创建模型目录
        os.makedirs(model_dir, exist_ok=True)
        
        # 1. 保存模型权重 (.pth格式)
        model_path = os.path.join(model_dir, "ltr_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存: {model_path}")
        
        # 2. 保存特征处理器
        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        imputer_path = os.path.join(model_dir, "imputer.pkl")
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(train_dataset.vectorizer, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(train_dataset.scaler, f)

        with open(imputer_path, 'wb') as f:
            pickle.dump(train_dataset.imputer, f)
        
        print(f"特征处理器已保存: {vectorizer_path}, {scaler_path}, {imputer_path}")
        
        # 3. 保存模型元数据
        metadata = {
            # 模型架构参数
            'input_dim': train_dataset.features.shape[1],
            'hidden_dims': model.hidden_dims,
            'dropout_rate': model.dropout_rate,
            'learning_rate': model.learning_rate,
            'weight_decay': model.weight_decay,
            'loss_type': model.loss_type,
            
            # 特征处理器路径
            'vectorizer_path': vectorizer_path,
            'scaler_path': scaler_path,
            'imputer_path': imputer_path,
            
            # 训练信息
            'training_date': datetime.now().isoformat(),
            'feature_version': 'v1.0',
            'model_version': '1.0.0',
            
            # 训练统计
            'train_size': len(train_dataset.samples),
            'feature_names': list(train_dataset.vectorizer.get_feature_names_out()),
            
            # 模型性能指标
            'best_val_ndcg@5': self.results.get('best_val_ndcg@5', 0.0),
            'test_ndcg@5': self.results.get('test_metrics', {}).get('test_ndcg@5_mean', 0.0),
            'test_auc': self.results.get('test_metrics', {}).get('test_auc_mean', 0.0),
        }
        
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"模型元数据已保存: {metadata_path}")
        
        return {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'vectorizer_path': vectorizer_path,
            'scaler_path': scaler_path
        }


def main():
    parser = argparse.ArgumentParser(description="深度学习LTR训练和对比")
    parser.add_argument("--config", default="./train.yaml", help="配置文件路径")
    parser.add_argument("--data",default="./data/ltr_samples.pkl", 
                       help="数据文件路径")
    
    args = parser.parse_args()
    
    # 创建训练器并开始对比
    trainer = DeepLTRTrainer(config_path=args.config)
    trainer.train_model(args.data)

if __name__ == "__main__":
    main()
