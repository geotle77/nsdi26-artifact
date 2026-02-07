#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from datetime import datetime

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .model import DeepLTRModel, LTRDataset, QueryBatchSampler


class SamplerEpochCallback(Callback):
    def __init__(self, sampler):
        self.sampler = sampler

    def on_train_epoch_start(self, trainer, pl_module):
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(trainer.current_epoch)


class DeepLTRTrainer:
    def __init__(self, config_path="configs/train.yaml"):
        with open(config_path, "r", encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file)
        self.results = {}

    def load_data(self, data_path):
        print(f"Loading dataset: {data_path}")
        with open(data_path, "rb") as file_obj:
            ltr_samples = pickle.load(file_obj)
        print(f"Total samples: {len(ltr_samples)}")
        return ltr_samples

    def temporal_split(self, ltr_samples, test_ratio=0.2, validation_ratio=0.1):
        query_info = {}
        for sample in ltr_samples:
            query_id = sample.features["query_id"]
            if query_id not in query_info:
                query_info[query_id] = {
                    "timestamp": sample.fault_time,
                    "samples": [],
                }
            query_info[query_id]["samples"].append(sample)

        sorted_queries = sorted(query_info.items(), key=lambda item: item[1]["timestamp"])

        total_queries = len(sorted_queries)
        test_split_point = int(total_queries * (1 - test_ratio))
        validation_split_point = int(total_queries * (1 - test_ratio - validation_ratio))

        train_samples = []
        validation_samples = []
        test_samples = []

        for index, (_, query_data) in enumerate(sorted_queries):
            if index < validation_split_point:
                train_samples.extend(query_data["samples"])
            elif index < test_split_point:
                validation_samples.extend(query_data["samples"])
            else:
                test_samples.extend(query_data["samples"])

        print("Temporal split summary:")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Validation samples: {len(validation_samples)}")
        print(f"  Test samples: {len(test_samples)}")

        return train_samples, validation_samples, test_samples

    def create_data_loaders(self, train_samples, validation_samples, test_samples):
        train_dataset = LTRDataset(train_samples, is_training=True)

        validation_dataset = LTRDataset(
            validation_samples,
            vectorizer=train_dataset.vectorizer,
            scaler=train_dataset.scaler,
            imputer=train_dataset.imputer,
            is_training=False,
        )

        test_dataset = LTRDataset(
            test_samples,
            vectorizer=train_dataset.vectorizer,
            scaler=train_dataset.scaler,
            imputer=train_dataset.imputer,
            is_training=False,
        )

        queries_per_batch = self.config["training"].get("queries_per_batch", 5)
        num_workers = self.config["training"].get("num_workers", 0)
        pin_memory = torch.cuda.is_available()

        train_sampler = QueryBatchSampler(
            train_dataset,
            queries_per_batch=queries_per_batch,
            shuffle=True,
            seed=42,
        )
        validation_sampler = QueryBatchSampler(
            validation_dataset,
            queries_per_batch=queries_per_batch,
            shuffle=False,
            seed=42,
        )
        test_sampler = QueryBatchSampler(
            test_dataset,
            queries_per_batch=queries_per_batch,
            shuffle=False,
            seed=42,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_sampler=validation_sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=pin_memory,
        )

        return train_loader, validation_loader, test_loader, train_dataset.features.shape[1], train_sampler

    def train_model(self, data_path):
        print("Starting training pipeline")
        print("=" * 50)

        ltr_samples = self.load_data(data_path)
        train_samples, validation_samples, test_samples = self.temporal_split(ltr_samples)

        train_loader, validation_loader, test_loader, input_dim, train_sampler = self.create_data_loaders(
            train_samples,
            validation_samples,
            test_samples,
        )

        model = DeepLTRModel(input_dim=input_dim, **self.config["model"])

        monitor_metric = self.config["training"].get("monitor_metric", "val_recall@5pct_mean")
        monitor_mode = self.config["training"].get("monitor_mode", "max")

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints",
            filename="best-{epoch:02d}",
            monitor=monitor_metric,
            mode=monitor_mode,
            save_top_k=1,
            auto_insert_metric_name=False,
        )
        early_stopping = EarlyStopping(
            monitor=monitor_metric,
            patience=self.config["training"].get("patience", 10),
            mode=monitor_mode,
        )
        sampler_callback = SamplerEpochCallback(train_sampler)

        logger = TensorBoardLogger(
            save_dir="logs",
            name="deep_ltr",
            version=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

        trainer = pl.Trainer(
            max_epochs=self.config["training"].get("max_epochs", 100),
            callbacks=[checkpoint_callback, early_stopping, sampler_callback],
            logger=logger,
            accelerator="auto",
            devices="auto",
            log_every_n_steps=50,
        )

        print("Training model")
        trainer.fit(model, train_loader, validation_loader)

        print("Running test evaluation")
        test_results = trainer.test(model, test_loader)

        self.results = {
            "test_metrics": test_results[0] if test_results else {},
            "best_model_path": checkpoint_callback.best_model_path,
            "train_size": len(train_samples),
            "validation_size": len(validation_samples),
            "test_size": len(test_samples),
        }

        print("Training completed")
        print("Saving inference artifacts")
        save_paths = self.save_model_for_inference(
            model=model,
            train_dataset=train_loader.dataset,
            model_dir="model",
        )

        print("Artifacts saved")
        print(f"  Model file: {save_paths['model_path']}")
        print(f"  Metadata file: {save_paths['metadata_path']}")
        return model, test_results

    def save_model_for_inference(self, model, train_dataset, model_dir="model"):
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "ltr_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model weights: {model_path}")

        vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        imputer_path = os.path.join(model_dir, "imputer.pkl")

        with open(vectorizer_path, "wb") as file_obj:
            pickle.dump(train_dataset.vectorizer, file_obj)
        with open(scaler_path, "wb") as file_obj:
            pickle.dump(train_dataset.scaler, file_obj)
        with open(imputer_path, "wb") as file_obj:
            pickle.dump(train_dataset.imputer, file_obj)

        print(f"Saved preprocessors: {vectorizer_path}, {scaler_path}, {imputer_path}")

        test_metrics = self.results.get("test_metrics", {})
        metadata = {
            "input_dim": train_dataset.features.shape[1],
            "hidden_dims": model.hidden_dims,
            "dropout_rate": model.dropout_rate,
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "loss_type": model.loss_type,
            "vectorizer_path": vectorizer_path,
            "scaler_path": scaler_path,
            "imputer_path": imputer_path,
            "training_date": datetime.now().isoformat(),
            "feature_version": "v1.0",
            "model_version": "1.0.0",
            "train_size": len(train_dataset.samples),
            "feature_names": list(train_dataset.vectorizer.get_feature_names_out()),
            "best_val_ndcg@5": test_metrics.get("val_ndcg@5_mean", 0.0),
            "test_ndcg@5": test_metrics.get("test_ndcg@5_mean", 0.0),
            "test_auc": test_metrics.get("test_mean_auc", 0.0),
        }

        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

        print(f"Saved metadata: {metadata_path}")
        return {
            "model_path": model_path,
            "metadata_path": metadata_path,
            "vectorizer_path": vectorizer_path,
            "scaler_path": scaler_path,
            "imputer_path": imputer_path,
        }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate the deep LTR model")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to YAML training config")
    parser.add_argument("--data", default="data/ltr_samples.pkl", help="Path to generated LTR dataset")
    args = parser.parse_args()

    trainer = DeepLTRTrainer(config_path=args.config)
    trainer.train_model(args.data)


if __name__ == "__main__":
    main()
