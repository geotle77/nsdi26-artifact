__version__ = "0.1.0"

__all__ = [
    "HostSample",
    "generate_adaptive_ltr_dataset",
    "DeepLTRModel",
    "LTRDataset",
    "QueryBatchSampler",
    "DeepLTRTrainer",
]


def __getattr__(name):
    if name in {"HostSample", "generate_adaptive_ltr_dataset"}:
        from .generate_dataset import HostSample, generate_adaptive_ltr_dataset
        return {"HostSample": HostSample, "generate_adaptive_ltr_dataset": generate_adaptive_ltr_dataset}[name]
    if name in {"DeepLTRModel", "LTRDataset", "QueryBatchSampler"}:
        from .model import DeepLTRModel, LTRDataset, QueryBatchSampler
        return {"DeepLTRModel": DeepLTRModel, "LTRDataset": LTRDataset, "QueryBatchSampler": QueryBatchSampler}[name]
    if name == "DeepLTRTrainer":
        from .trainer import DeepLTRTrainer
        return DeepLTRTrainer
    raise AttributeError(f"module 'fault_prediction' has no attribute '{name}'")
