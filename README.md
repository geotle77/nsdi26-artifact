# Fault Prediction Model - AUC Evaluation Based on Historical Fault Analysis

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Project Overview

This project implements a fault prediction model based on historical fault analysis, predicting fault risks by analyzing host fault history. The project uses data-driven machine learning methods to extract features from raw fault data, build training datasets, and train prediction models to assess host fault probability.

## Project Workflow (Open Source Version)

```
Processed Dataset → Dataset Generation → Model Training → Performance Evaluation
       ↓                    ↓                ↓                    ↓
 data/host_fault_detail.json  generate_optimized_ltr.py  deep_ltr_trainer.py  AUC Evaluation
```

> **Note**: This open source version only includes the "Dataset Construction and Training/Evaluation" part. Data acquisition, raw data processing, and online services are not included in the open source scope. Users are expected to have processed data (such as data/host_fault_detail.json and data/normal_hosts.json).

## Core Components

### 1. Dataset Generation Module (`generate_optimized_ltr.py`)
- Builds training datasets based on processed fault history information
- Applies Learning to Rank (LTR) techniques to optimize data organization
- Generates feature matrices and labels for model training

### 2. Model Definition Module (`deep_ltr_model.py`)
- Defines deep learning LTR model architecture
- Implements PyTorch Lightning-based neural networks
- Supports multiple loss functions (listwise, pairwise, pointwise)

### 3. Model Training Module (`deep_ltr_trainer.py`)
- Trains fault prediction models using generated datasets
- Supports multiple machine learning algorithms
- Performs model parameter tuning and validation
- Includes comprehensive evaluation metrics

## Project Structure

```
.
├── generate_optimized_ltr.py     # Dataset generation module
├── deep_ltr_model.py             # Model definition module
├── deep_ltr_trainer.py           # Model training module
├── dataset/                      # Dataset-related modules
│   ├── __init__.py
│   └── Sample.py
├── data/                         # Data directory (users need to place processed data)
│   ├── host_fault_detail.json
│   └── normal_hosts.json
├── examples/                     # Usage examples and tutorials
├── docs/                         # Documentation
├── ltr_config.json               # LTR configuration
├── deep_ltr_config.json          # Deep learning configuration
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
├── LICENSE                       # MIT License
└── README.md                     # Project documentation
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda package manager

### Quick Start

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/fault-prediction.git
cd fault-prediction
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install the package in development mode:**
```bash
pip install -e .
```

### Alternative Installation Methods

**Using conda:**
```bash
conda create -n fault-prediction python=3.11
conda activate fault-prediction
pip install -r requirements.txt
```

**Using Docker:**
```bash
docker build -t fault-prediction .
docker run -it fault-prediction
```

## Usage

### 1. Prepare Data

Place your processed fault data in the `data/` directory:
- `data/host_fault_detail.json`: Host fault history data
- `data/normal_hosts.json`: List of normal hosts (optional)

### 2. Generate Training Dataset

```bash
# Generate balanced LTR dataset
python generate_optimized_ltr.py --method balanced --fault-data data/host_fault_detail.json

# Generate adaptive LTR dataset
python generate_optimized_ltr.py --method adaptive --fault-data data/host_fault_detail.json

# Generate both datasets
python generate_optimized_ltr.py --method both --fault-data data/host_fault_detail.json
```

### 3. Train Prediction Model

```bash
# Train with balanced dataset
python deep_ltr_trainer.py --data data/balanced_ltr_samples.pkl

# Train with adaptive dataset
python deep_ltr_trainer.py --data data/adaptive_ltr_samples.pkl

# Use custom configuration
python deep_ltr_trainer.py --data data/balanced_ltr_samples.pkl --config custom_config.json
```

### 4. Complete Workflow Example

```bash
# Step 1: Generate datasets
python generate_optimized_ltr.py --method both --fault-data data/host_fault_detail.json

# Step 2: Train models
python deep_ltr_trainer.py --data data/balanced_ltr_samples.pkl
python deep_ltr_trainer.py --data data/adaptive_ltr_samples.pkl

# Step 3: Evaluate results
# Check the generated logs and checkpoints directories
```

### 5. Configuration

The project uses JSON configuration files:

- `ltr_config.json`: Dataset generation configuration
- `deep_ltr_config.json`: Model training configuration

Example configuration:
```json
{
  "model_params": {
    "hidden_dims": [512, 256, 128],
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "loss_type": "listwise"
  },
  "training_params": {
    "max_epochs": 100,
    "batch_size": 256,
    "patience": 10
  }
}
```

## Data Pipeline

### Stage 1: Dataset Construction
- `generate_optimized_ltr.py` converts processed data to machine learning format
- Applies feature engineering to build training samples
- Generates training, validation, and test sets
- Supports two strategies:
  - **Balanced**: Time-based sampling with balanced positive/negative ratios
  - **Adaptive**: Density-based sampling with adaptive time windows

### Stage 2: Model Training
- `deep_ltr_trainer.py` trains prediction models using generated datasets
- Supports deep learning algorithms with PyTorch Lightning
- Performs cross-validation and hyperparameter tuning
- Includes comprehensive evaluation metrics (NDCG, AUC, MAP)

### Stage 3: Model Evaluation
- Generates detailed ranking results
- Provides performance metrics and visualizations
- Saves trained models for inference

## Output Results

After running the program, dataset files will be generated in the `data/` directory, and training results will be generated in `checkpoints/` and `logs/`:

### 1. Generated Datasets
- `balanced_ltr_samples.pkl`: Balanced LTR dataset
- `adaptive_ltr_samples.pkl`: Adaptive LTR dataset

### 2. Training Results
- `checkpoints/`: Model checkpoints and best models
- `logs/`: Training logs and TensorBoard files
- `model/`: Saved models for inference
- `test_results/`: Detailed evaluation results

### 3. Performance Metrics
- NDCG@5, NDCG@10, NDCG@20, NDCG@30
- AUC (Area Under Curve)
- MAP (Mean Average Precision)
- Training and validation loss curves

## Model Features

- **Data-Driven**: Trained on real fault history data
- **Rich Features**: Considers fault frequency, type, time, and other dimensions
- **Extensible**: Supports adding new features and algorithms
- **Comprehensive Evaluation**: Uses AUC, NDCG, MAP and other metrics
- **Learning to Rank**: Optimized for ranking tasks
- **Deep Learning**: Neural network-based architecture
- **Time Series**: Handles temporal patterns in fault data

## Extensibility

The project is designed with good extensibility:

### 1. New Features
- Add new feature extraction logic in `dataset/Sample.py`
- Extend `build_features_for_host_faults_with_hash()` function
- Support custom feature engineering pipelines

### 2. New Algorithms
- Integrate new machine learning algorithms in `deep_ltr_trainer.py`
- Add custom loss functions in `deep_ltr_model.py`
- Support ensemble methods and model stacking

### 3. New Evaluation Metrics
- Add more model performance evaluation methods
- Extend `compute_ranking_metrics()` function
- Support custom evaluation criteria

### 4. New Data Sources
- Support different data formats
- Add data preprocessing pipelines
- Integrate with external data sources

## Important Notes

- Ensure quality and completeness of raw data
- Regularly update fault data to maintain model timeliness
- Adjust feature engineering strategies according to actual business scenarios
- Pay attention to data privacy and security protection
- Monitor model performance and retrain when necessary
- Consider computational resources for large-scale deployments

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

## Support


## Acknowledgments
