#!/usr/bin/env python3
"""
Main script for generating LTR datasets using optimization strategies
"""

import json
import argparse
import os
from datetime import datetime
from dataset.Sample import generate_balanced_ltr_dataset, generate_adaptive_ltr_dataset

def load_config(config_path="ltr_config.json"):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_with_config(faults_data, config, method="balanced"):
    """Generate LTR dataset according to configuration"""
    
    if method == "balanced":
        cfg = config["balanced_ltr_config"]
        output_path = os.path.join(config["data_paths"]["output_dir"], "balanced_ltr_samples.pkl")
        
        print(f"Generating LTR dataset using balanced method...")
        print(f"Configuration: min_positive_ratio={cfg['min_positive_ratio']}, "
              f"max_positive_ratio={cfg['max_positive_ratio']}, "
              f"target_positive_ratio={cfg['target_positive_ratio']}")
        
        return generate_balanced_ltr_dataset(
            faults_data,
            output_path,
            min_positive_ratio=cfg["min_positive_ratio"],
            max_positive_ratio=cfg["max_positive_ratio"],
            target_positive_ratio=cfg["target_positive_ratio"]
        )
    
    elif method == "adaptive":
        cfg = config["adaptive_ltr_config"]
        output_path = os.path.join(config["data_paths"]["output_dir"], "adaptive_ltr_samples.pkl")
        
        print(f"Generating LTR dataset using adaptive method...")
        print(f"Configuration: window_days={cfg['window_days']}, "
              f"min_positive_ratio={cfg['min_positive_ratio']}, "
              f"target_positive_ratio={cfg['target_positive_ratio']}")
        
        return generate_adaptive_ltr_dataset(
            faults_data,
            output_path,
            window_days=cfg["window_days"],
            min_positive_ratio=cfg["min_positive_ratio"],
            target_positive_ratio=cfg["target_positive_ratio"]
        )
    
    else:
        raise ValueError(f"Unsupported method: {method}")

def evaluate_dataset_quality(samples):
    """Evaluate dataset quality"""
    from collections import defaultdict
    import numpy as np
    
    # Statistics by query
    query_stats = defaultdict(lambda: [0, 0])  # [positive_samples, total_samples]
    
    for sample in samples:
        query_id = sample.features['query_id']
        query_stats[query_id][1] += 1
        if sample.label == 1:
            query_stats[query_id][0] += 1
    
    # Calculate quality metrics
    ratios = [pos/total for pos, total in query_stats.values() if total > 0]
    
    quality_metrics = {
        "total_samples": len(samples),
        "total_queries": len(query_stats),
        "valid_queries": sum(1 for r in ratios if r >= 0.01),
        "ratio_stats": {
            "min": min(ratios) if ratios else 0,
            "max": max(ratios) if ratios else 0,
            "mean": np.mean(ratios) if ratios else 0,
            "std": np.std(ratios) if ratios else 0,
            "median": np.median(ratios) if ratios else 0
        },
        "balance_score": 1 - np.std(ratios) if ratios else 0,  # Balance score
        "coverage_score": len(query_stats) / 70 if len(query_stats) <= 70 else 1.0  # Time coverage score
    }
    
    return quality_metrics

def print_quality_report(metrics, method_name):
    """Print quality report"""
    print(f"\n=== {method_name} Dataset Quality Report ===")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Total queries: {metrics['total_queries']}")
    print(f"Valid queries (ratio >= 1%): {metrics['valid_queries']}")
    print(f"Query validity rate: {metrics['valid_queries']/metrics['total_queries']*100:.1f}%")
    
    print(f"\nPositive sample ratio statistics:")
    print(f"  Min: {metrics['ratio_stats']['min']:.4f}")
    print(f"  Max: {metrics['ratio_stats']['max']:.4f}")
    print(f"  Mean: {metrics['ratio_stats']['mean']:.4f}")
    print(f"  Std: {metrics['ratio_stats']['std']:.4f}")
    print(f"  Median: {metrics['ratio_stats']['median']:.4f}")
    
    print(f"\nQuality scores:")
    print(f"  Balance score: {metrics['balance_score']:.3f}")
    print(f"  Time coverage score: {metrics['coverage_score']:.3f}")
    
    # Quality level assessment
    overall_score = (metrics['balance_score'] + metrics['coverage_score']) / 2
    if overall_score >= 0.8:
        quality_level = "Excellent"
    elif overall_score >= 0.6:
        quality_level = "Good"
    elif overall_score >= 0.4:
        quality_level = "Fair"
    else:
        quality_level = "Poor"
    
    print(f"  Overall quality level: {quality_level} ({overall_score:.3f})")

def main():
    parser = argparse.ArgumentParser(description="Generate optimized LTR dataset")
    parser.add_argument("--config", default="ltr_config.json", help="Configuration file path")
    parser.add_argument("--method", choices=["balanced", "adaptive", "both"], 
                       default="both", help="Generation method")
    parser.add_argument("--fault-data", help="Fault data file path (overrides config file)")
    parser.add_argument("--evaluate", default=True,action="store_true", help="Evaluate generated dataset quality")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load fault data
    fault_data_path = args.fault_data or config["data_paths"]["fault_data"]
    print(f"Loading fault data: {fault_data_path}")
    
    with open(fault_data_path, 'r', encoding="utf-8") as f:
        faults_data = json.load(f)
    
    print(f"Fault data contains {len(faults_data)} hosts")
    
    # Ensure output directory exists
    os.makedirs(config["data_paths"]["output_dir"], exist_ok=True)
    
    # Generate dataset
    results = {}
    
    if args.method in ["balanced", "both"]:
        print("\n" + "="*50)
        print("Generating balanced LTR dataset")
        print("="*50)
        
        balanced_samples = generate_with_config(faults_data, config, "balanced")
        results["balanced"] = balanced_samples
        
        if args.evaluate:
            metrics = evaluate_dataset_quality(balanced_samples)
            print_quality_report(metrics, "Balanced method")
    
    if args.method in ["adaptive", "both"]:
        print("\n" + "="*50)
        print("Generating adaptive LTR dataset")
        print("="*50)
        
        adaptive_samples = generate_with_config(faults_data, config, "adaptive")
        results["adaptive"] = adaptive_samples
        
        if args.evaluate:
            metrics = evaluate_dataset_quality(adaptive_samples)
            print_quality_report(metrics, "Adaptive method")
    
    # Comparative analysis
    if len(results) > 1 and args.evaluate:
        print("\n" + "="*50)
        print("Method comparison analysis")
        print("="*50)
        
        for method_name, samples in results.items():
            metrics = evaluate_dataset_quality(samples)
            print(f"\n{method_name} method:")
            print(f"  Sample count: {metrics['total_samples']}")
            print(f"  Query count: {metrics['total_queries']}")
            print(f"  Valid query rate: {metrics['valid_queries']/metrics['total_queries']*100:.1f}%")
            print(f"  Average positive ratio: {metrics['ratio_stats']['mean']:.4f}")
            print(f"  Ratio std: {metrics['ratio_stats']['std']:.4f}")
    
    print(f"\nDataset generation completed! Output directory: {config['data_paths']['output_dir']}")

if __name__ == "__main__":
    main()
