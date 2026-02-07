from typing import List,Dict,Any,Tuple
import json
from datetime import datetime
import random
import pickle
import numpy as np
import re
class HostSample:
    def __init__(self, host: str, fault_time: int, features: Dict[str, Any], label: int):
        self.host = host
        self.fault_time = fault_time  # 故障发生时间
        self.features = features      # 特征字典
        self.label = label            # 标签（如1为故障，0为正常）


def get_all_fault_types(faults_data):
    fault_types = set()
    for faults in faults_data.values():
        for f in faults:
            ft = f.get('FaultType', 'Unknown')
            fault_types.add(ft)
    fault_types = sorted(list(fault_types))  # 固定顺序
    return fault_types

def get_all_fault_levels(faults_data):
    fault_levels = set()
    fault_classes = set()
    fault_subclasses = set()
    for faults in faults_data.values():
        for f in faults:
            level = f.get('Level', 'Unknown')
            fault_levels.add(level)
            fault_classes.add(f.get('Class', 'Unknown'))
            fault_subclasses.add(f.get('SubClass', 'Unknown'))
    return sorted(list(fault_levels)),sorted(list(fault_classes)),sorted(list(fault_subclasses))

def get_normal_periods(faults: List[dict]) -> List[Tuple[int, int]]:
    """获取主机的正常时间段（故障时间段之外的时间）

    Args:
        faults: 主机的故障列表

    Returns:
        List[Tuple[int, int]]: 正常时间段列表，每个元素为(开始时间, 结束时间)
    """
    if not faults:
        return []

    # 按创建时间排序
    sorted_faults = sorted(faults, key=lambda x: x['CreatedTime'])

    # 获取最早和最晚时间
    earliest_time = sorted_faults[0]['CreatedTime']
    latest_time = sorted_faults[-1]['CreatedTime']

    # 构建正常时间段
    normal_periods = []
    current_time = earliest_time

    for fault in sorted_faults:
        # 故障开始时间
        fault_start = fault['CreatedTime']
        # 故障结束时间，如果未结束则使用下一个故障的开始时间
        fault_end = fault['ClosedTime'] if fault['ClosedTime'] else fault_start

        # 如果当前时间到故障开始时间之间有空隙，这就是一个正常时间段
        if current_time < fault_start:
            normal_periods.append((current_time, fault_start))

        # 更新当前时间为故障结束时间
        current_time = max(current_time, fault_end)

    # 添加最后一个故障之后到最新时间的正常时间段
    if current_time < latest_time:
        normal_periods.append((current_time, latest_time))

    return normal_periods

def generate_normal_samples(host: str,
                          faults: List[dict],
                          num_samples: int,
                          fault_type_list: List[str],
                          fault_level_list: List[str],
                          fault_class_list: List[str],
                          fault_subclass_list: List[str]) -> List[HostSample]:
    """在正常时间段内随机生成正常样本

    Args:
        host: 主机名
        faults: 主机的故障列表
        num_samples: 要生成的正常样本数量
        fault_type_list: 故障类型列表
        fault_level_list: 故障等级列表
        fault_class_list: 故障分类列表
        fault_subclass_list: 故障子分类列表

    Returns:
        List[HostSample]: 生成的正常样本列表
    """
    normal_periods = get_normal_periods(faults)
    if not normal_periods:
        return []

    normal_samples = []

    # 计算每个正常时间段的时长
    period_lengths = [(end - start) for start, end in normal_periods]
    total_length = sum(period_lengths)

    # 根据时间段长度分配样本数量
    for _ in range(num_samples):
        # 随机选择一个时间点
        random_time = random.uniform(0, total_length)

        # 确定随机时间点落在哪个时间段
        cumulative_length = 0
        for (start, end), length in zip(normal_periods, period_lengths):
            cumulative_length += length
            if random_time <= cumulative_length:
                # 在当前时间段内生成具体时间点
                sample_time = start + int(random_time - (cumulative_length - length))
                # 构建特征
                features = build_features_for_host_faults_with_hash(
                    faults,
                    sample_time,
                    fault_type_list,
                    fault_level_list,
                    fault_class_list,
                    fault_subclass_list,
                    host
                )
                # 创建样本（标签为0表示正常）
                sample = HostSample(host, sample_time, features, 0)
                normal_samples.append(sample)
                break
    return normal_samples


def build_normal_host_features(fault_type_list,fault_level_list,fault_class_list,fault_subclass_list,host_metadata = None):
    if not isinstance(host_metadata, dict):
        host_metadata = {}

    features = {
                    "fault_type_counts": [0] * len(fault_type_list),
                    "fault_level_counts": [0] * len(fault_level_list),
                    "fault_class_counts": [0] * len(fault_class_list),
                    "fault_subclass_counts": [0] * len(fault_subclass_list),
                    "history_fault_count": 0,
                    "days_since_last_fault": -1,
                    'max_fault_duration': 0,
                    'min_fault_duration'  :0,
                    'median_fault_duration' :0,
                    'unresolved_fault_count' : 0,
                    "mean_fault_interval": -1,
                    "min_fault_interval": -1,
                    "max_fault_interval": -1,
                    "std_fault_interval": -1,
                    "fault_frequency": 0,
                    "recent_fault_count": 0,
                    "recent_to_total_ratio": 0,
                }
    #features['metadata'] = host_metadata
    features['host_model'] = host_metadata.get('model', 'Unknown')
    features['gpu_model'] = host_metadata.get('gpu_model', 'Unknown')
    features['cpu_model'] = host_metadata.get('cpu_model', 'Unknown')
    features['quota_group'] = host_metadata.get('quota_group', 'Unknown')
    features['manufacturer'] = host_metadata.get('manufacturer', 'Unknown')
    features['main_board'] = get_main_board_model(host_metadata)

    return features


def build_features_for_host_faults_with_hash(faults, fault_time, fault_type_list,fault_level_list,fault_class_list,fault_subclass_list,host_metadata = None):
    if not isinstance(host_metadata, dict):
        host_metadata = {}
    history = [f for f in faults if f['CreatedTime'] < fault_time]
    features = {}
    #features['metadata'] = host_metadata
    features['host_model'] = host_metadata.get('model', 'Unknown')
    features['gpu_model'] = host_metadata.get('gpu_model', 'Unknown')
    features['cpu_model'] = host_metadata.get('cpu_model', 'Unknown')
    features['quota_group'] = host_metadata.get('quota_group', 'Unknown')
    features['main_board'] = get_main_board_model(host_metadata)
    features['manufacturer'] = host_metadata.get('manufacturer', 'Unknown')
    # 故障类型统计（固定顺序）
    fault_type_count = {ft: 0 for ft in fault_type_list}
    for f in history:
        ft = f.get('FaultType', 'Unknown')
        if ft in fault_type_count:
            fault_type_count[ft] += 1
    features['fault_type_counts'] = [fault_type_count[ft] for ft in fault_type_list]
    # 故障等级统计（固定顺序）
    fault_level_count = {lvl: 0 for lvl in fault_level_list}
    for f in history:
        lvl = f.get('Level', 'Unknown')
        if lvl in fault_level_count:
            fault_level_count[lvl] += 1
    features['fault_level_counts'] = [fault_level_count[lvl] for lvl in fault_level_list]
    # 故障类统计（固定顺序）
    fault_class_count = {cls: 0 for cls in fault_class_list}
    for f in history:
        cls = f.get('Class', 'Unknown')
        if cls in fault_class_count:
            fault_class_count[cls] += 1
    features['fault_class_counts'] = [fault_class_count[cls] for cls in fault_class_list]
    # 故障子类统计（固定顺序）
    fault_subclass_count = {sub: 0 for sub in fault_subclass_list}
    for f in history:
        sub = f.get('SubClass', 'Unknown')
        if sub in fault_subclass_count:
            fault_subclass_count[sub] += 1
    features['fault_subclass_counts'] = [fault_subclass_count[sub] for sub in fault_subclass_list]

    features['history_fault_count'] = len(history)    
    if history:
        last_fault_time = max(f['CreatedTime'] for f in history)
        days_since_last_fault = (fault_time - last_fault_time) / (1000 * 3600 * 24)
        features['days_since_last_fault'] = days_since_last_fault
    else:
        features['days_since_last_fault'] = -1
    # 4. 故障持续时间特征 - 简单统计
    durations = []
    for f in history:
        if f['ClosedTime']:
            durations.append((f['ClosedTime'] - f['CreatedTime']) / (1000 * 3600 ))
    # 5. 故障解决时间特征 - 更详细的统计
    if durations:
        features['max_fault_duration'] = max(durations)
        features['min_fault_duration'] = min(durations)
        features['median_fault_duration'] = sorted(durations)[len(durations)//2]
        features['unresolved_fault_count'] = sum(1 for f in history if not f.get('ClosedTime'))
    else:
        features['max_fault_duration'] = 0
        features['min_fault_duration'] = 0
        features['median_fault_duration'] = 0
        features['unresolved_fault_count'] = 0
    
    # 1. 时间衰减特征 - 给予近期故障更高的权重
    recent_faults = [f for f in history if (fault_time - f['CreatedTime']) < (3 * 24 * 3600 * 1000)]  # 最近3天
    features['recent_fault_count'] = len(recent_faults)
    features['recent_to_total_ratio'] = len(recent_faults) / max(1, len(history))
    # 2. 故障频率特征 - 计算单位时间内的故障数
    if history:
        time_span = (fault_time - min(f['CreatedTime'] for f in history)) / (7 * 24 * 3600 * 1000)  # 天数
        features['fault_frequency'] = len(history) / max(1, time_span)  # 每周故障数
    else:
        features['fault_frequency'] = 0
    # 3. 故障间隔特征 - 计算故障之间的时间间隔
    if len(history) >= 2:
        sorted_history = sorted(history, key=lambda x: x['CreatedTime'])
        intervals = [(sorted_history[i]['CreatedTime'] - sorted_history[i-1]['CreatedTime']) / (24 * 3600 * 1000) 
                    for i in range(1, len(sorted_history))]
        features['mean_fault_interval'] = sum(intervals) / len(intervals)
        features['min_fault_interval'] = min(intervals)
        features['max_fault_interval'] = max(intervals)
        features['std_fault_interval'] = (sum((x - features['mean_fault_interval'])**2 for x in intervals) / len(intervals))**0.5
    else:
        features['mean_fault_interval'] = -1
        features['min_fault_interval'] = -1
        features['max_fault_interval'] = -1
        features['std_fault_interval'] = -1

    return features

def extract_gpu_key_from_hostname(hostname: str) -> str:
    if not hostname:
        return ""
    h = hostname.lower()
    if "gpu-" not in h:
        return ""
    after = h.split("gpu-", 1)[1]
    token = re.split(r"[-.]", after, 1)[0]
    return token.strip()

def extract_gpu_key_from_gpu_model(gpu_model: str) -> str:
    if not gpu_model:
        return ""
    gm = gpu_model.lower()
    known_tokens = [
        "h100","h800","a100","a800","l40s","l40","v100","t4","a10","a30",
        "4090e","4090","3090","2080","p100","p40","k80"
    ]
    for token in known_tokens:
        if token in gm:
            return token
    return ""

def infer_gpu_key(hostname: str, host_metadata: Dict[str, Any]) -> str:
    key = extract_gpu_key_from_hostname(hostname)
    if key:
        return key
    if isinstance(host_metadata, dict):
        return extract_gpu_key_from_gpu_model(host_metadata.get("gpu_model", ""))
    return ""

def build_gpu_metadata_defaults(all_hosts_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    from collections import defaultdict, Counter
    field_names = ["model", "gpu_model", "cpu_model", "quota_group", "manufacturer"]
    counters = defaultdict(lambda: {fn: Counter() for fn in field_names + ["main_board_model"]})

    for host, meta in all_hosts_info.items():
        if not isinstance(meta, dict):
            continue
        gpu_key = infer_gpu_key(host, meta)
        if not gpu_key:
            continue
        for fn in field_names:
            val = meta.get(fn)
            if val not in (None, "", "Unknown"):
                counters[gpu_key][fn][val] += 1
        mb = meta.get("main_board")
        if isinstance(mb, dict):
            mb_model = mb.get("model")
            if mb_model not in (None, "", "Unknown"):
                counters[gpu_key]["main_board_model"][mb_model] += 1

    defaults = {}
    for gpu_key, field_counters in counters.items():
        meta = {}
        for fn in field_names:
            if field_counters[fn]:
                meta[fn] = field_counters[fn].most_common(1)[0][0]
        if field_counters["main_board_model"]:
            meta["main_board"] = {"model": field_counters["main_board_model"].most_common(1)[0][0]}
        if "gpu_model" not in meta:
            meta["gpu_model"] = gpu_key
        defaults[gpu_key] = meta
    return defaults

def normalize_host_metadata(hostname: str, host_metadata: Dict[str, Any], gpu_defaults: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(host_metadata, dict):
        host_metadata = {}
    gpu_key = infer_gpu_key(hostname, host_metadata)
    defaults = gpu_defaults.get(gpu_key, {})

    def pick(field: str, default_val: Any) -> Any:
        val = host_metadata.get(field)
        if val in (None, "", "Unknown"):
            return default_val if default_val not in (None, "", "Unknown") else "Unknown"
        return val

    normalized = {
        "model": pick("model", defaults.get("model", "Unknown")),
        "gpu_model": pick("gpu_model", defaults.get("gpu_model", gpu_key or "Unknown")),
        "cpu_model": pick("cpu_model", defaults.get("cpu_model", "Unknown")),
        "quota_group": pick("quota_group", defaults.get("quota_group", "Unknown")),
        "manufacturer": pick("manufacturer", defaults.get("manufacturer", "Unknown")),
    }

    mb_model = None
    mb = host_metadata.get("main_board")
    if isinstance(mb, dict):
        mb_model = mb.get("model")
    if mb_model in (None, "", "Unknown"):
        mb_default = None
        if isinstance(defaults.get("main_board"), dict):
            mb_default = defaults.get("main_board", {}).get("model")
        mb_model = mb_default if mb_default not in (None, "", "Unknown") else "Unknown"
    normalized["main_board"] = {"model": mb_model}
    return normalized

def get_main_board_model(host_metadata: Dict[str, Any]) -> str:
    if not isinstance(host_metadata, dict):
        return "Unknown"
    mb = host_metadata.get("main_board")
    if isinstance(mb, dict):
        return mb.get("model", "Unknown")
    return "Unknown"

def generate_adaptive_ltr_dataset(
    faults_data,
    all_hosts_info,
    output_path,
    window_days=7,
    min_positive_ratio=0.01,
    target_positive_ratio=0.1,
    enable_negative_sampling=True,
    sampling_seed=42,
    min_negatives_per_query=0,
    max_negatives_per_query=None,
):
    """
    基于故障密度的自适应时间窗口LTR数据集生成

    Args:
        faults_data: 故障数据
        output_path: 输出路径
        window_days: 预测时间窗口（天）
        min_positive_ratio: 最小正样本比例
        target_positive_ratio: 目标正样本比例
        enable_negative_sampling: 是否对负样本下采样（保留全部正样本）
        sampling_seed: 负采样随机种子（不同query会在此基础上加query_id）
        min_negatives_per_query: 每个query至少保留的负样本数（防止过小）
        max_negatives_per_query: 每个query最多保留的负样本数（防止过大）
    """
    from datetime import datetime, timedelta
    import os
    from collections import defaultdict

    # 获取所有特征列表
    fault_type_list = get_all_fault_types(faults_data)
    fault_level_list, fault_class_list, fault_subclass_list = get_all_fault_levels(faults_data)

    # 获取所有主机列表（尽可能保留所有 GPU 主机）
    all_hosts = sorted({
        host for host in (set(all_hosts_info.keys()) | set(faults_data.keys()))
        if "gpu" in host.lower()
    })

    gpu_defaults = build_gpu_metadata_defaults(all_hosts_info)

    print(f"总主机数: {len(all_hosts)}")

    # 收集所有故障时间点并按密度生成查询时间点
    all_fault_times = []
    for host_faults in faults_data.values():
        for fault in host_faults:
            all_fault_times.append(fault['CreatedTime'])

    all_fault_times.sort()
    print(f"总故障数: {len(all_fault_times)}")

    # 基于故障密度生成查询时间点
    query_timestamps = []
    window_ms = window_days * 24 * 3600 * 1000  # 转换为毫秒

    # 使用滑动窗口找到故障密集的时间段
    start_time = datetime(2024, 1, 1).timestamp() * 1000
    end_time = datetime(2026,1,1).timestamp() * 1000
    #end_time = datetime.now().timestamp() * 1000

    # 每3天生成一个候选查询点
    current_time = start_time
    step_ms = 3 * 24 * 3600 * 1000  # 3天步长

    while current_time < end_time:
        # 计算当前时间窗口内的故障数量
        window_start = current_time
        window_end = current_time + window_ms

        fault_count_in_window = sum(1 for ft in all_fault_times
                                  if window_start <= ft < window_end)

        # 如果故障数量足够，添加为候选查询点
        if fault_count_in_window >= len(all_hosts) * min_positive_ratio:
            query_timestamps.append(int(current_time))

        current_time += step_ms

    print(f"基于故障密度生成的查询数: {len(query_timestamps)}")

    # 生成样本
    ltr_samples = []
    query_to_timestamp = {}

    for query_id, query_timestamp in enumerate(query_timestamps):
        window_end_timestamp = query_timestamp + window_ms
        query_date = datetime.fromtimestamp(query_timestamp / 1000)
        print(f"处理查询 {query_id}: {query_date.strftime('%Y-%m-%d')}")

        # 先计算每台主机在该窗口是否为正样本，再做负采样（避免为将被丢弃的负样本构造特征，节省内存/时间）
        positive_hosts = []
        negative_hosts = []
        host_has_fault = {}

        for host in all_hosts:
            host_faults = faults_data.get(host, [])
            has_fault = any(
                query_timestamp <= fault['CreatedTime'] < window_end_timestamp and fault['SubClass']
                for fault in host_faults
            )
            host_has_fault[host] = has_fault
            if has_fault:
                positive_hosts.append(host)
            else:
                negative_hosts.append(host)

        positive_count = len(positive_hosts)
        total_hosts = len(all_hosts)
        raw_positive_ratio = positive_count / max(1, total_hosts)

        if positive_count == 0:
            # 没有正样本的 query 对排序训练帮助很有限，且会引入大量负样本；直接跳过
            print(f"  跳过：该 query 无正样本 (raw_pos_ratio={raw_positive_ratio:.4f})")
            query_to_timestamp.pop(query_id, None)
            continue
        query_to_timestamp[query_id] = query_date.strftime('%Y-%m-%d')

        rng = random.Random(sampling_seed + query_id)
        selected_negative_hosts = negative_hosts
        if enable_negative_sampling:
            # 目标：通过减少负样本，把正样本占比提升到 target_positive_ratio
            target_negative_count = int(np.ceil(positive_count * (1 - target_positive_ratio) / max(target_positive_ratio, 1e-8)))
            # 为了能计算 ndcg@10，尽量保证每个query至少有10条样本
            target_negative_count = max(target_negative_count, max(0, 10 - positive_count))
            target_negative_count = max(target_negative_count, int(min_negatives_per_query))
            if max_negatives_per_query is not None:
                target_negative_count = min(target_negative_count, int(max_negatives_per_query))
            target_negative_count = min(target_negative_count, len(negative_hosts))

            if target_negative_count < len(negative_hosts):
                selected_negative_hosts = rng.sample(negative_hosts, target_negative_count)

        selected_hosts = positive_hosts + selected_negative_hosts

        # 构造最终样本（只对 selected_hosts 构造特征）
        query_samples = []
        for host in selected_hosts:
            host_faults = faults_data.get(host, [])
            host_metadata = normalize_host_metadata(host, all_hosts_info.get(host, {}), gpu_defaults)

            if host in faults_data:
                features = build_features_for_host_faults_with_hash(
                    host_faults,
                    query_timestamp,
                    fault_type_list,
                    fault_level_list,
                    fault_class_list,
                    fault_subclass_list,
                    host_metadata,
                )
            else:
                features = build_normal_host_features(
                    fault_type_list,
                    fault_level_list,
                    fault_class_list,
                    fault_subclass_list,
                    host_metadata,
                )

            # 时间周期特征
            features['query_id'] = query_id  # 仅用于训练时分组，不作为预测特征
            features['month_sin'] = np.sin(2 * np.pi * query_date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * query_date.month / 12)
            features['day_of_week_sin'] = np.sin(2 * np.pi * query_date.weekday() / 7)
            features['day_of_week_cos'] = np.cos(2 * np.pi * query_date.weekday() / 7)
            features['is_weekend'] = 1 if query_date.weekday() >= 5 else 0

            label = 1 if host_has_fault.get(host, False) else 0
            query_samples.append(HostSample(host, query_timestamp, features, label))

        ltr_samples.extend(query_samples)

        final_positive_ratio = positive_count / max(1, len(query_samples))
        print(
            f"  raw_pos_ratio={raw_positive_ratio:.3f}, "
            f"after_sampling_pos_ratio={final_positive_ratio:.3f} "
            f"({positive_count}/{len(query_samples)})"
        )

    # 保存样本
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(ltr_samples, f)
    with open(output_path.replace(".pkl", "_query_to_timestamp.json"), "w") as f:
        json.dump(query_to_timestamp, f, ensure_ascii=False, indent=4)
    # 统计结果
    query_stats = defaultdict(lambda: [0, 0])
    for s in ltr_samples:
        query_stats[s.features['query_id']][1] += 1
        if s.label == 1:
            query_stats[s.features['query_id']][0] += 1

    print(f"\n=== 自适应数据集统计 ===")
    print(f"总样本数: {len(ltr_samples)}")
    print(f"有效查询数: {len(query_stats)}")

    ratios = [pos/total for pos, total in query_stats.values()]
    print(f"正样本比例统计: min={min(ratios):.3f}, max={max(ratios):.3f}, mean={sum(ratios)/len(ratios):.3f}")

    return ltr_samples

if __name__ == "__main__":
    import argparse
    with open("output/analysis/host_fault_detail.json", 'r', encoding="utf-8") as f:
        faults_data = json.load(f)
    with open("./all_hosts_info.json", 'r', encoding="utf-8") as f:
        all_hosts_info = json.load(f)
    final_faults_data = {}
    for hostname,faults in faults_data.items():
        if "gpu" in hostname.lower():
            final_faults_data[hostname] = faults
    del faults_data

    parser = argparse.ArgumentParser(description="Generate adaptive LTR dataset (with optional negative sampling).")
    parser.add_argument("--output", default="data/dataset/ltr_samples.pkl", help="Output pickle path")
    parser.add_argument("--target_positive_ratio", type=float, default=0.1, help="Target positive ratio per query")
    parser.add_argument("--sampling_seed", type=int, default=42, help="Random seed for negative sampling")
    parser.add_argument("--min_negatives_per_query", type=int, default=0, help="Minimum negatives kept per query")
    parser.add_argument(
        "--max_negatives_per_query",
        type=int,
        default=None,
        help="Maximum negatives kept per query (default: no limit)",
    )
    parser.add_argument(
        "--disable_negative_sampling",
        action="store_true",
        help="Disable negative down-sampling (keep all negatives)",
    )
    args = parser.parse_args()

    generate_adaptive_ltr_dataset(
        final_faults_data,
        all_hosts_info,
        args.output,
        target_positive_ratio=args.target_positive_ratio,
        enable_negative_sampling=not args.disable_negative_sampling,
        sampling_seed=args.sampling_seed,
        min_negatives_per_query=args.min_negatives_per_query,
        max_negatives_per_query=args.max_negatives_per_query,
    )
