from typing import List,Dict,Any,Tuple
import json
from datetime import datetime
import random
import pickle

class HostSample:
    def __init__(self, host: str, fault_time: int, features: Dict[str, Any], label: int):
        self.host = host
        self.fault_time = fault_time  # Fault occurrence time
        self.features = features      # Feature dictionary
        self.label = label            # Label (1 for fault, 0 for normal)


def get_all_fault_types(faults_data):
    fault_types = set()
    for faults in faults_data.values():
        for f in faults:
            ft = f.get('FaultType', 'Unknown')
            fault_types.add(ft)
    fault_types = sorted(list(fault_types))  # Fixed order
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
    """Get normal time periods for host (time outside fault periods)

    Args:
        faults: List of host faults

    Returns:
        List[Tuple[int, int]]: List of normal time periods, each element is (start_time, end_time)
    """
    if not faults:
        return []

    # Sort by creation time
    sorted_faults = sorted(faults, key=lambda x: x['CreatedTime'])

    # Get earliest and latest time
    earliest_time = sorted_faults[0]['CreatedTime']
    latest_time = sorted_faults[-1]['CreatedTime']

    # Build normal time periods
    normal_periods = []
    current_time = earliest_time

    for fault in sorted_faults:
        # Fault start time
        fault_start = fault['CreatedTime']
        # Fault end time, if not ended use next fault start time
        fault_end = fault['ClosedTime'] if fault['ClosedTime'] else fault_start

        # If there's a gap between current time and fault start time, this is a normal period
        if current_time < fault_start:
            normal_periods.append((current_time, fault_start))

        # Update current time to fault end time
        current_time = max(current_time, fault_end)

    # Add normal period from last fault to latest time
    if current_time < latest_time:
        normal_periods.append((current_time, latest_time))

    return normal_periods

def generate_normal_samples(host: str,
                          faults: List[dict],
                          num_samples: int,
                          fault_level_list: List[str],
                          fault_class_list: List[str],
                          fault_subclass_list: List[str]) -> List[HostSample]:
    """Randomly generate normal samples within normal time periods

    Args:
        host: Host name
        faults: List of host faults
        num_samples: Number of normal samples to generate
        fault_level_list: List of fault levels
        fault_class_list: List of fault classes
        fault_subclass_list: List of fault subclasses

    Returns:
        List[HostSample]: Generated normal sample list
    """
    normal_periods = get_normal_periods(faults)
    if not normal_periods:
        return []

    normal_samples = []

    # Calculate duration of each normal time period
    period_lengths = [(end - start) for start, end in normal_periods]
    total_length = sum(period_lengths)

    # Allocate sample count based on time period length
    for _ in range(num_samples):
        # Randomly select a time point
        random_time = random.uniform(0, total_length)

        # Determine which time period the random time point falls into
        cumulative_length = 0
        for (start, end), length in zip(normal_periods, period_lengths):
            cumulative_length += length
            if random_time <= cumulative_length:
                # Generate specific time point within current time period
                sample_time = start + int(random_time - (cumulative_length - length))
                # Build features
                features = build_features_for_host_faults_with_hash(
                    faults,
                    sample_time,
                    fault_level_list,
                    fault_class_list,
                    fault_subclass_list,
                    host
                )
                # Create sample (label 0 indicates normal)
                sample = HostSample(host, sample_time, features, 0)
                normal_samples.append(sample)
                break
    return normal_samples

def build_features_for_host_faults_with_hash(faults, fault_time, fault_level_list,fault_class_list,fault_subclass_list,hostname = None):
    history = [f for f in faults if f['CreatedTime'] < fault_time]
    features = {}

    # Add host-related features
    if hostname:
        # Parse hostname format: gpu-model-id.datacenter_ip
        parts = hostname.split('.')
        host_parts = parts[0].split('-') if parts else []
        
        # Extract host model
        if len(host_parts) >= 2:
            features['host_model'] = host_parts[1]  
        else:
            features['host_model'] = 'unknown'
            
        # Extract host ID
        if len(host_parts) >= 3:
            try:
                features['host_id'] = int(host_parts[2])
            except ValueError:
                features['host_id'] = -1
        else:
            features['host_id'] = -1
            
        # Extract datacenter information
        if len(parts) > 1:
            features['datacenter'] = hostname.removeprefix(parts[0] + '.') 
        else:
            features['datacenter'] = 'unknown'

    # Fault level statistics (fixed order)
    fault_level_count = {lvl: 0 for lvl in fault_level_list}
    for f in history:
        lvl = f.get('Level', 'Unknown')
        if lvl in fault_level_count:
            fault_level_count[lvl] += 1
    features['fault_level_counts'] = [fault_level_count[lvl] for lvl in fault_level_list]
    # Fault class statistics (fixed order)
    fault_class_count = {cls: 0 for cls in fault_class_list}
    for f in history:
        cls = f.get('Class', 'Unknown')
        if cls in fault_class_count:
            fault_class_count[cls] += 1
    features['fault_class_counts'] = [fault_class_count[cls] for cls in fault_class_list]
    # Fault subclass statistics (fixed order)
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
    # 4. Fault duration features - simple statistics
    durations = []
    for f in history:
        if f['ClosedTime']:
            durations.append((f['ClosedTime'] - f['CreatedTime']) / (1000 * 3600 ))
    # 5. Fault resolution time features - more detailed statistics
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
    
    # 1. Time decay features - give higher weight to recent faults
    recent_faults = [f for f in history if (fault_time - f['CreatedTime']) < (3 * 24 * 3600 * 1000)]  # Last 3 days
    features['recent_fault_count'] = len(recent_faults)
    features['recent_to_total_ratio'] = len(recent_faults) / max(1, len(history))
    # 2. Fault frequency features - calculate fault count per unit time
    if history:
        time_span = (fault_time - min(f['CreatedTime'] for f in history)) / (7 * 24 * 3600 * 1000)  # Days
        features['fault_frequency'] = len(history) / max(1, time_span)  # Faults per week
    else:
        features['fault_frequency'] = 0
    # 3. Fault interval features - calculate time intervals between faults
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

def generate_balanced_ltr_dataset(faults_data, output_path, min_positive_ratio=0.01, max_positive_ratio=0.5, target_positive_ratio=0.1):
    """
    Generate balanced training dataset for LTR algorithm using adaptive time windows and sample balancing strategies

    Args:
        faults_data: Dictionary containing host fault information, format: {host: [fault1, fault2, ...]}
        output_path: Path to output dataset
        min_positive_ratio: Minimum positive sample ratio threshold, queries below this will be filtered
        max_positive_ratio: Maximum positive sample ratio threshold, negative sampling will be performed above this
        target_positive_ratio: Target positive sample ratio

    Returns:
        Generated LTR dataset sample list
    """
    from datetime import datetime, timedelta
    import os
    from collections import defaultdict
    import random

    # Get all feature lists
    fault_level_list, fault_class_list, fault_subclass_list = get_all_fault_levels(faults_data)

    # Get all host lists
    all_hosts = list(faults_data.keys())

    # Add normal hosts
    normal_hosts = []
    try:
        with open("data/normal_hosts.json", "r", encoding="utf-8") as f:
            normal_hosts = json.load(f)
            all_hosts.extend(normal_hosts)
    except:
        print("normal_hosts.json file not found, using only hosts from fault data")

    # Remove duplicates
    all_hosts = list(set(all_hosts))
    print(f"Total hosts: {len(all_hosts)}")

    # Step 1: Collect all fault timestamps for generating adaptive time windows
    all_fault_times = []
    for host_faults in faults_data.values():
        for fault in host_faults:
            all_fault_times.append(fault['CreatedTime'])

    all_fault_times.sort()
    print(f"Total faults: {len(all_fault_times)}")

    # Step 2: Generate candidate query timestamps (based on fault density)
    start_date = datetime(2024, 1, 1)
    end_date = datetime.now()
    start_timestamp = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)

    # Generate candidate time windows (randomly select one day per week)
    candidate_queries = []
    current_week_start = start_date
    while current_week_start <= end_date:
        # Calculate this week's end time (day before next week starts)
        next_week_start = current_week_start + timedelta(weeks=1)
        # Ensure not exceeding end date
        week_end = min(next_week_start - timedelta(days=1), end_date)
        
        # Calculate available days in this week
        days_in_week = (week_end - current_week_start).days + 1
        
        # Randomly select one day in this week
        random_day = random.randint(0, days_in_week - 1)
        selected_date = current_week_start + timedelta(days=random_day)
        
        candidate_queries.append(selected_date)
        
        # Move to next week start
        current_week_start = next_week_start

    print(f"Candidate queries: {len(candidate_queries)}")

    # Step 3: Calculate positive ratio for each candidate query, and filter and balance
    valid_queries = []

    for query_date in candidate_queries:
        query_timestamp = int(query_date.timestamp() * 1000)
        next_week = query_date + timedelta(weeks=1)
        next_week_timestamp = int(next_week.timestamp() * 1000)

        # calculate the positive ratio in the time window
        positive_count = 0
        total_count = len(all_hosts)

        for host in all_hosts:
            host_faults = faults_data.get(host, [])
            has_fault = any(
                query_timestamp <= fault['CreatedTime'] < next_week_timestamp
                for fault in host_faults
            )
            if has_fault:
                positive_count += 1

        positive_ratio = positive_count / total_count if total_count > 0 else 0

        # only keep queries with positive ratio in reasonable range
        if positive_ratio >= min_positive_ratio:
            valid_queries.append({
                'date': query_date,
                'timestamp': query_timestamp,
                'next_week_timestamp': next_week_timestamp,
                'positive_ratio': positive_ratio,
                'positive_count': positive_count,
                'total_count': total_count
            })

    print(f"valid queries: {len(valid_queries)}")
    print(f"positive ratio distribution: min={min([q['positive_ratio'] for q in valid_queries]):.3f}, "
          f"max={max([q['positive_ratio'] for q in valid_queries]):.3f}, "
          f"mean={sum([q['positive_ratio'] for q in valid_queries])/len(valid_queries):.3f}")

    # Step 4: Generate balanced samples
    ltr_samples = []

    for query_id, query_info in enumerate(valid_queries):
        query_date = query_info['date']
        query_timestamp = query_info['timestamp']
        next_week_timestamp = query_info['next_week_timestamp']
        positive_ratio = query_info['positive_ratio']

        print(f"processing query {query_id}: {query_date.strftime('%Y-%m-%d')}, "
              f"positive ratio: {positive_ratio:.3f}")

        # collect all samples
        query_samples = []

        for host in all_hosts:
            host_faults = faults_data.get(host, [])

            # check if the host has fault in the next week
            has_fault = any(
                query_timestamp <= fault['CreatedTime'] < next_week_timestamp
                for fault in host_faults
            )


            # build features
            if host in faults_data:
                features = build_features_for_host_faults_with_hash(
                    host_faults,
                    query_timestamp,
                    fault_level_list,
                    fault_class_list,
                    fault_subclass_list,
                    host
                )
            else:
                # for hosts without fault history, set default features
                features = {
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
                    "recent_to_total_ratio": 0
                }

            # modify to more suitable time period features
            features['query_id'] = query_id  # only used for training, not for prediction
            features['month_of_year'] = query_date.month  # 1-12 represents the month of the year
            features['week_of_month'] = (query_date.day - 1) // 7 + 1  # 1-5 represents the week of the month
            features['day_of_week'] = query_date.weekday()  # 0-6 represents Monday to Sunday
            features['day_of_month'] = query_date.day  # 1-31 represents the day of the month
            features['is_weekend'] = 1 if query_date.weekday() >= 5 else 0  # 1 if it is weekend, 0 if it is not

            # create sample
            label = 1 if has_fault else 0
            sample = HostSample(host, query_timestamp, features, label)
            query_samples.append(sample)

        # Step 5: Sample balancing strategy
        positive_samples = [s for s in query_samples if s.label == 1]
        negative_samples = [s for s in query_samples if s.label == 0]

        if positive_ratio > max_positive_ratio:
            # if the positive ratio is too high, perform negative sampling
            target_negative_count = int(len(positive_samples) * (1 - target_positive_ratio) / target_positive_ratio)
            if target_negative_count < len(negative_samples):
                negative_samples = random.sample(negative_samples, target_negative_count)
                print(f"  negative sampling: {len(negative_samples)} negative samples")

        # merge samples
        balanced_samples = positive_samples + negative_samples

        # update query_id (because some samples may be filtered)
        for sample in balanced_samples:
            sample.features['query_id'] = query_id

        ltr_samples.extend(balanced_samples)

        final_positive_ratio = len(positive_samples) / len(balanced_samples)
        print(f"  final positive ratio: {final_positive_ratio:.3f} "
              f"({len(positive_samples)}/{len(balanced_samples)})")

    # ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # save samples
    with open(output_path, "wb") as f:
        pickle.dump(ltr_samples, f)

    return ltr_samples

def generate_adaptive_ltr_dataset(faults_data, output_path, window_days=7, min_positive_ratio=0.01, target_positive_ratio=0.1):
    """
    Generate LTR dataset with adaptive time window based on fault density

    Args:
        faults_data: Fault data
        output_path: Output path
        window_days: Prediction window (days)
        min_positive_ratio: Minimum positive sample ratio
        target_positive_ratio: Target positive sample ratio
    """
    from datetime import datetime, timedelta
    import os
    from collections import defaultdict
    import random

    # get all feature lists
    fault_level_list, fault_class_list, fault_subclass_list = get_all_fault_levels(faults_data)

    # get all host lists
    all_hosts = list(faults_data.keys())

    # add normal hosts
    normal_hosts = []
    try:
        with open("data/normal_hosts.json", "r", encoding="utf-8") as f:
            normal_hosts = json.load(f)
            all_hosts.extend(normal_hosts)
    except:
        print("normal_hosts.json file not found, using only hosts from fault data")

    all_hosts = list(set(all_hosts))
    print(f"Total hosts: {len(all_hosts)}")

    # collect all fault times and generate query times based on density
    all_fault_times = []
    for host_faults in faults_data.values():
        for fault in host_faults:
            all_fault_times.append(fault['CreatedTime'])

    all_fault_times.sort()
    print(f"Total faults: {len(all_fault_times)}")

    # generate query times based on fault density
    query_timestamps = []
    window_ms = window_days * 24 * 3600 * 1000  # convert to milliseconds

    # use sliding window to find fault dense time periods
    start_time = datetime(2024, 1, 1).timestamp() * 1000
    end_time = datetime.now().timestamp() * 1000

    # generate one candidate query point every 3 days
    current_time = start_time
    step_ms = 3 * 24 * 3600 * 1000  # 3 days step

    while current_time < end_time:
        # calculate the number of faults in the current time window
        window_start = current_time
        window_end = current_time + window_ms

        fault_count_in_window = sum(1 for ft in all_fault_times
                                  if window_start <= ft < window_end)

        # if the number of faults is enough, add as a candidate query point
        if fault_count_in_window >= len(all_hosts) * min_positive_ratio:
            query_timestamps.append(int(current_time))

        current_time += step_ms

    print(f"Number of queries generated based on fault density: {len(query_timestamps)}")

    # generate samples
    ltr_samples = []

    for query_id, query_timestamp in enumerate(query_timestamps):
        window_end_timestamp = query_timestamp + window_ms
        query_date = datetime.fromtimestamp(query_timestamp / 1000)

        print(f"Processing query {query_id}: {query_date.strftime('%Y-%m-%d')}")

        # collect all samples for this query
        query_samples = []
        positive_count = 0

        for host in all_hosts:
            host_faults = faults_data.get(host, [])

            # check if the host has fault in the time window
            has_fault = any(
                query_timestamp <= fault['CreatedTime'] < window_end_timestamp and fault['SubClass']
                for fault in host_faults
            )

            if has_fault:
                positive_count += 1

            # build features
            if host in faults_data:
                features = build_features_for_host_faults_with_hash(
                    host_faults,
                    query_timestamp,
                    fault_level_list,
                    fault_class_list,
                    fault_subclass_list,
                    host
                )
            else:
                    features = {
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
                    "recent_to_total_ratio": 0
                }

            # modify to more suitable time period features
            features['query_id'] = query_id  # only used for training, not for prediction
            features['month_of_year'] = query_date.month  # 1-12 represents the month of the year
            features['week_of_month'] = (query_date.day - 1) // 7 + 1  # 1-5 represents the week of the month
            features['day_of_week'] = query_date.weekday()  # 0-6 represents Monday to Sunday
            features['day_of_month'] = query_date.day  # 1-31 represents the day of the month
            features['is_weekend'] = 1 if query_date.weekday() >= 5 else 0  # 1 if it is weekend, 0 if it is not

            # create sample
            label = 1 if has_fault else 0
            sample = HostSample(host, query_timestamp, features, label)
            query_samples.append(sample)

        # sample balancing
        positive_samples = [s for s in query_samples if s.label == 1]
        negative_samples = [s for s in query_samples if s.label == 0]

        current_positive_ratio = len(positive_samples) / len(query_samples)

        # if the positive ratio is too high, perform negative sampling
        if current_positive_ratio > target_positive_ratio:
            target_negative_count = int(len(positive_samples) * (1 - target_positive_ratio) / target_positive_ratio)
            if target_negative_count < len(negative_samples):
                negative_samples = random.sample(negative_samples, target_negative_count)

        # merge samples
        balanced_samples = positive_samples + negative_samples
        ltr_samples.extend(balanced_samples)

        final_positive_ratio = len(positive_samples) / len(balanced_samples)
        print(f"  positive ratio: {final_positive_ratio:.3f} ({len(positive_samples)}/{len(balanced_samples)})")

    # save samples
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(ltr_samples, f)

    # statistics
    query_stats = defaultdict(lambda: [0, 0])
    for s in ltr_samples:
        query_stats[s.features['query_id']][1] += 1
        if s.label == 1:
            query_stats[s.features['query_id']][0] += 1

    print(f"\n=== Adaptive dataset statistics ===")
    print(f"Total samples: {len(ltr_samples)}")
    print(f"Valid queries: {len(query_stats)}")

    ratios = [pos/total for pos, total in query_stats.values()]
    print(f"Positive ratio statistics: min={min(ratios):.3f}, max={max(ratios):.3f}, mean={sum(ratios)/len(ratios):.3f}")

    return ltr_samples

def generate_ltr_dataset(faults_data, output_path):
    """
    Original LTR dataset generation function (for backward compatibility)
    """
    return generate_balanced_ltr_dataset(faults_data, output_path,
                                       min_positive_ratio=0.0,
                                       max_positive_ratio=1.0,
                                       target_positive_ratio=0.1)

if __name__ == "__main__":
    with open("data/host_fault_detail.json", 'r', encoding="utf-8") as f:
        faults_data = json.load(f)

    # generate LTR training dataset
    ltr_samples = generate_balanced_ltr_dataset(faults_data, "data/ltr_samples.pkl")
