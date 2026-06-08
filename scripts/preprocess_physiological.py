"""
生理数据预处理脚本
功能：StandardScaler标准化，划分正常/异常区间特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

np.random.seed(42)

HEALTH_RANGES = {
    'heart_rate': {'low': 60, 'high': 100, 'unit': 'bpm'},
    'sleep_hours': {'low': 7.0, 'high': 9.0, 'unit': 'hours'},
    'activity_level': {'low': 5000, 'high': 15000, 'unit': 'steps'},
    'blood_oxygen': {'low': 95.0, 'high': 100.0, 'unit': '%'},
    'stress_score': {'low': 0, 'high': 50, 'unit': 'score'}
}

def load_physiological_data():
    """加载生理数据"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'physiological_data.csv')
    df = pd.read_csv(data_path)
    df['record_date'] = pd.to_datetime(df['record_date'])
    df['measurement_time'] = pd.to_datetime(df['measurement_time'])
    return df

def check_missing_values(df):
    """检查缺失值"""
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print("发现缺失值:")
        for col, count in missing_cols.items():
            print(f"  {col}: {count}")
        for col in missing_cols.index:
            df[col] = df[col].fillna(df[col].median())
    else:
        print("未发现缺失值")
    return df

def create_normal_abnormal_features(df):
    """划分正常/异常区间特征"""
    df = df.copy()

    for feature, ranges in HEALTH_RANGES.items():
        if feature in df.columns:
            low, high = ranges['low'], ranges['high']
            df[f'{feature}_normal'] = ((df[feature] >= low) & (df[feature] <= high)).astype(int)
            df[f'{feature}_abnormal'] = 1 - df[f'{feature}_normal']

    return df

def extract_physiological_features(df):
    """提取生理特征"""
    features_list = []

    for student_id in df['student_id'].unique():
        student_df = df[df['student_id'] == student_id].copy()
        student_df = student_df.sort_values('record_date')

        features = {'student_id': student_id}

        numeric_cols = ['heart_rate', 'sleep_hours', 'activity_level', 'steps',
                       'calories', 'blood_oxygen', 'standing_hours',
                       'screen_time', 'stress_score']

        for col in numeric_cols:
            if col in student_df.columns:
                values = student_df[col]
                features[f'{col}_mean'] = values.mean()
                features[f'{col}_std'] = values.std() if len(values) > 1 else 0
                features[f'{col}_min'] = values.min()
                features[f'{col}_max'] = values.max()
                features[f'{col}_median'] = values.median()
            else:
                features[f'{col}_mean'] = 0
                features[f'{col}_std'] = 0
                features[f'{col}_min'] = 0
                features[f'{col}_max'] = 0
                features[f'{col}_median'] = 0

        normal_cols = [col for col in student_df.columns if col.endswith('_normal')]
        if normal_cols:
            for col in normal_cols:
                features[f'{col}_ratio'] = student_df[col].mean()
        else:
            for feature in HEALTH_RANGES.keys():
                features[f'{feature}_normal_ratio'] = 0.5

        emotion_counts = student_df['emotion_state'].value_counts()
        total = len(student_df)
        features['emotion_normal_ratio'] = emotion_counts.get('normal', 0) / total
        features['emotion_anxious_ratio'] = emotion_counts.get('anxious', 0) / total
        features['emotion_depressed_ratio'] = emotion_counts.get('depressed', 0) / total

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    return features_df

def standardize_features(df, fit=True, scaler=None):
    """StandardScaler标准化"""
    df = df.copy()

    numeric_cols = ['heart_rate', 'sleep_hours', 'activity_level', 'steps',
                   'calories', 'blood_oxygen', 'standing_hours',
                   'screen_time', 'stress_score']

    feature_cols = []
    for col in numeric_cols:
        feature_cols.extend([
            f'{col}_mean', f'{col}_std', f'{col}_min',
            f'{col}_max', f'{col}_median'
        ])

    cols_to_scale = [col for col in feature_cols if col in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler

def normalize_features(df):
    """MinMax归一化（0-1范围）"""
    df = df.copy()

    numeric_cols = ['heart_rate', 'sleep_hours', 'activity_level', 'steps',
                   'calories', 'blood_oxygen', 'standing_hours',
                   'screen_time', 'stress_score']

    feature_cols = []
    for col in numeric_cols:
        feature_cols.extend([
            f'{col}_mean', f'{col}_std', f'{col}_min',
            f'{col}_max', f'{col}_median'
        ])

    ratio_cols = [col for col in df.columns if '_ratio' in col]
    cols_to_normalize = ratio_cols

    if cols_to_normalize:
        min_max_scaler = MinMaxScaler()
        df[cols_to_normalize] = min_max_scaler.fit_transform(df[cols_to_normalize])

    return df

def preprocess_physiological_data():
    """预处理生理数据"""
    print("=" * 50)
    print("开始生理数据预处理...")
    print("=" * 50)

    print("\n1. 加载生理数据...")
    df = load_physiological_data()
    print(f"原始数据: {len(df)} 条记录, {df['student_id'].nunique()} 名学生")

    print("\n2. 检查并处理缺失值...")
    df = check_missing_values(df)

    print("\n3. 创建正常/异常区间特征...")
    df = create_normal_abnormal_features(df)
    normal_cols = [col for col in df.columns if col.endswith('_normal')]
    print(f"创建了 {len(normal_cols)} 个正常/异常区间特征")

    print("\n4. 提取生理特征...")
    features_df = extract_physiological_features(df)
    print(f"提取了 {len(features_df.columns) - 1} 个生理特征")

    print("\n5. StandardScaler标准化...")
    features_df, std_scaler = standardize_features(features_df, fit=True)
    print("StandardScaler标准化完成")

    print("\n6. MinMax归一化...")
    features_df = normalize_features(features_df)
    print("归一化完成")

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, 'physiological_data_cleaned.csv'), index=False, encoding='utf-8-sig')
    features_df.to_csv(os.path.join(output_dir, 'physiological_features.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("生理数据预处理完成！")
    print(f"清洗后数据: {os.path.join(output_dir, 'physiological_data_cleaned.csv')}")
    print(f"生理特征: {os.path.join(output_dir, 'physiological_features.csv')}")
    print(f"特征数量: {len(features_df.columns) - 1} 个")
    print("  - 统计特征: 45 个（9个指标 × 5个统计量）")
    print("  - 正常/异常比例: 10 个")
    print("  - 情绪状态比例: 3 个")
    print("=" * 50)

    return df, features_df

if __name__ == '__main__':
    preprocess_physiological_data()
