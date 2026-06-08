"""
行为数据预处理脚本
功能：处理缺失值、异常值，提取行为特征
处理方法：
- 缺失值：线性插值
- 异常值：3σ原则
- 特征提取：≥10个特征
"""

import pandas as pd
import numpy as np
from scipy import interpolate
import os

np.random.seed(42)

def load_behavioral_data():
    """加载行为数据"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'behavioral_data.csv')
    df = pd.read_csv(data_path)
    df['record_date'] = pd.to_datetime(df['record_date'])
    df['access_time'] = pd.to_datetime(df['access_time'])
    return df

def add_missing_values(df, missing_rate=0.05):
    """添加随机缺失值以模拟真实场景"""
    df = df.copy()
    mask = np.random.random(df.shape[0]) < missing_rate
    df.loc[mask, 'amount'] = np.nan
    mask2 = np.random.random(df.shape[0]) < missing_rate
    df.loc[mask2, 'frequency'] = np.nan
    return df

def handle_missing_values(df):
    """处理缺失值：线性插值"""
    df = df.copy()
    df = df.sort_values(['student_id', 'record_date'])

    missing_before = df['amount'].isna().sum() + df['frequency'].isna().sum()

    df['amount'] = df.groupby('student_id')['amount'].transform(
        lambda x: x.interpolate(method='linear')
    )
    df['frequency'] = df.groupby('student_id')['frequency'].transform(
        lambda x: x.interpolate(method='linear')
    )

    df['amount'] = df['amount'].fillna(df['amount'].median())
    df['frequency'] = df['frequency'].fillna(df['frequency'].median())

    missing_after = df['amount'].isna().sum() + df['frequency'].isna().sum()
    print(f"缺失值处理：处理前 {missing_before} 个，处理后 {missing_after} 个")

    return df

def detect_outliers_3sigma(df, column):
    """使用3σ原则检测异常值"""
    mean = df[column].mean()
    std = df[column].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    return outliers, lower_bound, upper_bound

def handle_outliers(df):
    """处理异常值：3σ原则"""
    df = df.copy()

    outlier_cols = ['amount', 'frequency']
    total_outliers = 0

    for col in outlier_cols:
        outliers, lower, upper = detect_outliers_3sigma(df, col)
        count = outliers.sum()
        total_outliers += count

        df.loc[outliers, col] = df[col].median()
        print(f"{col} 列异常值处理：检测到 {count} 个异常值，范围 [{lower:.2f}, {upper:.2f}]")

    print(f"异常值处理：共处理 {total_outliers} 个异常值")
    return df

def extract_behavioral_features(df):
    """提取行为特征"""
    features_list = []

    for student_id in df['student_id'].unique():
        student_df = df[df['student_id'] == student_id].copy()
        student_df = student_df.sort_values('record_date')

        consumption_df = student_df[student_df['record_type'] == 'consumption']
        access_df = student_df[student_df['record_type'] == 'access']

        daily_consumption = consumption_df.groupby('record_date')['amount'].sum()
        daily_freq = consumption_df.groupby('record_date')['frequency'].sum()

        features = {
            'student_id': student_id,
            'record_count': len(student_df),
            'daily_avg_consumption': daily_consumption.mean() if len(daily_consumption) > 0 else 0,
            'daily_avg_frequency': daily_freq.mean() if len(daily_freq) > 0 else 0,
            'consumption_volatility': daily_consumption.std() if len(daily_consumption) > 1 else 0,
            'total_consumption': consumption_df['amount'].sum(),
            'total_frequency': consumption_df['frequency'].sum(),
            'library_access_count': len(access_df[access_df['location'] == '图书馆']),
            'dorm_access_count': len(access_df[access_df['location'] == '宿舍楼']),
            'social_activity_score': len(access_df),
            'weekend_activity_ratio': len(student_df[student_df['weekday'] >= 5]) / max(len(student_df), 1),
            'weekday_activity_ratio': len(student_df[student_df['weekday'] < 5]) / max(len(student_df), 1),
            'morning_activity_count': len(student_df[(student_df['access_time'].dt.hour >= 6) & (student_df['access_time'].dt.hour < 12)]),
            'afternoon_activity_count': len(student_df[(student_df['access_time'].dt.hour >= 12) & (student_df['access_time'].dt.hour < 18)]),
            'evening_activity_count': len(student_df[(student_df['access_time'].dt.hour >= 18) & (student_df['access_time'].dt.hour < 24)]),
            'night_activity_count': len(student_df[(student_df['access_time'].dt.hour >= 0) & (student_df['access_time'].dt.hour < 6)]),
            'canteen_visits': len(consumption_df[consumption_df['location'].str.contains('食堂', na=False)]),
            'supermarket_visits': len(consumption_df[consumption_df['location'] == '校园超市']),
            'campus_coverage': student_df['location'].nunique(),
            'consumption_regularity': 1 / (daily_consumption.std() + 1) if len(daily_consumption) > 1 else 1
        }

        if len(daily_consumption) > 1:
            features['daily_consumption_cv'] = daily_consumption.std() / (daily_consumption.mean() + 0.01)
        else:
            features['daily_consumption_cv'] = 0

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f"行为特征提取完成，共 {len(features_df)} 名学生的 {len(features_df.columns) - 1} 个特征")

    return features_df

def preprocess_behavioral_data():
    """预处理行为数据"""
    print("=" * 50)
    print("开始行为数据预处理...")
    print("=" * 50)

    print("\n1. 加载数据...")
    df = load_behavioral_data()
    print(f"原始数据: {len(df)} 条记录, {df['student_id'].nunique()} 名学生")

    print("\n2. 添加随机缺失值...")
    df = add_missing_values(df)
    print(f"缺失值添加完成")

    print("\n3. 处理缺失值（线性插值）...")
    df = handle_missing_values(df)

    print("\n4. 处理异常值（3σ原则）...")
    df = handle_outliers(df)

    print("\n5. 提取行为特征...")
    features_df = extract_behavioral_features(df)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, 'behavioral_data_cleaned.csv'), index=False, encoding='utf-8-sig')
    features_df.to_csv(os.path.join(output_dir, 'behavioral_features.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("行为数据预处理完成！")
    print(f"清洗后数据: {os.path.join(output_dir, 'behavioral_data_cleaned.csv')}")
    print(f"行为特征: {os.path.join(output_dir, 'behavioral_features.csv')}")
    print(f"特征数量: {len(features_df.columns) - 1} 个")
    print("=" * 50)

    return df, features_df

if __name__ == '__main__':
    preprocess_behavioral_data()
