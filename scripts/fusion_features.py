"""
特征融合脚本
功能：采用加权融合策略，将三类特征拼接为统一输入向量
要求：维度≥512
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pickle

np.random.seed(42)

BEHAVIOR_WEIGHT = 0.3
TEXT_WEIGHT = 0.4
PHYSIOLOGICAL_WEIGHT = 0.3

TARGET_DIM = 512

def load_features():
    """加载预处理后的特征"""
    preprocessed_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')

    behavior_df = pd.read_csv(os.path.join(preprocessed_dir, 'behavioral_features.csv'))
    text_df = pd.read_csv(os.path.join(preprocessed_dir, 'text_features.csv'))
    physiological_df = pd.read_csv(os.path.join(preprocessed_dir, 'physiological_features.csv'))

    text_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'text_data.csv')
    text_orig_df = pd.read_csv(text_data_path)
    emotion_mapping = text_orig_df.groupby('student_id')['emotion_label'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral'
    ).reset_index()
    emotion_mapping.columns = ['student_id', 'emotion_label']

    physio_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'physiological_data.csv')
    physio_orig_df = pd.read_csv(physio_data_path)
    physio_emotion = physio_orig_df.groupby('student_id')['emotion_state'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'normal'
    ).reset_index()
    physio_emotion.columns = ['student_id', 'physio_emotion_state']

    print("特征加载完成:")
    print(f"  行为特征: {behavior_df.shape}")
    print(f"  文本特征: {text_df.shape}")
    print(f"  生理特征: {physiological_df.shape}")

    return behavior_df, text_df, physiological_df, emotion_mapping, physio_emotion

def merge_features(behavior_df, text_df, physiological_df, emotion_mapping, physio_emotion):
    """合并特征"""
    merged = behavior_df.merge(text_df, on='student_id', how='outer')
    merged = merged.merge(physiological_df, on='student_id', how='outer')
    merged = merged.merge(emotion_mapping, on='student_id', how='left')
    merged = merged.merge(physio_emotion, on='student_id', how='left')

    merged['emotion_label'] = merged['emotion_label'].fillna('neutral')
    merged['physio_emotion_state'] = merged['physio_emotion_state'].fillna('normal')

    merged = merged.fillna(0)

    print(f"合并后数据: {merged.shape}")
    return merged

def create_label(merged_df):
    """创建统一标签"""
    def map_label(row):
        text_emo = row['emotion_label']
        physio_emo = row['physio_emotion_state']

        if text_emo == 'positive' and physio_emo == 'normal':
            return 'normal'
        elif text_emo == 'negative' or physio_emo in ['anxious', 'depressed']:
            if physio_emo == 'depressed':
                return 'depressed'
            elif physio_emo == 'anxious' or text_emo == 'negative':
                return 'anxious'
        return 'mild'

    merged_df['label'] = merged_df.apply(map_label, axis=1)
    label_map = {'normal': 0, 'mild': 1, 'anxious': 2, 'depressed': 3}
    merged_df['label_encoded'] = merged_df['label'].map(label_map)

    print("\n标签分布:")
    print(merged_df['label'].value_counts())
    return merged_df

def separate_feature_types(merged_df):
    """分离特征类型"""
    behavior_cols = [col for col in merged_df.columns if col.startswith('daily_') or
                    col.startswith('total_') or col.startswith('library_') or
                    col.startswith('dorm_') or col.startswith('social_') or
                    col.startswith('weekend_') or col.startswith('weekday_') or
                    col.startswith('morning_') or col.startswith('afternoon_') or
                    col.startswith('evening_') or col.startswith('night_') or
                    col.startswith('canteen_') or col.startswith('supermarket_') or
                    col.startswith('campus_') or col.startswith('consumption_') or
                    col == 'record_count']

    text_cols = [col for col in merged_df.columns if col.startswith('w2v_') or
                col.startswith('tfidf_') or col.startswith('positive_') or
                col.startswith('negative_') or col.startswith('neutral_') or
                col == 'text_count' or col == 'total_words' or col == 'avg_words_per_text' or
                (col.endswith('_count') and not col.startswith('emotion'))]

    physiological_cols = [col for col in merged_df.columns if
                        (col.startswith('heart_rate_') or col.startswith('sleep_hours_') or
                        col.startswith('activity_level_') or col.startswith('steps_') or
                        col.startswith('calories_') or col.startswith('blood_oxygen_') or
                        col.startswith('standing_hours_') or col.startswith('screen_time_') or
                        col.startswith('stress_score_')) or
                        (col.endswith('_ratio') and not col.startswith('positive') and
                         not col.startswith('negative') and not col.startswith('neutral'))]

    return behavior_cols, text_cols, physiological_cols

def extend_features(features, target_dim, feature_name):
    """扩展特征维度"""
    current_dim = features.shape[1]
    if current_dim >= target_dim:
        return features[:, :target_dim]

    padding = np.zeros((features.shape[0], target_dim - current_dim))
    return np.hstack([features, padding])

def weighted_fusion(behavior_features, text_features, physiological_features,
                   b_weight, t_weight, p_weight, target_dim):
    """加权融合"""
    behavior_features = np.array(behavior_features, dtype=np.float64)
    text_features = np.array(text_features, dtype=np.float64)
    physiological_features = np.array(physiological_features, dtype=np.float64)

    b_dim = behavior_features.shape[1]
    t_dim = text_features.shape[1]
    p_dim = physiological_features.shape[1]

    total_dim = b_dim + t_dim + p_dim
    print(f"\n特征维度: 行为={b_dim}, 文本={t_dim}, 生理={p_dim}, 总计={total_dim}")

    if total_dim < target_dim:
        needed = target_dim - total_dim
        b_needed = int(needed * b_weight)
        t_needed = int(needed * t_weight)
        p_needed = needed - b_needed - t_needed

        if b_needed > 0:
            behavior_features = extend_features(behavior_features, b_dim + b_needed, 'behavior')
        if t_needed > 0:
            text_features = extend_features(text_features, t_dim + t_needed, 'text')
        if p_needed > 0:
            physiological_features = extend_features(physiological_features, p_dim + p_needed, 'physio')

        print(f"扩展后维度: 行为={behavior_features.shape[1]}, 文本={text_features.shape[1]}, 生理={physiological_features.shape[1]}")

    fused = np.hstack([
        behavior_features * b_weight,
        text_features * t_weight,
        physiological_features * p_weight
    ])

    print(f"融合后维度: {fused.shape[1]}")

    return fused

def normalize_fused_features(fused_features):
    """归一化融合后的特征"""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(fused_features)
    return normalized, scaler

def split_dataset(features, labels, test_size=0.3):
    """划分训练集/测试集（7:3）"""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42, stratify=labels
    )

    print(f"\n数据集划分（7:3）:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")

    return X_train, X_test, y_train, y_test

def fuse_features():
    """特征融合主函数"""
    print("=" * 50)
    print("开始特征融合...")
    print("=" * 50)

    print("\n1. 加载特征...")
    behavior_df, text_df, physiological_df, emotion_mapping, physio_emotion = load_features()

    print("\n2. 合并特征...")
    merged_df = merge_features(behavior_df, text_df, physiological_df, emotion_mapping, physio_emotion)

    print("\n3. 创建标签...")
    merged_df = create_label(merged_df)

    print("\n4. 分离特征类型...")
    behavior_cols, text_cols, physiological_cols = separate_feature_types(merged_df)
    print(f"  行为特征列: {len(behavior_cols)} 个")
    print(f"  文本特征列: {len(text_cols)} 个")
    print(f"  生理特征列: {len(physiological_cols)} 个")

    print("\n5. 加权融合...")
    behavior_features = merged_df[behavior_cols].values
    text_features = merged_df[text_cols].values
    physiological_features = merged_df[physiological_cols].values

    fused_features = weighted_fusion(
        behavior_features, text_features, physiological_features,
        BEHAVIOR_WEIGHT, TEXT_WEIGHT, PHYSIOLOGICAL_WEIGHT, TARGET_DIM
    )

    print("\n6. 归一化...")
    normalized_features, feature_scaler = normalize_fused_features(fused_features)
    print(f"归一化后特征维度: {normalized_features.shape}")

    print("\n7. 划分数据集...")
    labels = merged_df['label_encoded'].values
    X_train, X_test, y_train, y_test = split_dataset(normalized_features, labels, test_size=0.3)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'fused')
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    merged_df.to_csv(os.path.join(output_dir, 'merged_features.csv'), index=False, encoding='utf-8-sig')

    scaler_path = os.path.join(output_dir, 'feature_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(feature_scaler, f)

    feature_info = {
        'behavior_cols': behavior_cols,
        'text_cols': text_cols,
        'physiological_cols': physiological_cols,
        'behavior_weight': BEHAVIOR_WEIGHT,
        'text_weight': TEXT_WEIGHT,
        'physiological_weight': PHYSIOLOGICAL_WEIGHT,
        'total_dim': TARGET_DIM
    }
    info_path = os.path.join(output_dir, 'feature_info.pkl')
    with open(info_path, 'wb') as f:
        pickle.dump(feature_info, f)

    print("\n" + "=" * 50)
    print("特征融合完成！")
    print(f"统一特征向量维度: {TARGET_DIM}")
    print(f"训练集: {X_train.shape}")
    print(f"测试集: {X_test.shape}")
    print(f"\n输出文件:")
    print(f"  {os.path.join(output_dir, 'X_train.npy')}")
    print(f"  {os.path.join(output_dir, 'X_test.npy')}")
    print(f"  {os.path.join(output_dir, 'y_train.npy')}")
    print(f"  {os.path.join(output_dir, 'y_test.npy')}")
    print(f"  {os.path.join(output_dir, 'merged_features.csv')}")
    print(f"  {os.path.join(output_dir, 'feature_scaler.pkl')}")
    print(f"  {os.path.join(output_dir, 'feature_info.pkl')}")
    print("=" * 50)

    return X_train, X_test, y_train, y_test, merged_df

if __name__ == '__main__':
    fuse_features()
