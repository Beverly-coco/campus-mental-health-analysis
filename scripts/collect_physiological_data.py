"""
生理数据采集脚本
功能：模拟学生心率、睡眠时长、活动量等生理指标
数据量：≥2000条记录
关联情绪状态标签
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

STUDENT_COUNT = 500
RECORDS_PER_STUDENT = 5
TOTAL_RECORDS = STUDENT_COUNT * RECORDS_PER_STUDENT

EMOTION_STATES = ['normal', 'anxious', 'depressed']

def generate_student_ids(count):
    """生成学生ID列表"""
    return [f'S{str(i).zfill(6)}' for i in range(1, count + 1)]

def select_emotion_state():
    """根据概率选择情绪状态"""
    rand = random.random()
    if rand < 0.6:
        return 'normal'
    elif rand < 0.85:
        return 'anxious'
    else:
        return 'depressed'

def generate_physiological_record(student_id, base_date):
    """生成单条生理数据记录"""
    days_offset = random.randint(0, 90)
    record_date = base_date + timedelta(days=days_offset)

    emotion_state = select_emotion_state()

    if emotion_state == 'normal':
        heart_rate = int(np.random.normal(72, 8))
        heart_rate = max(55, min(100, heart_rate))
        sleep_hours = round(np.random.normal(7.5, 0.8), 1)
        sleep_hours = max(5.0, min(10.0, sleep_hours))
        activity_level = random.randint(5000, 15000)
        stress_score = random.randint(20, 50)
    elif emotion_state == 'anxious':
        heart_rate = int(np.random.normal(85, 10))
        heart_rate = max(65, min(115, heart_rate))
        sleep_hours = round(np.random.normal(6.5, 1.0), 1)
        sleep_hours = max(4.0, min(9.0, sleep_hours))
        activity_level = random.randint(3000, 8000)
        stress_score = random.randint(60, 90)
    else:
        heart_rate = int(np.random.normal(65, 8))
        heart_rate = max(50, min(90, heart_rate))
        sleep_hours = round(np.random.normal(9.0, 1.5), 1)
        sleep_hours = max(6.0, min(12.0, sleep_hours))
        activity_level = random.randint(2000, 6000)
        stress_score = random.randint(40, 70)

    blood_oxygen = round(np.random.normal(97.5, 1.0), 1)
    blood_oxygen = max(94.0, min(99.5, blood_oxygen))

    steps = activity_level
    calories = int(activity_level * 0.04 + np.random.normal(0, 50))
    standing_hours = round(np.random.uniform(8, 14), 1)
    screen_time = round(np.random.uniform(2, 8), 1)

    record_hour = random.randint(8, 20)
    measurement_time = record_date.replace(hour=record_hour, minute=random.randint(0, 59), second=0)

    return {
        'student_id': student_id,
        'record_date': record_date.strftime('%Y-%m-%d'),
        'measurement_time': measurement_time.strftime('%Y-%m-%d %H:%M:%S'),
        'heart_rate': heart_rate,
        'sleep_hours': sleep_hours,
        'activity_level': activity_level,
        'steps': steps,
        'calories': calories,
        'blood_oxygen': blood_oxygen,
        'standing_hours': standing_hours,
        'screen_time': screen_time,
        'stress_score': stress_score,
        'emotion_state': emotion_state
    }

def collect_physiological_data():
    """采集生理数据"""
    print("开始采集生理数据...")

    students = generate_student_ids(STUDENT_COUNT)
    base_date = datetime(2025, 9, 1)
    records = []

    for student_id in students:
        for _ in range(RECORDS_PER_STUDENT):
            record = generate_physiological_record(student_id, base_date)
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(['student_id', 'record_date']).reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'physiological_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"生理数据采集完成，共 {len(df)} 条记录")
    print(f"学生数量: {df['student_id'].nunique()}")
    print(f"正常状态: {len(df[df['emotion_state'] == 'normal'])} 条")
    print(f"焦虑状态: {len(df[df['emotion_state'] == 'anxious'])} 条")
    print(f"抑郁状态: {len(df[df['emotion_state'] == 'depressed'])} 条")
    print(f"数据已保存至: {output_path}")

    return df

if __name__ == '__main__':
    collect_physiological_data()
