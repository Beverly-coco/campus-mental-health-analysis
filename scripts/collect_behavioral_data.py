"""
行为数据采集脚本
功能：从校园一卡通系统获取行为数据（模拟），包括消费频次、消费金额、出入图书馆/宿舍时间
数据量：≥5000条记录
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

STUDENT_COUNT = 500
RECORDS_PER_STUDENT = 10
TOTAL_RECORDS = STUDENT_COUNT * RECORDS_PER_STUDENT

CONSUMPTION_LOCATIONS = ['食堂一楼', '食堂二楼', '食堂三楼', '校园超市', '浴室', '开水房', '快递点']
GATE_TYPES = ['图书馆', '宿舍楼']

def generate_student_ids(count):
    """生成学生ID列表"""
    return [f'S{str(i).zfill(6)}' for i in range(1, count + 1)]

def generate_behavioral_record(student_id, base_date):
    """生成单条行为记录"""
    days_offset = random.randint(0, 90)
    record_date = base_date + timedelta(days=days_offset)

    is_consumption = random.random() < 0.7
    if is_consumption:
        location = random.choice(CONSUMPTION_LOCATIONS)
        if location in ['食堂一楼', '食堂二楼', '食堂三楼']:
            amount = round(np.random.lognormal(3.2, 0.5), 2)
        elif location == '校园超市':
            amount = round(np.random.lognormal(3.0, 0.8), 2)
        else:
            amount = round(random.uniform(0.5, 5.0), 2)
        record_type = 'consumption'
        frequency = random.randint(1, 3)
    else:
        location = random.choice(GATE_TYPES)
        amount = 0.0
        record_type = 'access'
        frequency = 1

    if location == '图书馆':
        access_time = record_date.replace(hour=random.randint(8, 21), minute=random.randint(0, 59), second=0)
    else:
        access_time = record_date.replace(hour=random.randint(6, 23), minute=random.randint(0, 59), second=0)

    return {
        'student_id': student_id,
        'record_date': record_date.strftime('%Y-%m-%d'),
        'record_type': record_type,
        'location': location,
        'amount': amount,
        'frequency': frequency,
        'access_time': access_time.strftime('%Y-%m-%d %H:%M:%S'),
        'weekday': record_date.weekday()
    }

def collect_behavioral_data():
    """采集行为数据"""
    print("开始采集行为数据...")

    students = generate_student_ids(STUDENT_COUNT)
    base_date = datetime(2025, 9, 1)
    records = []

    for student_id in students:
        for _ in range(RECORDS_PER_STUDENT):
            record = generate_behavioral_record(student_id, base_date)
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(['student_id', 'record_date']).reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'behavioral_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"行为数据采集完成，共 {len(df)} 条记录")
    print(f"学生数量: {df['student_id'].nunique()}")
    print(f"消费记录数: {len(df[df['record_type'] == 'consumption'])}")
    print(f"门禁记录数: {len(df[df['record_type'] == 'access'])}")
    print(f"数据已保存至: {output_path}")

    return df

if __name__ == '__main__':
    collect_behavioral_data()
