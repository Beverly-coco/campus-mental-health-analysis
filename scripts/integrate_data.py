"""

数据整合脚本
功能：将三类数据按学生ID关联，存入MySQL数据库
使用批量插入提高效率
"""

import pandas as pd
import pymysql
import os
from datetime import datetime

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'YOUR_PASSWORD',
    'charset': 'utf8mb4'
}
DB_NAME = 'mental_health_db'
USE_MYSQL = False

def load_db_config():
    """从配置文件加载数据库配置"""
    global DB_CONFIG, USE_MYSQL
    config_path = os.path.join(os.path.dirname(__file__), 'db_config.py')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            exec(f.read(), globals())
        USE_MYSQL = True
        print("已加载MySQL配置")

def create_database():
    """创建数据库"""
    if not USE_MYSQL:
        return
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    cursor.execute(f"USE {DB_NAME}")
    conn.commit()
    cursor.close()
    conn.close()
    print(f"数据库 {DB_NAME} 创建/已存在")

def create_tables():
    """创建数据表"""
    if not USE_MYSQL:
        return
    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS behavioral_data")
    cursor.execute("DROP TABLE IF EXISTS text_data")
    cursor.execute("DROP TABLE IF EXISTS physiological_data")
    cursor.execute("DROP TABLE IF EXISTS students")

    cursor.execute("""
        CREATE TABLE students (
            student_id VARCHAR(10) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cursor.execute("""
        CREATE TABLE behavioral_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(10),
            record_date DATE,
            record_type VARCHAR(20),
            location VARCHAR(50),
            amount DECIMAL(10, 2),
            frequency INT,
            access_time DATETIME,
            weekday INT,
            INDEX idx_student_id (student_id),
            INDEX idx_record_date (record_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cursor.execute("""
        CREATE TABLE text_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(10),
            post_time DATETIME,
            platform VARCHAR(50),
            text_content TEXT,
            emotion_label VARCHAR(20),
            INDEX idx_student_id (student_id),
            INDEX idx_post_time (post_time),
            INDEX idx_emotion_label (emotion_label)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    cursor.execute("""
        CREATE TABLE physiological_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(10),
            record_date DATE,
            measurement_time DATETIME,
            heart_rate INT,
            sleep_hours DECIMAL(4, 1),
            activity_level INT,
            steps INT,
            calories INT,
            blood_oxygen DECIMAL(4, 1),
            standing_hours DECIMAL(4, 1),
            screen_time DECIMAL(4, 1),
            stress_score INT,
            emotion_state VARCHAR(20),
            INDEX idx_student_id (student_id),
            INDEX idx_record_date (record_date),
            INDEX idx_emotion_state (emotion_state)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("数据表创建完成")

def batch_insert(conn, table_name, columns, data_list, batch_size=500):
    """批量插入数据"""
    if not data_list:
        return 0

    placeholders = ', '.join(['%s'] * len(columns))
    sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"

    cursor = conn.cursor()
    total_inserted = 0

    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        cursor.executemany(sql, batch)
        total_inserted += len(batch)
        conn.commit()
        print(f"  已插入 {total_inserted}/{len(data_list)} 条")

    cursor.close()
    return total_inserted

def insert_students(df_list):
    """插入学生信息"""
    if not USE_MYSQL:
        return
    student_ids = set()
    for df in df_list:
        student_ids.update(df['student_id'].unique())

    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE students")

    data_list = [(sid,) for sid in sorted(student_ids)]
    cursor.executemany("INSERT INTO students (student_id) VALUES (%s)", data_list)
    conn.commit()
    cursor.close()
    conn.close()
    print(f"学生表插入完成，共 {len(student_ids)} 名学生")

def insert_behavioral_data(df):
    """批量插入行为数据"""
    if not USE_MYSQL:
        return
    print("开始插入行为数据...")
    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)

    data_list = [
        (row['student_id'], row['record_date'], row['record_type'], row['location'],
         float(row['amount']), int(row['frequency']), row['access_time'], int(row['weekday']))
        for _, row in df.iterrows()
    ]

    columns = ['student_id', 'record_date', 'record_type', 'location', 'amount', 'frequency', 'access_time', 'weekday']
    count = batch_insert(conn, 'behavioral_data', columns, data_list)
    conn.close()
    print(f"行为数据表插入完成，共 {count} 条记录")

def insert_text_data(df):
    """批量插入文本数据"""
    if not USE_MYSQL:
        return
    print("开始插入文本数据...")
    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)

    data_list = [
        (row['student_id'], row['post_time'], row['platform'], row['text_content'], row['emotion_label'])
        for _, row in df.iterrows()
    ]

    columns = ['student_id', 'post_time', 'platform', 'text_content', 'emotion_label']
    count = batch_insert(conn, 'text_data', columns, data_list)
    conn.close()
    print(f"文本数据表插入完成，共 {count} 条记录")

def insert_physiological_data(df):
    """批量插入生理数据"""
    if not USE_MYSQL:
        return
    print("开始插入生理数据...")
    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)

    data_list = [
        (row['student_id'], row['record_date'], row['measurement_time'],
         int(row['heart_rate']), float(row['sleep_hours']), int(row['activity_level']),
         int(row['steps']), int(row['calories']), float(row['blood_oxygen']),
         float(row['standing_hours']), float(row['screen_time']),
         int(row['stress_score']), row['emotion_state'])
        for _, row in df.iterrows()
    ]

    columns = ['student_id', 'record_date', 'measurement_time', 'heart_rate', 'sleep_hours',
               'activity_level', 'steps', 'calories', 'blood_oxygen', 'standing_hours',
               'screen_time', 'stress_score', 'emotion_state']
    count = batch_insert(conn, 'physiological_data', columns, data_list)
    conn.close()
    print(f"生理数据表插入完成，共 {count} 条记录")

def create_csv_backup(behavioral_df, text_df, physiological_df):
    """创建CSV备份"""
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'integrated')
    os.makedirs(output_dir, exist_ok=True)

    students_df = pd.DataFrame({'student_id': pd.concat([behavioral_df['student_id'], text_df['student_id'], physiological_df['student_id']]).unique()})
    students_df.to_csv(os.path.join(output_dir, 'students.csv'), index=False, encoding='utf-8-sig')
    behavioral_df.to_csv(os.path.join(output_dir, 'behavioral_data.csv'), index=False, encoding='utf-8-sig')
    text_df.to_csv(os.path.join(output_dir, 'text_data.csv'), index=False, encoding='utf-8-sig')
    physiological_df.to_csv(os.path.join(output_dir, 'physiological_data.csv'), index=False, encoding='utf-8-sig')

    print(f"CSV备份已保存至: {output_dir}")

def verify_data():
    """验证数据库中的数据"""
    if not USE_MYSQL:
        return
    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)
    cursor = conn.cursor()

    tables = ['students', 'behavioral_data', 'text_data', 'physiological_data']
    print("\n数据库验证:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} 条记录")

    cursor.close()
    conn.close()

def integrate_data():
    """整合所有数据"""
    print("=" * 60)
    print("开始数据整合...")
    print("=" * 60)

    load_db_config()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

    print("\n读取数据文件...")
    behavioral_df = pd.read_csv(os.path.join(data_dir, 'behavioral_data.csv'))
    text_df = pd.read_csv(os.path.join(data_dir, 'text_data.csv'))
    physiological_df = pd.read_csv(os.path.join(data_dir, 'physiological_data.csv'))

    print(f"行为数据: {len(behavioral_df)} 条")
    print(f"文本数据: {len(text_df)} 条")
    print(f"生理数据: {len(physiological_df)} 条")

    if USE_MYSQL:
        print("\n创建数据库和表...")
        create_database()
        create_tables()

        print("\n插入数据...")
        insert_students([behavioral_df, text_df, physiological_df])
        insert_behavioral_data(behavioral_df)
        insert_text_data(text_df)
        insert_physiological_data(physiological_df)

        print("\n验证数据...")
        verify_data()
    else:
        print("\n[跳过] MySQL未配置，创建CSV备份...")
        create_csv_backup(behavioral_df, text_df, physiological_df)

    print("\n" + "=" * 60)
    print("数据整合完成！")
    print("=" * 60)

if __name__ == '__main__':
    integrate_data()
