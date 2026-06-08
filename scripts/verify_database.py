"""
MySQL数据库验证脚本
验证多模态数据是否正确存储在MySQL数据库中
"""

import pymysql
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from db_config import DB_CONFIG, DB_NAME

def verify_database():
    """验证数据库内容"""
    print("=" * 60)
    print("MySQL数据库验证")
    print("=" * 60)

    conn = pymysql.connect(**DB_CONFIG, database=DB_NAME)
    cursor = conn.cursor()

    tables = ['students', 'behavioral_data', 'text_data', 'physiological_data']

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"{table}: {count} 条记录")

    print("\n" + "-" * 60)
    print("数据样本验证")
    print("-" * 60)

    cursor.execute("SELECT * FROM students LIMIT 3")
    print("\n学生表样本:")
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.execute("SELECT * FROM behavioral_data LIMIT 3")
    print("\n行为数据表样本:")
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.execute("SELECT * FROM text_data LIMIT 3")
    print("\n文本数据表样本:")
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.execute("SELECT * FROM physiological_data LIMIT 3")
    print("\n生理数据表样本:")
    for row in cursor.fetchall():
        print(f"  {row}")

    cursor.close()
    conn.close()

    print("\n" + "=" * 60)
    print("数据库验证完成！")
    print("=" * 60)

if __name__ == '__main__':
    verify_database()
