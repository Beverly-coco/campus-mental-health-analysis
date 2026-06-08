"""
文本数据采集脚本
功能：模拟校园论坛、匿名留言板的学生文本内容
数据量：≥3000条记录
情绪标签：积极/消极/中性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

STUDENT_COUNT = 500
RECORDS_PER_STUDENT = 6
TOTAL_RECORDS = STUDENT_COUNT * RECORDS_PER_STUDENT

POSITIVE_TEMPLATES = [
    "今天天气真好，心情也很棒！",
    "考试终于结束了，感觉发挥得不错！",
    "和朋友们一起吃饭好开心呀！",
    "运动完之后整个人都轻松了，强烈推荐大家多锻炼！",
    "收到了心仪公司的offer，太激动了！",
    "今天上课老师讲的内容很有趣，学到了很多新知识。",
    "周末和室友一起去看了电影，很放松！",
    "完成了一个月的早起打卡计划，坚持就是胜利！",
    "今天的午餐特别好吃，食堂阿姨太棒了！",
    "和同学讨论问题很有收获，团队合作很重要。",
    "今天的阳光很温暖，适合出去走走。",
    "完成了所有作业，可以好好休息一下了。",
    "图书馆的氛围很好，学习效率很高。",
    "收到了好朋友的礼物，感动！",
    "今天尝试了一道新菜，很成功！",
    "小组项目进展顺利，大家配合得很好。",
    "天气凉爽，适合跑步，感觉状态很好。",
    "今天被老师表扬了，有点小骄傲！",
    "解决了困扰我好久的问题，太开心了！",
    "和新同学认识聊得很愉快，交到了新朋友。"
]

NEGATIVE_TEMPLATES = [
    "今天感觉好累，什么都不想做。",
    "考试没考好，心情很低落...",
    "最近压力好大，睡不着觉。",
    "感觉好孤独，没有人理解我...",
    "对未来很迷茫，不知道该怎么办。",
    "今天和室友闹矛盾了，心情很差。",
    "感觉自己什么都做不好，好失败...",
    "最近食欲不好，什么都不想吃。",
    "期末复习太累了，脑子转不动。",
    "想家了，好想回家...",
    "今天的课好难，完全听不懂。",
    "生活费快花完了，又不好意思向家里要...",
    "人际交往好复杂，感觉好累。",
    "每天都很忙碌但又觉得没有意义...",
    "最近总是莫名的焦虑不安。",
    "被同学误会了，解释也没人听...",
    "论文写不出来，deadline又快到了。",
    "感觉和周围的人格格不入。",
    "今天摔了一跤，膝盖好疼...",
    "连续几天失眠，白天没精神。"
]

NEUTRAL_TEMPLATES = [
    "今天上了三节课，分别是高数、英语和体育。",
    "食堂今天的菜色和昨天差不多。",
    "图书馆九点闭馆，需要早点去。",
    "明天有小组会议，要准备一下PPT。",
    "这周生活费还剩一半，需要省着点花。",
    "天气预报说明天会下雨，记得带伞。",
    "周末打算去超市买点日用品。",
    "今天作业不多，可以早点休息。",
    "下个月有英语四级考试，要开始准备了。",
    "学校最近在举办运动会报名。",
    "快递到了，要去菜鸟驿站取。",
    "室友今天生日，准备买个蛋糕。",
    "下节课换到了更大的教室。",
    "手机内存不够了，需要清理一下。",
    "今天食堂推出了新菜品，排队的人很多。",
    "下周有社团活动，需要招募志愿者。",
    "耳机丢了，最近只能外放了。",
    "电脑风扇声音很大，该清灰了。",
    "图书馆二楼的自习室比较安静。",
    "明天早八有课，要定闹钟了。"
]

def generate_student_ids(count):
    """生成学生ID列表"""
    return [f'S{str(i).zfill(6)}' for i in range(1, count + 1)]

def select_emotion_label():
    """根据概率选择情绪标签"""
    rand = random.random()
    if rand < 0.3:
        return 'positive'
    elif rand < 0.6:
        return 'negative'
    else:
        return 'neutral'

def generate_text_content(emotion_label):
    """根据情绪标签生成文本内容"""
    if emotion_label == 'positive':
        templates = POSITIVE_TEMPLATES
    elif emotion_label == 'negative':
        templates = NEGATIVE_TEMPLATES
    else:
        templates = NEUTRAL_TEMPLATES

    base_text = random.choice(templates)

    extensions = [
        "", "", "", "",
        " #日常",
        " #校园生活",
        " #今天",
        " #记录",
        " 分享给大家~",
        " 希望明天会更好",
        " 给自己加油！",
        " 加油！"
    ]

    return base_text + random.choice(extensions)

def generate_text_record(student_id, base_date):
    """生成单条文本记录"""
    days_offset = random.randint(0, 90)
    hours_offset = random.randint(0, 23)
    minutes_offset = random.randint(0, 59)
    record_time = base_date + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)

    emotion_label = select_emotion_label()
    text_content = generate_text_content(emotion_label)

    platform_choice = random.choices(
        ['校园论坛', '匿名留言板', '树洞'],
        weights=[0.5, 0.3, 0.2]
    )[0]

    return {
        'student_id': student_id,
        'post_time': record_time.strftime('%Y-%m-%d %H:%M:%S'),
        'platform': platform_choice,
        'text_content': text_content,
        'emotion_label': emotion_label
    }

def collect_text_data():
    """采集文本数据"""
    print("开始采集文本数据...")

    students = generate_student_ids(STUDENT_COUNT)
    base_date = datetime(2025, 9, 1)
    records = []

    for student_id in students:
        for _ in range(RECORDS_PER_STUDENT):
            record = generate_text_record(student_id, base_date)
            records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values(['student_id', 'post_time']).reset_index(drop=True)

    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'text_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print(f"文本数据采集完成，共 {len(df)} 条记录")
    print(f"学生数量: {df['student_id'].nunique()}")
    print(f"积极情绪: {len(df[df['emotion_label'] == 'positive'])} 条")
    print(f"消极情绪: {len(df[df['emotion_label'] == 'negative'])} 条")
    print(f"中性情绪: {len(df[df['emotion_label'] == 'neutral'])} 条")
    print(f"数据已保存至: {output_path}")

    return df

if __name__ == '__main__':
    collect_text_data()
