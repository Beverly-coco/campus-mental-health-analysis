# 校园心理健康智能评估与预警系统

> 西华大学智能科学与技术学院 · 毕业设计

基于多模态数据融合与深度学习的校园心理健康评估系统，整合学生的行为数据、社交媒体文本和生理指标，构建512维融合特征向量，实现对学生心理状态的智能分类与早期预警。

---

## 系统架构

```
┌──────────────┐   ┌────────────────┐   ┌────────────────────┐
│   行为数据    │   │     文本数据     │   │      生理数据       │
│  (20 维特征)  │   │   (229 维特征)  │   │    (53 维特征)     │
└──────┬───────┘   └───────┬────────┘   └──────────┬─────────┘
       │                    │                        │
       ▼                    ▼                        ▼
┌──────────────┐   ┌────────────────┐   ┌────────────────────┐
│  行为特征     │   │   文本预处理     │   │     生理特征        │
│   提取工程    │   │ Word2Vec+TF-IDF │   │     提取工程        │
└──────┬───────┘   └───────┬────────┘   └──────────┬─────────┘
       │                   │                        │
       │              NLTK 分词                      │
       │            情感倾向分析                      │
       └────────────────────┘                        │
                    │                                │
                    ▼                                │
            ┌───────────────┐                        │
            │   特征融合     │                        │
            │  (512 维向量)  │◄───────────────────────┘
            └───────┬───────┘
                    │
                    ▼
            ┌───────────────┐
            │  深度学习模型   │
            │  CNN / LSTM /  │
            │     MLP       │
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │  四级心理状态   │
            │  分类预警结果   │
            └───────────────┘
```

---

## 功能特性

- **多模态数据融合**：整合行为、文本、生理三类数据，构建512维统一特征向量
- **丰富的NLP处理**：Word2Vec 词向量（100维）+ TF-IDF 关键词（120维）+ NLTK 情感分析
- **多维度生理指标**：心率、睡眠、活动量、压力评分、血氧等9项指标
- **行为轨迹追踪**：消费模式、门禁记录、社交活跃度、时段活动分析
- **加权融合策略**：文本特征（40%）+ 行为特征（30%）+ 生理特征（30%）
- **MySQL持久化存储**：支持数据库存储，CSV文件双重备份
- **一键全流程运行**：通过 `run_all.py` 一键执行数据采集、预处理、融合全流程

---

## 目录结构

```
bishe/
├── data/                          # 数据目录
│   ├── text_data.csv              # 原始文本数据（学生帖子）
│   ├── behavioral_data.csv        # 原始行为数据（消费、门禁）
│   ├── physiological_data.csv     # 原始生理数据
│   ├── preprocessed/              # 预处理后数据
│   │   ├── text_features.csv      # 文本特征（229维）
│   │   ├── behavioral_features.csv # 行为特征
│   │   ├── physiological_features.csv
│   │   ├── tfidf_vectorizer.pkl   # TF-IDF模型
│   │   └── word2vec.model        # Word2Vec词向量模型
│   └── fused/                     # 融合特征数据
│       ├── X_train.npy / X_test.npy
│       ├── y_train.npy / y_test.npy
│       ├── merged_features.csv
│       ├── feature_scaler.pkl
│       └── feature_info.pkl
├── scripts/                       # 核心脚本
│   ├── db_config.py               # MySQL数据库配置
│   ├── run_all.py                 # 全流程一键运行
│   ├── collect_behavioral_data.py # 采集行为数据
│   ├── collect_text_data.py       # 采集文本数据
│   ├── collect_physiological_data.py # 采集生理数据
│   ├── preprocess_text.py         # 文本预处理（NLP）
│   ├── preprocess_behavioral.py   # 行为特征工程
│   ├── preprocess_physiological.py # 生理特征工程
│   ├── integrate_data.py          # 数据整合入库
│   └── verify_database.py         # 数据库连接验证
├── docs/
│   └── feature_description.md     # 特征说明文档（中文）
└── requirements.txt
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：

- `pandas`、`numpy` — 数据处理
- `tensorflow` — 深度学习模型
- `nltk`、`gensim` — 自然语言处理（Word2Vec、分词）
- `scikit-learn` — TF-IDF、特征标准化、数据划分
- `mysql-connector-python` / `pymysql` — MySQL数据库
- `faker` — 模拟数据生成

### 2. 一键运行全流程

```bash
python scripts/run_all.py
```

执行步骤：

1. 采集行为数据 → 2. 采集文本数据 → 3. 采集生理数据 → 4. 整合入库 → 5. 预处理行为特征 → 6. 预处理文本特征 → 7. 预处理生理特征 → 8. 融合特征并划分数据集

### 3. 加载融合数据（训练模型）

```python
import numpy as np

X_train = np.load('data/fused/X_train.npy')
X_test  = np.load('data/fused/X_test.npy')
y_train = np.load('data/fused/y_train.npy')
y_test  = np.load('data/fused/y_test.npy')

print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
# 训练集: (350, 512), 标签: (350,)
```

### 4. 查看特征信息

```python
import pickle

with open('data/fused/feature_info.pkl', 'rb') as f:
    info = pickle.load(f)

print(f"行为特征数: {len(info['behavior_cols'])}")
print(f"文本特征数: {len(info['text_cols'])}")
print(f"生理特征数: {len(info['physiological_cols'])}")
```

---

## 数据说明

### 文本数据（text_data.csv）

| 字段            | 类型     | 说明                                    |
| --------------- | -------- | --------------------------------------- |
| `student_id`    | string   | 学生唯一标识                            |
| `post_time`     | datetime | 发帖时间                                |
| `platform`      | string   | 平台：校园论坛 / 树洞 / 匿名留言板      |
| `text_content`  | string   | 帖子内容                                |
| `emotion_label` | string   | 情感倾向：positive / negative / neutral |

### 行为数据（behavioral_data.csv）

| 字段          | 类型     | 说明                                          |
| ------------- | -------- | --------------------------------------------- |
| `student_id`  | string   | 学生唯一标识                                  |
| `record_date` | date     | 记录日期                                      |
| `record_type` | string   | 记录类型：consumption（消费）/ access（门禁） |
| `location`    | string   | 地点：食堂 / 宿舍楼 / 图书馆 / 校园超市等     |
| `amount`      | float    | 交易金额（门禁记录为0）                       |
| `frequency`   | int      | 频次                                          |
| `access_time` | datetime | 刷卡/消费时间                                 |
| `weekday`     | int      | 星期几（0=周一）                              |

### 生理数据（physiological_data.csv）

| 字段               | 类型     | 说明                                   |
| ------------------ | -------- | -------------------------------------- |
| `student_id`       | string   | 学生唯一标识                           |
| `record_date`      | date     | 记录日期                               |
| `measurement_time` | datetime | 测量时间                               |
| `heart_rate`       | float    | 心率（bpm）                            |
| `sleep_hours`      | float    | 睡眠时长（小时）                       |
| `activity_level`   | int      | 活动量                                 |
| `steps`            | int      | 步数                                   |
| `calories`         | float    | 消耗卡路里                             |
| `blood_oxygen`     | float    | 血氧饱和度（%）                        |
| `standing_hours`   | float    | 站立时长（小时）                       |
| `screen_time`      | float    | 每日屏幕使用时长（小时）               |
| `stress_score`     | float    | 压力评分（0-100）                      |
| `emotion_state`    | string   | 情绪状态：normal / anxious / depressed |

---

## 特征工程详解

### 特征维度一览

| 数据类型 | 维度数                                   | 融合权重 | 融合后维度 |
| -------- | ---------------------------------------- | -------- | ---------- |
| 行为特征 | 20                                       | 30%      | 84         |
| 文本特征 | 229（9基础 + 100 Word2Vec + 120 TF-IDF） | 40%      | 313        |
| 生理特征 | 53（45统计量 + 5正常比例 + 3情绪比例）   | 30%      | 116        |
| **合计** | **303**                                  | **100%** | **512**    |

### 融合策略

```
融合向量 = concat(
    行为特征 × 填充系数(84/20),
    文本特征 × 填充系数(313/229),
    生理特征 × 填充系数(116/53)
)
= 84 + 313 + 116 = 512维
```

---

## 心理状态标签定义

| 标签                  | 编码 | 判定规则                 |
| --------------------- | ---- | ------------------------ |
| 正常（normal）        | 0    | 文本积极且生理状态正常   |
| 轻度异常（mild）      | 1    | 不满足焦虑/抑郁条件      |
| 焦虑倾向（anxious）   | 2    | 文本消极或生理状态为焦虑 |
| 抑郁倾向（depressed） | 3    | 生理状态为抑郁           |

---

## 数据集划分

| 数据集 | 样本数 | 比例 |
| ------ | ------ | ---- |
| 训练集 | 350    | 70%  |
| 测试集 | 150    | 30%  |

---

## 数据库配置（可选）

如需启用MySQL存储，修改 `scripts/db_config.py` 中的配置：

```python
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'your_username',
    'password': 'your_password',
    'database': 'mental_health_db',
}
```

```bash
# 验证数据库连接
python scripts/verify_database.py
```

> 注：如不需要MySQL，可将 `integrate_data.py` 中的 `USE_MYSQL` 设为 `False`，系统将以CSV格式保存数据。

---

## 技术栈

| 类别         | 技术                                             |
| ------------ | ------------------------------------------------ |
| 语言         | Python 3.8+                                      |
| 数据处理     | pandas、numpy                                    |
| 自然语言处理 | NLTK、Gensim（Word2Vec）、scikit-learn（TF-IDF） |
| 深度学习     | TensorFlow / Keras                               |
| 数据库       | MySQL 8.0+                                       |
| 模拟数据     | Faker                                            |

---
