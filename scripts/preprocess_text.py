"""
文本数据预处理脚本
功能：用NLTK进行分词、去停用词，通过Word2Vec将文本转化为词向量，结合TF-IDF提取情绪关键词
"""

import pandas as pd
import numpy as np
import os
import re
import pickle

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

np.random.seed(42)

STOPWORDS = set(stopwords.words('chinese')) if 'chinese' in stopwords.fileids() else set()

CHINESE_STOPWORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
    '自己', '这', '那', '她', '他', '它', '们', '什么', '怎么', '这个', '那个',
    '啊', '吧', '呢', '哦', '嗯', '哈', '呀', '嘛', '啦', '噢', '嗨',
    '今天', '昨天', '明天', '现在', '然后', '所以', '但是', '因为', '如果',
    '可以', '没', '还是', '这样', '那样', '怎么', '为什么', '吗', '吧', '呢',
    '一下', '一点', '有点', '太', '真', '好', '很', '非常', '特别', '比较',
    '日常', '校园生活', '记录', '分享', '大家', '给'
])

ALL_STOPWORDS = STOPWORDS.union(CHINESE_STOPWORDS)

def load_text_data():
    """加载文本数据"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'text_data.csv')
    df = pd.read_csv(data_path)
    return df

def clean_text(text):
    """清洗文本：去除特殊字符、标点"""
    if pd.isna(text):
        return ""

    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+#', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def nltk_tokenize(text):
    """使用NLTK进行分词"""
    text = clean_text(text)
    if not text:
        return []

    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]+')
    english_pattern = re.compile(r'[a-zA-Z]+')

    words = []
    for match in chinese_pattern.finditer(text):
        word = match.group()
        if len(word) >= 2:
            words.append(word)

    for match in english_pattern.finditer(text):
        words.append(match.group().lower())

    return words

def nltk_remove_stopwords(words):
    """使用NLTK去停用词"""
    return [w for w in words if w and len(w) > 1 and w.lower() not in ALL_STOPWORDS]

def preprocess_text(text):
    """预处理文本：NLTK分词 + 去停用词"""
    words = nltk_tokenize(text)
    words = nltk_remove_stopwords(words)
    return words

def build_word2vec_model(tokenized_texts):
    """训练Word2Vec模型"""
    print("训练Word2Vec模型...")
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=10,
        seed=42
    )

    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed', 'word2vec.model')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Word2Vec模型已保存，词汇量: {len(model.wv)}")

    return model

def text_to_vector(text, model, vector_size=100):
    """将文本转换为词向量"""
    words = preprocess_text(text)
    if not words:
        return np.zeros(vector_size)

    vectors = []
    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])

    if not vectors:
        return np.zeros(vector_size)

    return np.mean(vectors, axis=0)

def build_tfidf_vectorizer(tokenized_texts):
    """训练TF-IDF向量化器"""
    print("训练TF-IDF向量化器...")
    texts_str = [' '.join(tokens) for tokens in tokenized_texts]

    vectorizer = TfidfVectorizer(
        max_features=200,
        min_df=2,
        max_df=0.95
    )

    tfidf_matrix = vectorizer.fit_transform(texts_str)

    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed', 'tfidf_vectorizer.pkl')
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"TF-IDF向量化器已保存，特征维度: {tfidf_matrix.shape[1]}")

    return vectorizer, tfidf_matrix

def extract_text_features(df, word2vec_model, tfidf_vectorizer):
    """提取文本特征"""
    print("提取文本特征...")

    features_list = []

    for student_id in df['student_id'].unique():
        student_df = df[df['student_id'] == student_id]
        student_texts = student_df['text_content'].tolist()
        emotion_labels = student_df['emotion_label'].tolist()

        all_tokens = []
        word2vec_features = []

        for text in student_texts:
            tokens = preprocess_text(text)
            all_tokens.extend(tokens)
            vec = text_to_vector(text, word2vec_model, vector_size=100)
            word2vec_features.append(vec)

        if word2vec_features:
            avg_word2vec = np.mean(word2vec_features, axis=0)
        else:
            avg_word2vec = np.zeros(100)

        if all_tokens:
            combined_text = ' '.join(all_tokens)
            tfidf_vec = tfidf_vectorizer.transform([combined_text]).toarray()[0]
        else:
            tfidf_vec = np.zeros(200)

        pos_count = emotion_labels.count('positive')
        neg_count = emotion_labels.count('negative')
        neu_count = emotion_labels.count('neutral')
        total = len(emotion_labels)

        features = {
            'student_id': student_id,
            'text_count': total,
            'total_words': len(all_tokens),
            'avg_words_per_text': len(all_tokens) / max(total, 1),
            'positive_count': pos_count,
            'negative_count': neg_count,
            'neutral_count': neu_count,
            'positive_ratio': pos_count / max(total, 1),
            'negative_ratio': neg_count / max(total, 1),
            'neutral_ratio': neu_count / max(total, 1)
        }

        for i, val in enumerate(avg_word2vec):
            features[f'w2v_{i}'] = val

        for i, val in enumerate(tfidf_vec):
            features[f'tfidf_{i}'] = val

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f"文本特征提取完成，共 {len(features_df)} 名学生的 {len(features_df.columns) - 1} 个特征")

    return features_df

def preprocess_text_data():
    """预处理文本数据"""
    print("=" * 50)
    print("开始文本数据预处理（NLTK分词+Word2Vec+TF-IDF）...")
    print("=" * 50)

    print("\n1. 加载文本数据...")
    df = load_text_data()
    print(f"原始数据: {len(df)} 条记录, {df['student_id'].nunique()} 名学生")

    print("\n2. NLTK分词处理...")
    tokenized_texts = []
    for text in df['text_content']:
        tokens = preprocess_text(text)
        tokenized_texts.append(tokens)

    total_words = sum(len(t) for t in tokenized_texts)
    print(f"NLTK分词完成，共 {total_words} 个词，平均 {total_words/len(tokenized_texts):.1f} 词/条")

    print("\n3. NLTK去停用词...")
    stopword_count = sum(len(t) - len([w for w in t if w.lower() not in ALL_STOPWORDS]) for t in tokenized_texts)
    print(f"去除停用词 {stopword_count} 个")

    print("\n4. 训练Word2Vec模型...")
    word2vec_model = build_word2vec_model(tokenized_texts)

    print("\n5. 训练TF-IDF向量化器...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_vectorizer(tokenized_texts)

    print("\n6. 提取文本特征...")
    features_df = extract_text_features(df, word2vec_model, tfidf_vectorizer)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed')
    os.makedirs(output_dir, exist_ok=True)

    df_cleaned = df.copy()
    df_cleaned['tokenized_text'] = [' '.join(tokens) for tokens in tokenized_texts]
    df_cleaned.to_csv(os.path.join(output_dir, 'text_data_cleaned.csv'), index=False, encoding='utf-8-sig')
    features_df.to_csv(os.path.join(output_dir, 'text_features.csv'), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print("文本数据预处理完成！")
    print(f"清洗后数据: {os.path.join(output_dir, 'text_data_cleaned.csv')}")
    print(f"文本特征: {os.path.join(output_dir, 'text_features.csv')}")
    print(f"特征数量: {len(features_df.columns) - 1} 个")
    print("  - 基础特征: 9 个")
    print("  - Word2Vec特征: 100 维（NLTK分词）")
    print("  - TF-IDF特征: 200 维")
    print("=" * 50)

    return df, features_df, word2vec_model, tfidf_vectorizer

if __name__ == '__main__':
    preprocess_text_data()
