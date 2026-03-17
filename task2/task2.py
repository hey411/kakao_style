# -*- coding: utf-8 -*-
"""
카카오스타일 데이터사이언스팀 과제
Task #2: 상품 카테고리 분류기 (Classification)
=====================================
상품명과 설명 텍스트를 기반으로 cat_1, cat_2, cat_3 카테고리를 자동 분류하는 ML 파이프라인 구현
"""

import os
import sys
import re

# stdout을 UTF-8로 강제 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
import time
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 파일로 저장
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
import joblib

# 한글 폰트 설정
try:
    font_candidates = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic', 'AppleGothic']
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    selected_font = next((f for f in font_candidates if f in available_fonts), None)
    if selected_font:
        matplotlib.rcParams['font.family'] = selected_font
    matplotlib.rcParams['axes.unicode_minus'] = False
except:
    pass

# 출력 디렉토리 설정
BASE_DIR = r'task02'
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'items.csv')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_model.pkl')

print("=" * 70)
print("한국 이커머스 상품 카테고리 분류 시스템")
print("=" * 70)

# ============================================================
# 1. 데이터 로딩
# ============================================================
print("\n[1/10] 데이터 로딩...")
t0 = time.time()
df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"  로딩 완료: {time.time()-t0:.1f}초")
print(f"  데이터 크기: {df.shape[0]:,}행 x {df.shape[1]}열")
print(f"  컬럼: {df.columns.tolist()}")

# ============================================================
# 2. 기본 EDA
# ============================================================
print("\n[2/10] 기본 탐색 (EDA)...")

print("\n  [결측치 분석]")
missing = df.isnull().sum()
for col, cnt in missing[missing > 0].items():
    print(f"    {col}: {cnt:,}건 ({cnt/len(df)*100:.1f}%)")

print("\n  [카테고리 고유값 수]")
for cat in ['cat_1', 'cat_2', 'cat_3']:
    print(f"    {cat}: {df[cat].nunique()} 종류")

print("\n  [cat_1 분포 (상위 15개)]")
cat1_counts = df['cat_1'].value_counts()
for cat, cnt in cat1_counts.head(15).items():
    bar = '#' * int(cnt / cat1_counts.max() * 30)
    print(f"    {cat:15s} {cnt:6,}건 {bar}")

# cat_1 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
cat1_counts.head(20).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('cat_1 분포 (상위 20개)', fontsize=14)
axes[0].set_xlabel('샘플 수')
axes[0].invert_yaxis()

cat1_counts.tail(14).plot(kind='barh', ax=axes[1], color='salmon')
axes[1].set_title('cat_1 분포 (하위 14개)', fontsize=14)
axes[1].set_xlabel('샘플 수')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_cat1_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig1_cat1_distribution.png")

# cat_2 분포 시각화 (상위 30개)
fig, ax = plt.subplots(figsize=(14, 10))
cat2_counts = df['cat_2'].value_counts().head(30)
cat2_counts.plot(kind='barh', ax=ax, color='teal')
ax.set_title('cat_2 분포 (상위 30개)', fontsize=13)
ax.set_xlabel('샘플 수')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_cat2_distribution.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig2_cat2_distribution.png")

# 텍스트 길이 분포
print("\n  [텍스트 길이 분포]")
df['name_len'] = df['name'].str.len().fillna(0)
df['desc_len'] = df['description'].str.len().fillna(0)
print(f"    상품명 길이: 평균 {df['name_len'].mean():.0f}, 중앙 {df['name_len'].median():.0f}, 최대 {df['name_len'].max():.0f}")
print(f"    설명 길이:   평균 {df['desc_len'].mean():.0f}, 중앙 {df['desc_len'].median():.0f}, 최대 {df['desc_len'].max():.0f}")

# ============================================================
# 3. 텍스트 전처리
# ============================================================
print("\n[3/10] 텍스트 전처리...")

HTML_TAG_RE = re.compile(r'<[^>]+>')
URL_RE = re.compile(r'https?://\S+|www\.\S+')
SPECIAL_RE = re.compile(r'[^가-힣a-zA-Z0-9\s]')
SPACE_RE = re.compile(r'\s+')

def clean_text(text):
    if not isinstance(text, str) or text == '':
        return ''
    # HTML entity
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>') \
               .replace('&nbsp;', ' ').replace('&quot;', '"')
    # HTML 태그 제거
    text = HTML_TAG_RE.sub(' ', text)
    # URL 제거
    text = URL_RE.sub(' ', text)
    # 특수문자 제거
    text = SPECIAL_RE.sub(' ', text)
    # 공백 정규화
    text = SPACE_RE.sub(' ', text).strip()
    return text

t0 = time.time()
name_list = df['name'].tolist()
desc_list = df['description'].tolist()

name_clean = [clean_text(t) for t in name_list]
desc_clean = [clean_text(t) for t in desc_list]

df['name_clean'] = name_clean
df['desc_clean'] = desc_clean
df['text_combined'] = [n + ' ' + d for n, d in zip(name_clean, desc_clean)]
elapsed = time.time() - t0
print(f"  전처리 완료: {elapsed:.1f}초 ({len(df):,}건)")
print(f"  예시 - 상품명 전처리 전: {df['name'].iloc[0][:60]}")
print(f"  예시 - 상품명 전처리 후: {df['name_clean'].iloc[0][:60]}")
print(f"  예시 - 설명 전처리 후(100자): {df['desc_clean'].iloc[0][:100]}")

# 전처리 후 텍스트 길이
df['combined_len'] = df['text_combined'].str.len()
print(f"\n  통합 텍스트 길이: 평균 {df['combined_len'].mean():.0f}, 중앙 {df['combined_len'].median():.0f}")

# 빈 설명 통계
empty_desc = (df['desc_clean'].str.len() < 5).sum()
print(f"  설명이 거의 없는 상품: {empty_desc:,}건 ({empty_desc/len(df)*100:.1f}%)")

# ============================================================
# 4. 데이터 분할
# ============================================================
print("\n[4/10] 데이터 분할 (train 80% / test 20%)...")

# cat_1 분류용 (전체 사용)
df_cat1 = df[df['cat_1'].notna()].copy()

# cat_2 분류용 (cat_2가 있는 것만)
df_cat2 = df[df['cat_2'].notna()].copy()

# cat_3 분류용 (cat_3가 있는 것만)
df_cat3 = df[df['cat_3'].notna()].copy()

print(f"  cat_1 분류용 데이터: {len(df_cat1):,}건")
print(f"  cat_2 분류용 데이터: {len(df_cat2):,}건")
print(f"  cat_3 분류용 데이터: {len(df_cat3):,}건")

# cat_1 train/test split
X_cat1 = df_cat1['text_combined'].values
y_cat1 = df_cat1['cat_1'].values
X_name_cat1 = df_cat1['name_clean'].values

X_train_c1, X_test_c1, y_train_c1, y_test_c1 = train_test_split(
    X_cat1, y_cat1, test_size=0.2, random_state=42, stratify=y_cat1)
X_name_train_c1, X_name_test_c1, _, _ = train_test_split(
    X_name_cat1, y_cat1, test_size=0.2, random_state=42, stratify=y_cat1)

# cat_2 train/test split (샘플 2개 미만 클래스 제거)
cat2_counts_all = df_cat2['cat_2'].value_counts()
valid_cat2 = cat2_counts_all[cat2_counts_all >= 2].index
df_cat2 = df_cat2[df_cat2['cat_2'].isin(valid_cat2)]
X_cat2 = df_cat2['text_combined'].values
y_cat2 = df_cat2['cat_2'].values
X_train_c2, X_test_c2, y_train_c2, y_test_c2 = train_test_split(
    X_cat2, y_cat2, test_size=0.2, random_state=42, stratify=y_cat2)

# cat_3 train/test split (샘플 2개 미만 클래스 제거)
cat3_counts = pd.Series(df_cat3['cat_3']).value_counts()
valid_cat3 = cat3_counts[cat3_counts >= 2].index
df_cat3_filt = df_cat3[df_cat3['cat_3'].isin(valid_cat3)]
X_cat3 = df_cat3_filt['text_combined'].values
y_cat3 = df_cat3_filt['cat_3'].values
X_train_c3, X_test_c3, y_train_c3, y_test_c3 = train_test_split(
    X_cat3, y_cat3, test_size=0.2, random_state=42, stratify=y_cat3)

print(f"\n  Train/Test 분할:")
print(f"    cat_1: 훈련 {len(X_train_c1):,} / 테스트 {len(X_test_c1):,}")
print(f"    cat_2: 훈련 {len(X_train_c2):,} / 테스트 {len(X_test_c2):,}")
print(f"    cat_3: 훈련 {len(X_train_c3):,} / 테스트 {len(X_test_c3):,}")

# ============================================================
# 5. 베이스라인 모델
# ============================================================
print("\n[5/10] 베이스라인 모델 학습...")

TFIDF_PARAMS = dict(
    max_features=50000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    analyzer='word'
)

results = []  # 모든 실험 결과 저장

def train_eval(name, pipeline, X_train, X_test, y_train, y_test, cat_level):
    """모델 학습 및 평가"""
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t_start
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"    [{cat_level}] {name}")
    print(f"      Accuracy: {acc:.4f} | F1-macro: {f1_macro:.4f} | F1-weighted: {f1_weighted:.4f} | 시간: {train_time:.1f}s")
    results.append({
        '모델명': name,
        '카테고리': cat_level,
        '정확도(Accuracy)': round(acc, 4),
        'F1-macro': round(f1_macro, 4),
        'F1-weighted': round(f1_weighted, 4),
        '학습시간(s)': round(train_time, 1),
        'pipeline': pipeline,
        'y_pred': y_pred,
        'y_test': y_test
    })
    return pipeline, y_pred

print("\n  ① TF-IDF (상품명만) + Logistic Regression")
pipe_lr_name = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=-1))
])
train_eval('LR (상품명만)', pipe_lr_name, X_name_train_c1, X_name_test_c1, y_train_c1, y_test_c1, 'cat_1')

print("\n  ② TF-IDF (통합 텍스트) + Logistic Regression")
pipe_lr = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=-1))
])
train_eval('LR (통합텍스트)', pipe_lr, X_train_c1, X_test_c1, y_train_c1, y_test_c1, 'cat_1')

print("\n  ③ TF-IDF (통합 텍스트) + Naive Bayes")
pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', MultinomialNB(alpha=0.1))
])
train_eval('Naive Bayes', pipe_nb, X_train_c1, X_test_c1, y_train_c1, y_test_c1, 'cat_1')

print("\n  ④ TF-IDF (통합 텍스트) + LinearSVC")
pipe_svc = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LinearSVC(C=1.0, max_iter=2000))
])
train_eval('LinearSVC', pipe_svc, X_train_c1, X_test_c1, y_train_c1, y_test_c1, 'cat_1')

# cat_2 모델 (최적 모델 적용)
print("\n  ⑤ cat_2 분류: LinearSVC (통합텍스트)")
pipe_svc_c2 = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LinearSVC(C=1.0, max_iter=2000))
])
train_eval('LinearSVC', pipe_svc_c2, X_train_c2, X_test_c2, y_train_c2, y_test_c2, 'cat_2')

# cat_3 모델
print("\n  ⑥ cat_3 분류: LinearSVC (통합텍스트)")
pipe_svc_c3 = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LinearSVC(C=1.0, max_iter=2000))
])
train_eval('LinearSVC', pipe_svc_c3, X_train_c3, X_test_c3, y_train_c3, y_test_c3, 'cat_3')

# 결과 테이블 출력
results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('pipeline','y_pred','y_test')} for r in results])
print("\n  [베이스라인 결과 비교]")
print(results_df.to_string(index=False))

# ============================================================
# 6. 성능 평가 시각화
# ============================================================
print("\n[6/10] 성능 평가 & 시각화...")

# 모델별 정확도 비교
cat1_res = results_df[results_df['카테고리'] == 'cat_1'].copy()
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

models = cat1_res['모델명'].tolist()
acc_vals = cat1_res['정확도(Accuracy)'].tolist()
f1_vals = cat1_res['F1-macro'].tolist()
x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, acc_vals, width, label='Accuracy', color='steelblue', alpha=0.85)
axes[0].bar(x + width/2, f1_vals, width, label='F1-macro', color='darkorange', alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=20, ha='right')
axes[0].set_ylim(0, 1.0)
axes[0].set_title('cat_1 베이스라인 모델 성능 비교', fontsize=13)
axes[0].legend()
axes[0].set_ylabel('Score')
for i, (a, f) in enumerate(zip(acc_vals, f1_vals)):
    axes[0].text(i - width/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
    axes[0].text(i + width/2, f + 0.01, f'{f:.3f}', ha='center', fontsize=9)

# 카테고리 레벨별 성능
cat_levels = ['cat_1', 'cat_2', 'cat_3']
svc_results = []
for cat in cat_levels:
    row = results_df[(results_df['카테고리'] == cat) & (results_df['모델명'] == 'LinearSVC')]
    if len(row):
        svc_results.append(row.iloc[0])

if svc_results:
    svc_df = pd.DataFrame(svc_results)
    x2 = np.arange(len(svc_df))
    axes[1].bar(x2 - width/2, svc_df['정확도(Accuracy)'], width, label='Accuracy', color='steelblue', alpha=0.85)
    axes[1].bar(x2 + width/2, svc_df['F1-macro'], width, label='F1-macro', color='darkorange', alpha=0.85)
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(svc_df['카테고리'], fontsize=12)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_title('LinearSVC - 카테고리 레벨별 성능', fontsize=13)
    axes[1].legend()
    axes[1].set_ylabel('Score')
    for i, row in enumerate(svc_df.itertuples()):
        axes[1].text(i - width/2, row._3 + 0.01, f'{row._3:.3f}', ha='center', fontsize=10)
        axes[1].text(i + width/2, row._4 + 0.01, f'{row._4:.3f}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_model_comparison.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig3_model_comparison.png")

# 최고 성능 모델 선택 (cat_1 기준)
best_cat1 = results_df[results_df['카테고리'] == 'cat_1'].nlargest(1, '정확도(Accuracy)').iloc[0]
best_model_name = best_cat1['모델명']
best_result = next(r for r in results if r['카테고리'] == 'cat_1' and r['모델명'] == best_model_name)
y_pred_best = best_result['y_pred']
y_test_best = best_result['y_test']
print(f"\n  cat_1 최고 모델: {best_model_name} (Accuracy={best_cat1['정확도(Accuracy)']:.4f})")

# classification report
print(f"\n  [분류 리포트 - cat_1 {best_model_name}]")
print(classification_report(y_test_best, y_pred_best, zero_division=0))

# 혼동 행렬
classes = sorted(set(y_test_best))
cm = confusion_matrix(y_test_best, y_pred_best, labels=classes)
fig, ax = plt.subplots(figsize=(20, 16))
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(classes)))
ax.set_yticks(range(len(classes)))
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
ax.set_yticklabels(classes, fontsize=8)
ax.set_title(f'혼동 행렬 (정규화) - cat_1 {best_model_name}', fontsize=14)
ax.set_xlabel('예측값')
ax.set_ylabel('실제값')
# 셀에 값 표시 (높은 것만)
for i in range(len(classes)):
    for j in range(len(classes)):
        val = cm_norm[i, j]
        if val > 0.1:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color='white' if val > 0.5 else 'black', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_confusion_matrix.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig4_confusion_matrix.png")

# 클래스별 F1 점수 분석
report_dict = classification_report(y_test_best, y_pred_best, output_dict=True, zero_division=0)
f1_by_class = {k: v['f1-score'] for k, v in report_dict.items() if k not in ('accuracy','macro avg','weighted avg')}
f1_series = pd.Series(f1_by_class).sort_values()
fig, ax = plt.subplots(figsize=(14, 8))
colors = ['salmon' if v < 0.7 else 'steelblue' for v in f1_series.values]
f1_series.plot(kind='barh', ax=ax, color=colors)
ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, label='F1=0.7 기준선')
ax.set_title(f'cat_1 클래스별 F1 Score ({best_model_name})', fontsize=13)
ax.set_xlabel('F1 Score')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_f1_by_class.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig5_f1_by_class.png")

# ============================================================
# 7. 오류 분석
# ============================================================
print("\n[7/10] 오류 분석...")

# 오분류 샘플 추출
test_indices = df_cat1.index[-len(y_test_c1):]
_, test_idx = train_test_split(df_cat1.index, test_size=0.2, random_state=42, stratify=y_cat1)

df_test_cat1 = df_cat1.loc[test_idx].copy()
df_test_cat1['y_pred'] = y_pred_best
df_test_cat1['is_wrong'] = df_test_cat1['cat_1'] != df_test_cat1['y_pred']
wrong_df = df_test_cat1[df_test_cat1['is_wrong']].copy()

print(f"\n  오분류 통계:")
print(f"    전체 테스트: {len(df_test_cat1):,}건")
print(f"    오분류: {len(wrong_df):,}건 ({len(wrong_df)/len(df_test_cat1)*100:.1f}%)")
print(f"    정분류: {len(df_test_cat1)-len(wrong_df):,}건 ({(len(df_test_cat1)-len(wrong_df))/len(df_test_cat1)*100:.1f}%)")

# 오분류 쌍 분석
error_pairs = wrong_df.groupby(['cat_1', 'y_pred']).size().reset_index(name='count')
error_pairs = error_pairs.sort_values('count', ascending=False)
print("\n  [주요 오분류 쌍 (상위 15개)]")
print(f"  {'실제 카테고리':20s}  {'예측 카테고리':20s}  {'건수':>6s}")
print("  " + "-" * 52)
for _, row in error_pairs.head(15).iterrows():
    print(f"  {row['cat_1']:20s}  {row['y_pred']:20s}  {row['count']:>6,}건")

# 오분류 예시 출력
print("\n  [오분류 예시 샘플]")
for _, ep_row in error_pairs.head(5).iterrows():
    true_cat = ep_row['cat_1']
    pred_cat = ep_row['y_pred']
    samples = wrong_df[(wrong_df['cat_1'] == true_cat) & (wrong_df['y_pred'] == pred_cat)].head(2)
    print(f"\n  [{true_cat} -> {pred_cat}로 오분류]")
    for _, s in samples.iterrows():
        print(f"    상품명: {s['name'][:60]}")
        print(f"    실제: {true_cat} | 예측: {pred_cat}")

# 오분류 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 오분류 쌍 (상위 15개)
top_errors = error_pairs.head(15).copy()
top_errors['pair'] = top_errors['cat_1'] + '\n→ ' + top_errors['y_pred']
top_errors.set_index('pair')['count'].sort_values().plot(kind='barh', ax=axes[0], color='tomato')
axes[0].set_title('주요 오분류 쌍 (상위 15개)', fontsize=12)
axes[0].set_xlabel('오분류 건수')

# 카테고리별 오분류율
cat1_error_rate = wrong_df['cat_1'].value_counts() / df_test_cat1['cat_1'].value_counts()
cat1_error_rate = cat1_error_rate.dropna().sort_values(ascending=True)
cat1_error_rate.plot(kind='barh', ax=axes[1], color='salmon')
axes[1].set_title('cat_1별 오분류율', fontsize=12)
axes[1].set_xlabel('오분류율')
axes[1].axvline(x=cat1_error_rate.mean(), color='red', linestyle='--', label=f'평균: {cat1_error_rate.mean():.3f}')
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_error_analysis.png'), dpi=120, bbox_inches='tight')
plt.close()
print("\n  -> 저장: fig6_error_analysis.png")

# 텍스트 길이와 오분류 관계
df_test_cat1['text_len'] = df_test_cat1['text_combined'].str.len()
df_test_cat1['len_bin'] = pd.cut(df_test_cat1['text_len'], bins=[0, 50, 200, 500, 2000, 100000],
                                  labels=['~50', '50~200', '200~500', '500~2000', '2000+'])
error_by_len = df_test_cat1.groupby('len_bin', observed=True)['is_wrong'].mean()

fig, ax = plt.subplots(figsize=(10, 5))
error_by_len.plot(kind='bar', ax=ax, color='steelblue', alpha=0.8)
ax.set_title('텍스트 길이별 오분류율', fontsize=13)
ax.set_xlabel('텍스트 길이 구간')
ax.set_ylabel('오분류율')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for i, v in enumerate(error_by_len):
    ax.text(i, v + 0.002, f'{v:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_error_by_length.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig7_error_by_length.png")

# ============================================================
# 8. 개선 모델
# ============================================================
print("\n[8/10] 개선 모델 실험...")

improved_results = []

def train_eval_improved(name, pipeline, X_train, X_test, y_train, y_test, cat_level='cat_1'):
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t_start
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"    {name}: Acc={acc:.4f} | F1-macro={f1_macro:.4f} | 시간={train_time:.1f}s")
    improved_results.append({
        '모델명': name,
        '카테고리': cat_level,
        '정확도(Accuracy)': round(acc, 4),
        'F1-macro': round(f1_macro, 4),
        'F1-weighted': round(f1_weighted, 4),
        '학습시간(s)': round(train_time, 1),
        'pipeline': pipeline,
        'y_pred': y_pred
    })
    return pipeline, y_pred

# C 파라미터 튜닝
print("\n  [C 파라미터 튜닝 - Logistic Regression]")
for C_val in [0.1, 1.0, 5.0, 10.0]:
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('clf', LogisticRegression(C=C_val, max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=-1))
    ])
    train_eval_improved(f'LR C={C_val}', pipe, X_train_c1, X_test_c1, y_train_c1, y_test_c1)

# class_weight='balanced'
print("\n  [class_weight=balanced 적용]")
pipe_balanced = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LogisticRegression(C=5.0, max_iter=1000, class_weight='balanced',
                                solver='lbfgs', multi_class='multinomial', n_jobs=-1))
])
train_eval_improved('LR C=5 balanced', pipe_balanced, X_train_c1, X_test_c1, y_train_c1, y_test_c1)

pipe_svc_balanced = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LinearSVC(C=1.0, max_iter=2000, class_weight='balanced'))
])
train_eval_improved('SVC balanced', pipe_svc_balanced, X_train_c1, X_test_c1, y_train_c1, y_test_c1)

# 앙상블 (다수결 투표)
print("\n  [앙상블 - 다수결 투표 (LR + SVC + NB)]")
best_lr_C = max(improved_results, key=lambda x: x['정확도(Accuracy)'] if 'LR C=' in x['모델명'] else 0)
best_lr_pipe = best_lr_C['pipeline']
best_svc_pipe = pipe_svc  # 기존 SVC
best_nb_pipe = pipe_nb    # 기존 NB

pred_lr = best_lr_pipe.predict(X_test_c1)
pred_svc = best_svc_pipe.predict(X_test_c1)
pred_nb = best_nb_pipe.predict(X_test_c1)

# 다수결 투표
ensemble_preds = []
for lr, svc, nb in zip(pred_lr, pred_svc, pred_nb):
    votes = Counter([lr, svc, nb])
    ensemble_preds.append(votes.most_common(1)[0][0])
ensemble_preds = np.array(ensemble_preds)

ens_acc = accuracy_score(y_test_c1, ensemble_preds)
ens_f1 = f1_score(y_test_c1, ensemble_preds, average='macro', zero_division=0)
print(f"    앙상블 (다수결): Acc={ens_acc:.4f} | F1-macro={ens_f1:.4f}")
improved_results.append({
    '모델명': '앙상블 (다수결)',
    '카테고리': 'cat_1',
    '정확도(Accuracy)': round(ens_acc, 4),
    'F1-macro': round(ens_f1, 4),
    'F1-weighted': round(f1_score(y_test_c1, ensemble_preds, average='weighted', zero_division=0), 4),
    '학습시간(s)': 0
})

# 개선 결과 시각화
imp_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ('pipeline','y_pred')} for r in improved_results])
imp_df_sorted = imp_df.sort_values('정확도(Accuracy)', ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['gold' if v == imp_df['정확도(Accuracy)'].max() else 'steelblue' for v in imp_df_sorted['정확도(Accuracy)']]
bars = ax.barh(imp_df_sorted['모델명'], imp_df_sorted['정확도(Accuracy)'], color=colors, alpha=0.85)
ax.axvline(x=best_cat1['정확도(Accuracy)'], color='red', linestyle='--',
           label=f'베이스라인 ({best_cat1["정확도(Accuracy)"]:.4f})')
ax.set_title('개선 모델 성능 비교 (cat_1 Accuracy)', fontsize=13)
ax.set_xlabel('Accuracy')
ax.legend()
for bar, val in zip(bars, imp_df_sorted['정확도(Accuracy)']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_improved_models.png'), dpi=120, bbox_inches='tight')
plt.close()
print("\n  -> 저장: fig8_improved_models.png")

# ============================================================
# 9. 계층적 분류 (Hierarchical Classification)
# ============================================================
print("\n[9/10] 계층적 분류 (2-stage: cat_1 → cat_2)...")

# 비교를 위한 Flat cat_2 모델 (이미 학습됨)
flat_cat2_result = next(r for r in results if r['카테고리'] == 'cat_2')
flat_cat2_acc = flat_cat2_result['정확도(Accuracy)']

# Stage 1: cat_1 분류기
print("  Stage 1: cat_1 분류기 학습...")
t0 = time.time()
stage1_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
    ('clf', LinearSVC(C=1.0, max_iter=2000))
])
stage1_pipe.fit(df_cat2['text_combined'].values, df_cat2['cat_1'].values)
print(f"    완료: {time.time()-t0:.1f}s")

# Stage 2: cat_1별 cat_2 분류기
print("  Stage 2: cat_1별 cat_2 분류기 학습...")
t0 = time.time()
stage2_classifiers = {}
cat1_groups = df_cat2.groupby('cat_1')

for cat1_val, group in cat1_groups:
    cat2_counts_g = group['cat_2'].value_counts()
    if len(cat2_counts_g) < 2:
        # 단일 클래스: 항상 그 클래스 예측
        stage2_classifiers[cat1_val] = ('single', cat2_counts_g.index[0])
        continue
    # 최소 2개 이상 샘플 있는 클래스만
    valid_c2 = cat2_counts_g[cat2_counts_g >= 2].index
    grp_filt = group[group['cat_2'].isin(valid_c2)]
    if len(grp_filt) < 4 or grp_filt['cat_2'].nunique() < 2:
        stage2_classifiers[cat1_val] = ('single', cat2_counts_g.index[0])
        continue
    try:
        pipe_s2 = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=1)),
            ('clf', LinearSVC(C=1.0, max_iter=1000))
        ])
        pipe_s2.fit(grp_filt['text_combined'].values, grp_filt['cat_2'].values)
        stage2_classifiers[cat1_val] = ('model', pipe_s2)
    except Exception as e:
        stage2_classifiers[cat1_val] = ('single', cat2_counts_g.index[0])

print(f"    cat_1 그룹 수: {len(stage2_classifiers)} | 완료: {time.time()-t0:.1f}s")

# 계층적 분류 평가
print("  계층적 분류 평가...")
_, test_idx_c2 = train_test_split(range(len(df_cat2)), test_size=0.2, random_state=42,
                                   stratify=df_cat2['cat_2'].values)
X_test_hier = df_cat2.iloc[test_idx_c2]['text_combined'].values
y_test_hier = df_cat2.iloc[test_idx_c2]['cat_2'].values

# Stage 1 예측
pred_cat1_hier = stage1_pipe.predict(X_test_hier)

# Stage 2 예측
y_pred_hier = []
for text, pred_c1 in zip(X_test_hier, pred_cat1_hier):
    if pred_c1 in stage2_classifiers:
        clf_info = stage2_classifiers[pred_c1]
        if clf_info[0] == 'single':
            y_pred_hier.append(clf_info[1])
        else:
            try:
                y_pred_hier.append(clf_info[1].predict([text])[0])
            except:
                y_pred_hier.append(pred_c1 + '_unknown')
    else:
        y_pred_hier.append('unknown')

y_pred_hier = np.array(y_pred_hier)
hier_acc = accuracy_score(y_test_hier, y_pred_hier)
hier_f1 = f1_score(y_test_hier, y_pred_hier, average='macro', zero_division=0)

print(f"\n  [계층적 분류 vs Flat 분류 비교]")
print(f"    Flat LinearSVC:       Acc={flat_cat2_acc:.4f}")
print(f"    계층적 2-Stage:       Acc={hier_acc:.4f} | F1-macro={hier_f1:.4f}")

# 시각화
comparison = {
    '방식': ['Flat LinearSVC', '계층적 2-Stage'],
    'Accuracy': [flat_cat2_acc, hier_acc],
    'F1-macro': [flat_cat2_result['F1-macro'], round(hier_f1, 4)]
}
comp_df = pd.DataFrame(comparison)
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(2)
w = 0.3
ax.bar(x - w/2, comp_df['Accuracy'], w, label='Accuracy', color='steelblue', alpha=0.85)
ax.bar(x + w/2, comp_df['F1-macro'], w, label='F1-macro', color='darkorange', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(comp_df['방식'], fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title('cat_2 분류: Flat vs 계층적 분류 비교', fontsize=13)
ax.legend()
for i, row in comp_df.iterrows():
    ax.text(i - w/2, row['Accuracy'] + 0.01, f"{row['Accuracy']:.4f}", ha='center', fontsize=10)
    ax.text(i + w/2, row['F1-macro'] + 0.01, f"{row['F1-macro']:.4f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_hierarchical_vs_flat.png'), dpi=120, bbox_inches='tight')
plt.close()
print("  -> 저장: fig9_hierarchical_vs_flat.png")

# ============================================================
# 10. 최종 모델 저장 & 결과 요약
# ============================================================
print("\n[10/10] 최종 모델 저장 & 결과 요약...")

# 최고 성능 모델 결정
all_cat1_results = (
    [r for r in results if r['카테고리'] == 'cat_1'] +
    [r for r in improved_results if r.get('카테고리', 'cat_1') == 'cat_1' and 'pipeline' in r]
)
best_overall = max(all_cat1_results, key=lambda x: x['정확도(Accuracy)'])
print(f"  최고 성능 cat_1 모델: {best_overall['모델명']} (Acc={best_overall['정확도(Accuracy)']:.4f})")

# 모델 저장
save_package = {
    'cat1_model': best_overall['pipeline'],
    'cat2_model': flat_cat2_result['pipeline'],
    'cat3_model': pipe_svc_c3,
    'meta': {
        'cat1_accuracy': best_overall['정확도(Accuracy)'],
        'cat1_f1_macro': best_overall['F1-macro'],
        'cat2_accuracy': flat_cat2_result['정확도(Accuracy)'],
        'cat3_accuracy': next(r for r in results if r['카테고리'] == 'cat_3')['정확도(Accuracy)'],
        'best_cat1_model_name': best_overall['모델명']
    }
}
joblib.dump(save_package, MODEL_SAVE_PATH)
print(f"  모델 저장 완료: {MODEL_SAVE_PATH}")

# 전체 실험 결과 요약
print("\n" + "=" * 70)
print("전체 실험 결과 요약")
print("=" * 70)

final_rows = []
# 베이스라인
for r in results:
    final_rows.append({
        '구분': '베이스라인',
        '모델명': r['모델명'],
        '카테고리': r['카테고리'],
        'Accuracy': r['정확도(Accuracy)'],
        'F1-macro': r['F1-macro'],
        'F1-weighted': r['F1-weighted']
    })
# 개선
for r in improved_results:
    final_rows.append({
        '구분': '개선',
        '모델명': r['모델명'],
        '카테고리': r.get('카테고리', 'cat_1'),
        'Accuracy': r['정확도(Accuracy)'],
        'F1-macro': r['F1-macro'],
        'F1-weighted': r['F1-weighted']
    })
# 계층적
final_rows.append({
    '구분': '계층적',
    '모델명': '2-Stage Hierarchical',
    '카테고리': 'cat_2',
    'Accuracy': round(hier_acc, 4),
    'F1-macro': round(hier_f1, 4),
    'F1-weighted': round(f1_score(y_test_hier, y_pred_hier, average='weighted', zero_division=0), 4)
})

final_df = pd.DataFrame(final_rows)
print(final_df.to_string(index=False))

# 최종 요약 시각화
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
for idx, (cat_lv, ax) in enumerate(zip(['cat_1', 'cat_2', 'cat_3'], axes)):
    subset = final_df[final_df['카테고리'] == cat_lv].copy()
    if len(subset) == 0:
        ax.axis('off')
        continue
    subset_sorted = subset.sort_values('Accuracy', ascending=True)
    colors = ['gold' if v == subset['Accuracy'].max() else 'steelblue' for v in subset_sorted['Accuracy']]
    bars = ax.barh(subset_sorted['모델명'], subset_sorted['Accuracy'], color=colors, alpha=0.85)
    ax.set_title(f'{cat_lv} - 모델별 정확도', fontsize=12)
    ax.set_xlabel('Accuracy')
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, subset_sorted['Accuracy']):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=8)

plt.suptitle('전체 실험 최종 결과 요약', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_final_summary.png'), dpi=120, bbox_inches='tight')
plt.close()
print("\n  -> 저장: fig10_final_summary.png")

# 최종 추천 모델 요약
print("\n" + "=" * 70)
print("프로덕션 추천 모델")
print("=" * 70)
for cat_lv in ['cat_1', 'cat_2', 'cat_3']:
    subset = final_df[final_df['카테고리'] == cat_lv]
    if len(subset):
        best_row = subset.nlargest(1, 'Accuracy').iloc[0]
        print(f"  {cat_lv}: {best_row['모델명']:25s} | Acc={best_row['Accuracy']:.4f} | F1-macro={best_row['F1-macro']:.4f}")

# 추론 예시
print("\n" + "=" * 70)
print("모델 추론 예시 (신규 상품 카테고리 예측)")
print("=" * 70)
loaded = joblib.load(MODEL_SAVE_PATH)
test_samples = [
    "레이스 플리츠 미디 스커트 A라인 봄 여름 원피스",
    "실리콘 에어팟 케이스 투명 보호 커버 아이폰",
    "캠핑 텐트 2인용 방수 야외 등산 백패킹",
    "코튼 오버핏 후드티 기모 맨투맨 스트리트",
    "14K 골드 체인 목걸이 레이어드 미니멀",
]
for sample in test_samples:
    pred_c1 = loaded['cat1_model'].predict([sample])[0]
    pred_c2 = loaded['cat2_model'].predict([sample])[0]
    print(f"  입력: {sample[:45]}")
    print(f"  예측: cat_1={pred_c1} | cat_2={pred_c2}")
    print()

print("=" * 70)
print("모든 실험 완료!")
print(f"결과 이미지 저장 위치: {OUTPUT_DIR}")
print(f"모델 저장 위치: {MODEL_SAVE_PATH}")
print("=" * 70)

# 결과를 JSON으로도 저장
results_json_path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
save_data = {
    'baseline': [{k: v for k, v in r.items() if k not in ('pipeline','y_pred','y_test')} for r in results],
    'improved': [{k: v for k, v in r.items() if k not in ('pipeline','y_pred')} for r in improved_results],
    'hierarchical_cat2': {
        'accuracy': round(hier_acc, 4),
        'f1_macro': round(hier_f1, 4)
    }
}
with open(results_json_path, 'w', encoding='utf-8') as f:
    json.dump(save_data, f, ensure_ascii=False, indent=2)
print(f"결과 JSON 저장: {results_json_path}")
