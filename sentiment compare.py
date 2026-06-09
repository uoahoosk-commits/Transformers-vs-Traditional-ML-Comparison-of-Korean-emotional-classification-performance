# =============================================================
#  트랜스포머 vs 전통 ML 정확도 비교 (한국어 감성분류, NSMC)
#  - 모델1: TF-IDF + 로지스틱 회귀 (밑바닥부터 학습)
#  - 모델2: 사전학습 트랜스포머 (학습 없이 예측만)
#  ※ 구글 코랩에서 실행 추천 (런타임 > 하드웨어가속기 > GPU)
# =============================================================

# --- 코랩에서 최초 1회만 실행 ---
# !pip install transformers

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# -------------------------------------------------------------
# 1) 데이터 불러오기 (네이버 영화리뷰, 긍정=1 / 부정=0)
# -------------------------------------------------------------
url_train = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt"
url_test  = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"

train = pd.read_csv(url_train, sep="\t").dropna()
test  = pd.read_csv(url_test,  sep="\t").dropna()

# 속도를 위해 일부만 사용 (단순화). 숫자를 키우면 더 정확해짐.
train = train.sample(3000, random_state=42)
test  = test.sample(1000, random_state=42)

X_train, y_train = train["document"].tolist(), train["label"].tolist()
X_test,  y_test  = test["document"].tolist(),  test["label"].tolist()

# -------------------------------------------------------------
# 2) 모델1 : TF-IDF + 로지스틱 회귀
#    글자 단위 n-gram을 쓰므로 형태소 분석기(konlpy) 설치 불필요
# -------------------------------------------------------------
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=20000)
Xtr = vec.fit_transform(X_train)
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xtr, y_train)
pred_ml = clf.predict(Xte)

acc_ml = accuracy_score(y_test, pred_ml)
f1_ml  = f1_score(y_test, pred_ml)

# -------------------------------------------------------------
# 3) 모델2 : 사전학습 트랜스포머 (예측만, 학습 X)
# -------------------------------------------------------------
from transformers import pipeline

sentiment = pipeline(
    "sentiment-analysis",
    model="sangrimlee/bert-base-multilingual-cased-nsmc",
    truncation=True,
)

# (중요) 모델마다 출력 라벨 이름이 다르므로 자동으로 매핑을 찾는다.
#  확실한 긍정/부정 문장을 넣어 어떤 라벨이 '긍정'인지 알아냄.
pos_label = sentiment("정말 재미있고 최고의 영화였어요")[0]["label"]
neg_label = sentiment("최악이에요 시간 낭비였습니다")[0]["label"]
print(f"[라벨 확인] 긍정 라벨 = {pos_label} / 부정 라벨 = {neg_label}")

raw = sentiment(X_test)                       # 테스트셋 예측
pred_tf = [1 if r["label"] == pos_label else 0 for r in raw]

acc_tf = accuracy_score(y_test, pred_tf)
f1_tf  = f1_score(y_test, pred_tf)

# -------------------------------------------------------------
# 4) 결과 비교
# -------------------------------------------------------------
print("\n================ 비교 결과 ================")
print(f"{'모델':<28}{'정확도':>8}{'F1':>8}")
print(f"{'TF-IDF + 로지스틱회귀':<24}{acc_ml:>8.3f}{f1_ml:>8.3f}")
print(f"{'사전학습 트랜스포머':<25}{acc_tf:>8.3f}{f1_tf:>8.3f}")
print("==========================================")
