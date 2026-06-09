# 도심 침수 예측 GNN (Urban Flood Prediction with GNN)

> 서울시 하수관로·도로노면 수위 센서 네트워크를 이종 그래프(Heterogeneous Graph)로 모델링하여 도로 침수를 최대 180분 전에 예측하는 GNN 비교 실험 프로젝트

---

## 목차

1. [프로젝트 배경](#프로젝트-배경)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [데이터셋](#데이터셋)
4. [그래프 구조](#그래프-구조)
5. [비교 모델](#비교-모델)
6. [실험 결과](#실험-결과)
7. [환경 설정](#환경-설정)
8. [실행 방법](#실행-방법)
9. [프로젝트 구조](#프로젝트-구조)
10. [향후 계획](#향후-계획)

---

## 프로젝트 배경

서울시는 집중호우 시 도로 침수로 인한 인명·재산 피해가 반복적으로 발생합니다. 현행 시스템은 도로노면 수위 센서가 침수를 **감지한 이후**에야 경보를 발령하는 사후 대응 구조입니다.

**핵심 가설**: 하수관로 수위 상승이 도로 침수에 **선행**한다. 하수관로→도로 수리학적 연결 관계를 그래프로 모델링하면 침수를 사전에 예측할 수 있다.

### 목표

- 하수관로 수위 패턴으로 도로 침수를 **최대 180분 전** 예측
- GNN 기반 공간 정보 활용이 단순 시계열 모델 대비 실질적 이득을 제공하는지 검증
- 실시간 조기경보 시스템 적용 가능성 평가

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                       입력 (T_in = 60분)                         │
│  하수관로 수위 센서 404개  ──┐                                    │
│  도로노면 수위 센서  90개   ──┼── 이종 그래프 (Hetero Graph)      │
│  강수량 관측소       48개  ──┘                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────▼────────────┐
         │   GNN 시공간 인코더      │
         │  (GC-GRU / Hetero-GAT   │
         │   / DCRNN / STGCN)     │
         └────────────┬────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                       출력 (T_out = 180분)                        │
│  ① 회귀:   도로 수위 level_norm 예측       shape: (90, 18)       │
│  ② 이진:   침수 여부 flood_flag 예측       shape: (90, 18)       │
└─────────────────────────────────────────────────────────────────┘
```

### 시간 설정

| 항목 | 값 | 설명 |
|------|-----|------|
| 데이터 해상도 | 10분 | 원천 1분 → 10분 리샘플 |
| 입력 window (T_in) | 6 스텝 = **60분** | 과거 수위 관측 |
| 예측 horizon (T_out) | 18 스텝 = **180분** | 미래 수위 예측 |

---

## 데이터셋

### 원천 데이터

| 도메인 | 기간 | 센서 수 | 해상도 |
|--------|------|---------|--------|
| 하수관로 수위 | 2022-01 ~ 2025-08 | 404개 | 1분 |
| 도로노면 수위 | 2024-01 ~ 2025-12 | 90개 | 1분 |
| 강수량 (AWS) | 2022-01 ~ 2025-08 | 48개 관측소 | 1시간 |

### 학습/검증/테스트 분할

```
Train : 2024-01-01 ~ 2024-12-31 (12개월, ~52,560 타임스텝)
Val   : 2025-01-01 ~ 2025-05-31  (5개월, ~21,900 타임스텝)
Test  : 2025-06-01 ~ 2025-08-31  (3개월, 장마철 포함)
```

> Test 기간을 장마철(6~8월)로 설정하여 실제 침수 이벤트에 대한 **엄격한 일반화 성능** 평가

### 클래스 불균형

```
flood_flag=1 (침수): 2,149,356건 (4.8%)
flood_flag=0 (정상): 42,440,891건 (95.2%)
불균형 비율: 1 : 20  →  pos_weight = 19.75 (BCEWithLogitsLoss)
```

### 피처 구성

**하수관로 노드 피처 (10-dim)**
| 피처 | 설명 |
|------|------|
| `level_norm` | 센서별 min-max 정규화 수위 |
| `level_diff_norm` | 수위 변화율 (z-score, clip ±5) |
| `fill_rate` | 관 만수율 (level / pipe_height) |
| `hour_sin/cos` | 시간 주기 인코딩 |
| `month_sin/cos` | 월 주기 인코딩 |
| `season` | 계절 (0=봄, 1=여름, 2=가을, 3=겨울) |
| `is_weekend` | 주말 여부 |
| `rainfall_norm` | 최근접 AWS 시간당 강수량 (mm/hr 정규화) |

**도로노면 노드 피처 (10-dim)**
| 피처 | 설명 |
|------|------|
| `level_norm` | 정규화 수위 |
| `level_diff_norm` | 수위 변화율 |
| `flood_flag` | 침수 여부 (0/1) — **이진 분류 타겟** |
| `flood_stage` | 침수 단계 (0~4) |
| `hour_sin/cos`, `month_sin/cos`, `season`, `is_weekend` | 시간 피처 |

---

## 그래프 구조

```
이종 그래프 (Heterogeneous Graph)
├── Sewer  노드: 404개
├── Road   노드:  90개
├── Sewer → Road  엣지:   383개  (집수구역 + 거리 ≤ 1km)
└── Sewer → Sewer 엣지: 1,192개  (거리 ≤ 500m, 양방향)

엣지 가중치: Gaussian 커널  w = exp(-d / σ),  σ = 300m,  임계값 = 0.1
```

<div align="center">
<img src="docs/graph_structure.png" width="600" alt="그래프 구조 시각화"/>
</div>

---

## 비교 모델

### 1. Baseline-LSTM
- 공간 정보 **미사용** (도로 노드 독립 처리)
- 2-layer LSTM → FC head
- 파라미터: ~55K
- 역할: GNN 공간 이득 측정 기준선

### 2. GC-GRU
- GCNConv 공간 집계 + GRU 시간 모델링
- 동종 그래프 근사 (sewer + road 통합)
- 파라미터: ~28K

### 3. Hetero-GAT-GRU
- **이종 그래프** 어텐션: sewer→road GATConv + sewer→sewer GCNConv
- 이웃 하수관의 중요도를 동적으로 학습
- 파라미터: ~72K

### 4. DCRNN
- 확산 컨볼루션 (Diffusion Conv, K=2) + GRU
- D⁻¹A 정규화 인접행렬 기반 K-hop 확산
- 배치 벡터화: `einsum('nm,bnf->bmf')` 적용
- 파라미터: ~78K

### 5. STGCN *(실험 제외)*
- ChebConv 공간 + Conv1d 시간 교차 블록
- T_OUT=18 장기 예측에 구조적 부적합 → 비교 제외

---

## 실험 결과

### 전체 기간 기준 (QUICK_RUN=False)

| 모델 | F1 ↑ | RMSE ↓ | AUPRC ↑ | Recall ↑ | CSI ↑ |
|------|------|--------|---------|---------|-------|
| Baseline-LSTM | 0.828 | **0.0036** | 0.876 | 정상 | — |
| GC-GRU | 0.800 | 0.0056 | 0.851 | 정상* | — |
| Hetero-GAT-GRU | **0.829** | **0.0035** | 0.874 | 정상* | — |
| DCRNN | 0.827 | 0.0036 | **0.877** | 정상 | — |

> \* GC-GRU, Hetero-GAT-GRU는 `POS_WEIGHT=60`으로 분류 붕괴 해소

### 학습 곡선

<div align="center">
<img src="docs/learning_curves.png" width="900" alt="모델별 학습 곡선"/>
</div>

### 레이더 차트

<div align="center">
<img src="docs/radar_chart.png" width="500" alt="Test 성능 레이더 차트"/>
</div>

### 핵심 발견

1. **공간 이득 불명확**: 전체 기간 재학습 후에도 Baseline-LSTM이 RMSE 기준 GNN과 동등하거나 우세 → 침수 구간 분리 MAE로 정밀 검증 진행 중
2. **DCRNN 최우수 GNN**: AUPRC 1위, 회귀·분류 균형, 학습 안정
3. **분류 불균형 대응 필수**: GAT/GCN 기반 모델은 기본 pos_weight(19.75)에서 Recall≈0 붕괴, `POS_WEIGHT=60` 적용 시 회복

---

## 환경 설정

### 요구사항

```
Python >= 3.10
CUDA (선택, CPU 학습 가능)
```

### 설치

```bash
git clone https://github.com/<your-username>/city_flood.git
cd city_flood

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 패키지 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric torch_scatter
pip install pandas pyarrow numpy scikit-learn matplotlib seaborn tqdm
pip install jupyter
```

### 주요 패키지 버전

| 패키지 | 역할 |
|--------|------|
| `torch` | 딥러닝 프레임워크 |
| `torch_geometric` | GNN 레이어 (GCNConv, GATConv, ChebConv) |
| `torch_scatter` | 그래프 scatter 연산 |
| `pyarrow` | 고속 Parquet I/O |
| `pandas` | 데이터 처리 |
| `scikit-learn` | 평가 지표 |

---

## 실행 방법

### 1. 전처리

```bash
# Step 1-3: 원천 데이터 → 정규화 피처
jupyter nbconvert --to notebook --execute preprocessing.ipynb

# Step 4: 강수량 조인
jupyter nbconvert --to notebook --execute preprocessing_rain_join.ipynb
```

### 2. 모델 학습 및 비교

```bash
jupyter notebook model_comparison.ipynb
```

**노트북 실행 순서:**

| 셀 | 내용 |
|----|------|
| a005 ~ a008 | 임포트, 설정, 그래프 로드 |
| a010 | 텐서 변환 + 캐시 생성 (첫 실행 시 수 분 소요) |
| a012 | DataLoader 생성 |
| a015 ~ a023 | 모델 클래스 정의 |
| a025 ~ a026 | 학습/평가 함수 정의 |
| a028 | 실험 실행 (모델 5개 순차 학습) |
| a030 ~ a034 | 결과 시각화 및 분석 |

### 3. 주요 설정 변경

```python
# model_comparison.ipynb — a006 셀
CFG = {
    'QUICK_RUN': False,  # True: 장마철 2개월 (빠른 테스트)
                         # False: 전체 기간 (논문용)
    'EPOCHS': 30,
    'BATCH' : 128,
    'T_IN'  : 6,         # 입력 60분
    'T_OUT' : 18,        # 예측 180분
}
```

---

## 프로젝트 구조

```
city_flood/
├── README.md
├── CLAUDE.md                          # AI 작업 가이드 (개발 이력·버그 해결책)
├── EVALUATION_FRAMEWORK.md            # 성능 평가 지표 체계
├── PROJECT_ANALYSIS.md                # 전처리 파이프라인 상세 분석
│
├── model_comparison.ipynb             # ★ 메인 실험 노트북
├── preprocessing.ipynb                # 원천 데이터 전처리 (Step 01~09)
├── preprocessing_rain_join.ipynb      # 강수량 피처 조인
├── preprocessing_rain.ipynb           # 강수량 데이터 정제
├── download_kma_file.py               # 기상청 AWS 데이터 다운로드
│
└── dataset/
    └── processed/
        └── features/
            ├── gnn_config.json        # 그래프 설정, 클래스 가중치
            ├── adjacency_expanded.parquet
            ├── sewer_sewer_edges.parquet
            ├── sewer_node_index.parquet
            ├── road_node_index.parquet
            └── overlap/
                ├── train/             # sewer_train.parquet, road_train.parquet
                ├── val/               # sewer_val.parquet, road_val.parquet
                ├── test/              # sewer_test.parquet, road_test.parquet
                └── tensor_cache/      # 학습용 텐서 캐시 (.pt)
```

---

## 평가 지표

| 범주 | 지표 | 설명 |
|------|------|------|
| **회귀 (전체)** | MAE, RMSE, R² | level_norm 예측 오차 |
| **회귀 (Horizon)** | MAE@10/30/60/180min | 예측 거리별 오차 분리 |
| **분류** | **AUPRC** (주), AUROC (보조) | 1:20 불균형 → AUPRC 핵심 |
| **분류** | F1, Recall, Precision, CSI | 최적 임계값 PR 곡선 F1 최대화 자동 탐색 |
| **공간** | NodeMAE 중앙값·P90 | 노드별 성능 편차 |
| **침수 구간** | MAE (flood/normal 분리) | 공간 이득 정밀 검증 |

> AUROC는 1:20 불균형 데이터에서 낙관적 과평가 — **AUPRC를 주 지표로 사용**

---

## 향후 계획

- [ ] 침수 구간 분리 MAE 분석 완료 → GNN 공간 이득 정량화
- [ ] 그래프 구조 개선: 상류-하류 방향성 반영, 집수구역 기반 엣지 재설계
- [ ] GC-GRU 이종 그래프 구조 전환
- [ ] STGCN T_OUT=6 단기 예측 전용 재설계 후 재포함
- [ ] Detection Lead Time 운영 지표 평가 (대피 가능 사전 경보 시간)
- [ ] Event-wise Recall 측정 (타임스텝이 아닌 이벤트 단위)

---

## 참고 문헌

- Li, Y. et al. (2018). *Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting.* ICLR.
- Yu, B. et al. (2018). *Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting.* IJCAI.
- Veličković, P. et al. (2018). *Graph Attention Networks.* ICLR.

---

## 라이선스

본 프로젝트의 코드는 MIT License를 따릅니다. 데이터는 서울시 공공데이터 활용 정책에 따릅니다.
