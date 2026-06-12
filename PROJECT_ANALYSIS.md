# 도심 침수 예측 GNN — 프로젝트 분석 문서

> 작성일: 2026-06-04  
> 작업 디렉토리: `/home/namjun/city_flood`

---

## 1. 프로젝트 배경 및 목표

### 배경
서울시는 집중호우 시 도로 침수 사고가 빈번하게 발생하며, 현재는 도로노면 수위 센서가 침수를 감지한 이후에야 경보가 발령된다. 하수관로 수위는 도로 침수에 선행하는 패턴을 보이므로, 이를 활용하면 **사전 경보**가 가능하다.

### 목표
- 서울시 **하수관로 수위 센서(456개)** + **도로노면 수위 센서(112개)**를 공간 그래프로 연결
- GNN 기반 시공간 모델로 도로 침수를 **최대 30분 전 예측**
- 3가지 태스크 동시 지원: 수위 회귀, 이진 침수 분류, 다단계 침수 분류

---

## 2. 데이터 현황

### 원천 데이터

| 도메인 | 기간 | 파일 수 | 센서 수 | 해상도 |
|--------|------|---------|---------|--------|
| 하수관로 수위 | 2022-01 ~ 2025-08 | 44개 CSV | 456개 | 1분 |
| 도로노면 수위 | 2024-01 ~ 2025-12 | 24개 TXT | 112개 | 1분 |

### 공통 학습 기간
```
2024-01-01 ~ 2025-08-31 (20개월)
├── Train : 2024-01-01 ~ 2024-12-31 (12개월)
├── Val   : 2025-01-01 ~ 2025-05-31  (5개월)
└── Test  : 2025-06-01 ~ 2025-08-31  (3개월, 장마철 포함)
```

> **Test 기간 선택 근거**: 장마철(6~8월)을 Test로 지정하여 실제 침수 이벤트에 대한 일반화 성능을 엄격하게 평가

### 처리 후 데이터 볼륨 (10분 해상도 환산)

| Split | 하수관로 행 수 | 도로 행 수 | 타임스텝 수 |
|-------|--------------|-----------|------------|
| Train (2024) | ~33M | ~8M | ~52,560 |
| Val (2025 Jan-May) | ~15M | ~4M | ~21,900 |
| Test (2025 Jun-Aug) | ~6M | ~1.5M | ~13,104 |

---

## 3. 전처리 파이프라인 요약

### Step 01 — CSV/TXT → Parquet 변환
- 인코딩 자동 감지: cp949 → utf-8-sig → utf-8 순서로 시도
- 컬럼명 한·영 혼용 → 통일 매핑 (`COLUMN_MAP`)
- 출력: `dataset/processed/raw_parquet/`

### Step 02 — 스키마 통일 및 통합
- 2025-07/08 신규 파일의 영문 컬럼 (`se_cd`, `se_nm`) → 기존 한글 기준으로 통일
- `위치정보` 컬럼 제거 (기준 스키마 외 도메인)
- 출력: `dataset/processed/merged/sewer_all.parquet`, `road_all.parquet`

### Step 03 — 이상치 제거 + 결측치 보간
```
하수관로 이상치:
  - 음수 level → NaN
  - 측정범위 최댓값(제원표) 초과 → NaN

도로노면 이상치:
  - 오류코드 {312, 419, 999, 1000} → NaN

보간 정책:
  - 갭 ≤ 10분 : 선형 보간 (limit_direction='forward')
  - 갭 > 10분 : NaN 유지 (데이터 없는 구간으로 처리)

병렬처리:
  - 하수관로: workers=2 (파일당 ~970 MB)
  - 도로노면: workers=4 (파일당 ~141 MB)
```

### Step 04 — 제원표 조인 (노드 메타데이터)
- `서울시 수위계(하수관로) 제원표_20260310.xlsx`
- `서울시 수위계(도로) 제원표_20260310.xlsx`
- 도로 센서는 지점명 기반 3단계 매핑:
  1. 정확 매핑 (exact)
  2. 자치구 접두사 제거 후 매핑 (prefix_strip)
  3. 미매핑 (unmatched)
- 관규격 파싱 → `pipe_height_m` (하수관로 만수율 계산용)
- 출력: `sewer_node.parquet`, `road_node.parquet`

### Step 05 — 탐색적 분석 + 상관 분석
- 여름철 데이터(장마 2개월 × 2년) 10분 집계 후 분석
- 센서 쌍 구성 기준: **동일 배수구역 + Haversine 거리 ≤ 1km**
- 765개 후보 쌍에 대해 ±60분(6 lag) 교차 상관 계산
- 이벤트 조건부 상관 (`corr_event`, road_level > 0인 시점만)
- 출력: `correlation_results.parquet`, 상관 히트맵, 지도 시각화

### Step 06 — 파생 피처 생성

#### 하수관로 파생 피처
| 피처 | 계산식 | 의미 |
|------|--------|------|
| `level_diff` | `level.diff()` | 수위 변화량 |
| `fill_rate` | `level / pipe_height_m` (clip 0~1) | 관 만수율 |
| `hour`, `month` | timestamp 추출 | 절대 시간 |
| `season` | month → {1:봄, 2:여름, 3:가을, 4:겨울} | 계절 |
| `is_weekend` | dayofweek ≥ 5 | 주말 여부 |

#### 도로노면 파생 피처
| 피처 | 계산식 | 의미 |
|------|--------|------|
| `level_diff` | `level.diff()` | 수위 변화량 |
| `flood_flag` | `level > 0` | 침수 여부 (이진) |
| `flood_stage` | `pd.cut(level, [-1,0,5,20,50,∞])` | 침수 단계 (0~4) |
| `hour_sin/cos` | `sin/cos(2π·hour/24)` | 시간 주기 인코딩 |
| `month_sin/cos` | `sin/cos(2π·month/12)` | 월 주기 인코딩 |

### Step 07 — 인접행렬 생성 (Gaussian 가중치)

```
Gaussian 가중치: w = exp(-d / σ)
  σ (sigma) = 300m
  threshold = 0.1 (이하 엣지 제거)

sewer → road 엣지 (501개):
  - 동일 배수구역 + 거리 ≤ 1km
  - 고립 도로 센서 fallback: 2km 이내 최근접 하수관 연결

sewer → sewer 엣지 (1,192개):
  - 동일 배수구역 + 거리 ≤ 500m
```

### Step 08-09 — 공통 기간 분리 및 정규화

```
정규화 (Data Leakage 방지):
  - Train 데이터(2024년)에서만 파라미터 계산
  - level_norm: 센서별 max(학습최대, 물리최대) 기준 min-max → [0, 1]
  - level_diff_norm: 센서별 z-score → clip [-5, 5]
  - hour_sin/cos, month_sin/cos: 수식 기반 (파라미터 없음)
```

### Step 10-11 — 분할 및 GNN 설정 저장
- Train/Val/Test parquet 파일 생성 완료
- `gnn_config.json`: 그래프 통계, 피처 목록, 클래스 가중치, 시간 설정

---

## 4. 그래프 구조 분석

### 그래프 요약

```
이종 그래프 (Heterogeneous Graph)
├── 노드 타입 1: Sewer  (456개, 9-dim 피처)
├── 노드 타입 2: Road   (112개, 10-dim 피처)
├── 엣지 타입 1: Sewer → Road  (501개)
└── 엣지 타입 2: Sewer → Sewer (1,192개)
```

### 엣지 분포

| 통계 | Sewer→Road | Sewer→Sewer |
|------|-----------|-------------|
| 총 엣지 수 | 501 | 1,192 |
| 평균 거리 | ~430m | ~280m |
| 가중치 범위 | 0.10 ~ 1.00 | 0.10 ~ 1.00 |
| 평균 가중치 | ~0.36 | ~0.55 |

### 고립 노드 처리
- 초기 도로 노드 중 일부가 어떤 배수구역의 하수 센서와도 1km 내에 없어 엣지 없음
- **Fallback 전략**: 2km 이내 최근접 하수 센서와 강제 연결 (`edge_type='fallback'`)

---

## 5. 피처 엔지니어링 결정 사항

### GNN 입력 피처 (최종)

| 도메인 | 피처 (순서) | 차원 |
|--------|------------|------|
| Sewer | level_norm, level_diff_norm, fill_rate, hour_sin, hour_cos, month_sin, month_cos, season, is_weekend | 9 |
| Road | level_norm, level_diff_norm, flood_flag, flood_stage, hour_sin, hour_cos, month_sin, month_cos, season, is_weekend | 10 |

### 강우량 피처 (미확보, Placeholder)
- 위치: sewer 피처 10번째 (`rainfall_norm`)
- 단위: mm/hr, 최근접 AWS 기상관측소 데이터
- 정규화: min-max (0~100 mm/hr 기준)
- **현재 상태**: AWS 데이터 확보 후 `sewer_normalized.parquet` 재생성 필요

### 타겟 피처 (3가지 예측 태스크)

| 태스크 | 타겟 | 타입 | 비고 |
|--------|------|------|------|
| 회귀 | `level_norm` | 연속 | 도로 수위 예측 |
| 이진 분류 | `flood_flag` | 0/1 | 침수 여부 |
| 다중 분류 | `flood_stage` | 0~4 | 침수 단계 |

---

## 6. 클래스 불균형 분석

### 이진 분류 (flood_flag)

```
학습 데이터 기준:
  양성 (침수): 2,149,356건 (약 4.8%)
  음성 (정상): 42,440,891건 (약 95.2%)
  불균형 비율: 1 : 19.7 ≈ 1 : 20
  
대응: pos_weight = 19.75 (BCEWithLogitsLoss)
```

### 다중 분류 (flood_stage)

| 단계 | 기준 (cm) | 의미 | 클래스 가중치 |
|------|----------|------|--------------|
| 0 | 0 | 정상 | 0.21 |
| 1 | 0~5 | 경미 | 4.39 |
| 2 | 5~20 | 주의 | 80.34 |
| 3 | 20~50 | 위험 | 1,191.46 |
| 4 | 50+ | 심각 | 7,714.58 |

> Stage 3/4는 극도로 희귀하여 모델 학습 시 WeightedRandomSampler 또는 Focal Loss 고려 필요

---

## 7. 시계열 설정

```
해상도  : 10분 (원천 1분 데이터를 리샘플링)
입력창  : T_in  = 6 스텝 = 60분 과거 관측
예측창  : T_out = 3 스텝 = 30분 미래 예측
슬라이딩: 스텝 1 (1 stride)
```

---

## 8. 처리 완료 파일 목록

```
dataset/features/
├── adjacency_expanded.parquet     # sewer→road 엣지 (501개)
├── sewer_sewer_edges.parquet      # sewer→sewer 엣지 (1,192개)
├── sewer_node_index.parquet       # 하수 노드 인덱스 (456개)
├── road_node_index.parquet        # 도로 노드 인덱스 (112개)
├── gnn_config.json                # GNN 설정
├── overlap/
│   ├── sewer_normalized.parquet   # 전체 정규화 (2024~2025-08)
│   ├── road_normalized.parquet
│   ├── sewer_norm_params.parquet  # 정규화 파라미터
│   ├── road_norm_params.parquet
│   ├── train/
│   │   ├── sewer_train.parquet    # 2024
│   │   └── road_train.parquet
│   ├── val/
│   │   ├── sewer_val.parquet      # 2025-01~05
│   │   └── road_val.parquet
│   └── test/
│       ├── sewer_test.parquet     # 2025-06~08
│       └── road_test.parquet
```

---

## 9. 미결 사항

| 항목 | 우선순위 | 비고 |
|------|---------|------|
| AWS 강우량 데이터 확보 | 높음 | sewer 피처 9→10 차원 확장 |
| road→sewer 역방향 엣지 검토 | 중간 | 양방향 메시지 패싱 가능성 |
| 결측 노드 보완 | 낮음 | 미매핑 도로 센서 처리 방안 |

---

## 10. 다음 단계 — GNN 모델 학습

`model_comparison.ipynb` 에서 아래 5개 모델을 비교 실험:

| # | 모델 | 특징 |
|---|------|------|
| 1 | **Baseline-LSTM** | 공간 정보 없음, 시간만 | 
| 2 | **GC-GRU** | GCN 공간 + GRU 시간 |
| 3 | **Hetero-GAT-GRU** | 이종 그래프 어텐션 + GRU |
| 4 | **STGCN** | 시공간 교차 컨볼루션 |
| 5 | **DCRNN** | 확산 그래프 컨볼루션 + GRU |
