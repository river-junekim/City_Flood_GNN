# (레거시) GNN 5모델 비교 프로젝트 — 재설계 이전 기록
> 프로젝트가 '시계열 전처리 재설계 + 하수 surcharge 중심'으로 전환됨. 아래는 참고용 과거 기록(데이터 구조·모델·버그·평가지표 이력).

## 1. 프로젝트 개요

서울시 **하수관로 수위 센서(404개)** 와 **도로노면 수위 센서(90개)** 를 이종 그래프(Heterogeneous Graph)로 연결하여, 하수관로 수위 패턴으로 도로 침수를 **최대 180분 전** 예측하는 GNN 모델 비교 실험 프로젝트.

### 핵심 목표
- 도로 수위 **회귀 예측** (level_norm, T_out=18 스텝, 10분 해상도)
- 도로 **침수 이진 분류** (flood_flag, 1:20 클래스 불균형)
- 5개 모델 비교: Baseline-LSTM, GC-GRU, Hetero-GAT-GRU, STGCN, DCRNN

---

## 2. ⚠️ 데이터 접근 제약 (반드시 준수)

```
❌ 직접 읽기 금지: 도로노면(road_*.parquet), 하수관로(sewer_*.parquet) 데이터 파일
✅ 읽기 허용: 강수량 데이터, gnn_config.json, 노드 인덱스, 엣지 파일
✅ 코드 작성은 가능: 데이터를 직접 실행하지 않는 코드는 작성 가능

이유: 데이터 크기가 수백 MB ~ 수 GB 수준이며, 민감한 센서 원시 데이터
```

---

## 3. 디렉토리 구조

```
city_flood/
├── CLAUDE.md                          # 이 파일
├── EVALUATION_FRAMEWORK.md            # 성능 평가 지표 체계 문서
├── PROJECT_ANALYSIS.md                # 전처리 파이프라인 분석 문서
├── model_comparison.ipynb             # 메인 실험 노트북 ★
├── preprocessing_rain_join.ipynb      # 강수량 조인 전처리
├── preprocessing.ipynb                # 원천 데이터 전처리
├── preprocessing_rain.ipynb           # 강수량 전처리
├── download_kma_file.py               # 기상청 데이터 다운로드
│
├── dataset/features/
│   ├── gnn_config.json                # 그래프 설정, 피처 목록, 클래스 가중치
│   ├── adjacency_expanded.parquet     # sewer→road 엣지
│   ├── sewer_sewer_edges.parquet      # sewer→sewer 엣지
│   ├── sewer_node_index.parquet       # 하수 노드 인덱스 (404개)
│   ├── road_node_index.parquet        # 도로 노드 인덱스 (90개)
│   └── overlap/
│       ├── train/                     # 2024-01-01 ~ 2024-12-31
│       ├── val/                       # 2025-01-01 ~ 2025-05-31
│       └── test/                      # 2025-06-01 ~ 2025-08-31
│
└── dataset/processed/
    └── tensor_cache/                  # load_tensor() 캐시 (.pt 파일)
```

---

## 4. 그래프 구조

```
이종 그래프 (Heterogeneous Graph)
├── Sewer 노드: 404개, 9-dim 피처
│   피처: level_norm, level_diff_norm, fill_rate,
│         hour_sin, hour_cos, month_sin, month_cos,
│         is_weekend, rainfall_norm (서울 강우망 48개소 통합 완료)
│
├── Road 노드: 90개, 8-dim 피처
│   피처: level_norm, level_diff_norm, flood_flag,
│         hour_sin, hour_cos, month_sin, month_cos, is_weekend
│   ※ season 제거(month와 중복), flood_stage는 parquet 보존하되
│     멀티클래스 타겟 전용 — 입력 피처에서 제외 (road_multiclass)
│
├── Sewer→Road 엣지: 383개 (Gaussian 가중치, σ=300m)
└── Sewer→Sewer 엣지: 1,192개 (양방향 포함)
```

### FloodDataset 슬라이딩 윈도우
```python
T_IN  = 6   # 입력: 60분 과거
T_OUT = 18  # 예측: 180분 미래
해상도 = 10분

반환 형태:
  x_sewer: (N_s=404, T_in=6, F_s=9)
  x_road:  (N_r=90,  T_in=6, F_r=8)
  y_reg:   (N_r=90,  T_out=18)   # level_norm 타겟 (ROAD_FEATS[0])
  y_cls:   (N_r=90,  T_out=18)   # flood_flag 타겟 (ROAD_FEATS[2])
```

---

## 5. model_comparison.ipynb 셀 구조

| 셀 ID | 내용 |
|-------|------|
| a005 | Import + DEVICE + TF32 설정 |
| a006 | CFG 하이퍼파라미터 (QUICK_RUN, T_IN, T_OUT, BATCH, EPOCHS 등) |
| a008 | 그래프 엣지/노드 로드 (SS_EI, SR_EI, SS_EW, SR_EW) |
| a010 | `load_tensor()` — parquet → (T,N,F) 텐서 변환 + 디스크 캐시 |
| a012 | `FloodDataset` 클래스 + DataLoader 생성 |
| a015 | Baseline-LSTM |
| a017 | GC-GRU (GCNConv + GRU) |
| a019 | Hetero-GAT-GRU (GATConv + GRU) |
| a021 | STGCN (STBlock + ChebConv) |
| a023 | DCRNN (DiffusionConv + GRU) |
| a025 | `compute_metrics()`, `train_epoch()`, `eval_epoch()` |
| a026 | `run_experiment()` — 학습 루프 + 최고 모델 선택 |
| a028 | `make_models()` + 실험 실행 루프 |
| a030 | 결과 테이블 출력 |
| a031 | 학습 곡선 시각화 |
| a032 | 레이더 차트 |
| a033 | Params vs 성능 산점도 |
| a033b | 침수 구간 분리 MAE (신규) |
| a034 | 결론 및 권장 모델 |

---

## 6. 핵심 설정값 (CFG)

```python
CFG = {
    'QUICK_RUN'  : False,    # False=전체 기간(공간 이득 재확인용)
                             # True=장마철 2개월(빠른 테스트)
    'T_IN'       : 6,        # 입력 60분
    'T_OUT'      : 18,       # 예측 180분
    'HIDDEN'     : 64,
    'CHEB_K'     : 3,        # ChebConv 다항식 차수
    'DIFF_K'     : 2,        # DCRNN 확산 스텝
    'GAT_HEADS'  : 4,
    'DROPOUT'    : 0.1,
    'EPOCHS'     : 30,       # 50→30 (전체 기간에서 수렴 빠름)
    'BATCH'      : 128,      # 32→128 (속도 4x 향상)
    'LR'         : 1e-3,
    'WEIGHT_DEC' : 1e-4,
    'POS_WEIGHT' : 19.75,    # 기본 클래스 가중치 (1:20 불균형)
}

# 모델별 오버라이드 (a028)
MODEL_CFG_OVERRIDES = {
    'Hetero-GAT-GRU': {'POS_WEIGHT': 60.0},  # 분류 붕괴 방지
    'GC-GRU':         {'POS_WEIGHT': 60.0},  # 분류 붕괴 방지
}
```

### 날짜 범위
```
QUICK_RUN=False (전체 기간):
  train : 2024-01-01 ~ 2024-12-31
  val   : 2025-01-01 ~ 2025-05-31
  test  : 2025-06-01 ~ 2025-08-31
```

---

## 7. load_tensor() 핵심 구현

```python
# 성능 최적화 이력:
# 1. iterrows() 루프 → 벡터화 numpy 인덱싱 (40~50분 → 수 분)
# 2. groupby().resample() → dt.floor() + groupby (2~5x 속도 향상)
# 3. 디스크 캐시 (.pt) → 재실행 시 즉시 로드
# 4. np.nan_to_num(copy=False) → parquet 내 NaN → 0 (학습 발산 방지)

def load_tensor(path, id2idx, feat_cols, resample='10min', agg='mean', date_range=None):
    # 캐시 키: 파일 크기 + 피처 + 날짜 범위 해시
    cache = _cache_path(path, feat_cols, resample, agg, date_range)
    if cache.exists():
        return torch.load(cache, weights_only=False)  # 캐시 HIT

    # 핵심: 벡터화 인덱싱
    t_idx = df['timestamp'].map(t_map).values.astype(np.int32)
    n_idx = df['sensor_id'].map(id2idx).values.astype(np.int32)
    vals  = df[feat_cols].values.astype(np.float32)
    tensor = np.zeros((n_t, n_n, n_f), dtype=np.float32)
    tensor[t_idx, n_idx, :] = vals
    np.nan_to_num(tensor, nan=0.0, copy=False)  # ← NaN 제거 필수
```

> **캐시 주의**: parquet 파일이 변경되면 st_size 기반으로 자동 무효화됨.  
> 수동 무효화 필요 시: `rm dataset/processed/tensor_cache/*.pt`

---

## 8. 모델별 특징 및 현재 상태

### Baseline-LSTM
- 구조: 도로 노드 독립 LSTM (공간 정보 미사용)
- 파라미터: ~55K
- 상태: 정상 ✅
- 특이사항: RMSE 기준 전체 1위 — GNN 공간 이득이 미미함을 시사

### GC-GRU
- 구조: 동종 그래프 근사 (sewer+road 통합) + GRU
- 파라미터: ~28K
- 상태: POS_WEIGHT=60 적용 ✅ (기본값에서 Recall≈0 붕괴)
- 구조적 한계: 이종 신호 혼재로 RMSE 불안정 → 장기적으로 이종 구조 전환 필요

### Hetero-GAT-GRU
- 구조: sewer→road GATConv + sewer→sewer GCNConv + GRU
- 파라미터: ~72K
- 상태: POS_WEIGHT=60 적용 후 정상 ✅ (기본값 19.75에서 Recall≈0 붕괴)
- forward 내 `for t in range(T)` 루프 존재 (Python 루프 — 개선 여지)

### STGCN
- 구조: ChebConv 공간 + Conv1d 시간 교차 블록
- 파라미터: ~90K
- 상태: **실험 제외** ❌
- 제외 이유: `fc_reg = Linear(hidden*T_in, T_out)` 구조가 T_OUT=18 장기 예측에 부적합
  Val MAE 30에폭 내내 진동, Recall≈0

### DCRNN
- 구조: 확산 컨볼루션(einsum 벡터화) + GRU
- 파라미터: ~78K
- 상태: 정상 ✅
- 최적화: `for b in range(B)` 배치 루프 제거 → `einsum('nm,bnf->bmf')` 벡터화
- AUPRC 1위, 회귀·분류 균형 최우수

---

## 9. 평가 지표 체계

### 주요 지표 (compute_metrics 함수)

| 지표 | 설명 | 비고 |
|------|------|------|
| AUPRC | PR 곡선 면적 | **1:20 불균형 데이터 핵심 지표** |
| AUROC | ROC 곡선 면적 | 보조 (불균형에서 낙관적 과평가) |
| F1 | 조화 평균 | 임계값: PR 곡선 F1 최대화 자동 탐색 |
| Recall | 침수 탐지율 | 운영 최우선 (Miss > False Alarm) |
| CSI | TP/(TP+FP+FN) | 기상·침수 분야 표준 |
| MAE | 전체 평균 절대 오차 | level_norm 단위 |
| MAE@Xmin | Horizon별 MAE (X=10,30,60,180) | 장기 예측 성능 분리 평가 |
| NodeMAE P50/P90 | 노드별 MAE 중앙값/90분위 | 공간 편향 탐지 |
| R² | 설명력 | 보조 |

### 분류 임계값
```python
# eval_epoch에서 자동 탐색
prec_c, rec_c, thr_c = precision_recall_curve(true_cls, prob)
f1_c = 2 * prec_c[:-1] * rec_c[:-1] / (prec_c[:-1] + rec_c[:-1] + 1e-9)
threshold = thr_c[np.nanargmax(f1_c)]
```

---

## 10. 발생했던 주요 버그 및 해결책

### Bug 1: `KeyError: 'Column rainfall_norm does not exist in schema'`
- 파일: `preprocessing_rain_join.ipynb`, `add_rainfall_norm()`
- 원인: parquet 파일에 `rainfall_norm` 컬럼이 이미 존재할 때 `schema.append()` 중복 호출
- 해결: `if 'rainfall_norm' not in old_schema.names:` 멱등성 체크 추가

### Bug 2: VSCode 메모리 크래시 (Step 4 전처리)
- 원인: `pd.read_parquet()`으로 156M 행 전체 RAM 로드
- 해결: `pq.ParquetFile.iter_batches(batch_size=1_000_000)` + `ParquetWriter` 청크 처리
  + `tmp_path.replace(path)` 원자적 교체

### Bug 3: `ValueError: Input contains NaN` (sklearn 메트릭)
- 원인 A: `load_tensor()`에서 `np.nan_to_num` 누락 → parquet 내 NaN이 텐서에 잔존
- 원인 B: NaN 입력 → loss NaN → 모델 가중치 NaN → 예측값 NaN
- 해결: `np.nan_to_num(tensor, nan=0.0, copy=False)` in `load_tensor()`
  + `compute_metrics()` 시작부에 4개 입력 배열 모두 `nan_to_num`

### Bug 4: Hetero-GAT-GRU / GC-GRU Recall ≈ 0 (분류 붕괴)
- 원인: GAT/GCN 모델이 1:20 불균형에서 분류 손실 신호를 무시
- 해결: `MODEL_CFG_OVERRIDES`로 해당 모델에만 `POS_WEIGHT=60.0` 적용

### Bug 5: DCRNN 학습 매우 느림
- 원인: `for b in range(B):` 배치 루프로 샘플별 그래프 연산 (T×B×3×K = 4608회 Python 루프)
- 해결: `DiffusionConv.forward`를 `torch.einsum('nm,bnf->bmf', support, x_k)`로 벡터화
  → 배치 루프 완전 제거

### Bug 6: 외부 파일 수정이 Jupyter에 반영 안 됨
- 원인: VSCode Jupyter 열린 상태에서 외부 스크립트로 `.ipynb` 수정 시, VSCode가 셀 실행 시 자신의 버전으로 덮어씀
- 해결: 노트북 탭 우클릭 → **Revert File** 후 해당 셀 재실행

### Bug 7: DataLoader `num_workers > 0` 에러
- 원인: Jupyter 환경에서 워커 프로세스가 `FloodDataset` 클래스를 pickle로 import 불가
- 해결: `num_workers=0` 고정 (Jupyter에서는 multi-process DataLoader 불가)

---

## 11. 성능 최적화 이력

### 데이터 로딩 (a010)
```
Before: iterrows() 루프 → 40~50분 (QUICK_RUN=False 기준)
After:  벡터화 numpy 인덱싱 + dt.floor()+groupby + 디스크 캐시 → 수 분 (첫 실행) / 수 초 (캐시 HIT)
```

### 학습 속도 (a006, a012)
```
BATCH: 32 → 128  (배치 수 1641→410, 약 4x 속도 향상)
EPOCHS: 50 → 30  (전체 기간에서 충분히 수렴)
num_workers: 0   (Jupyter 환경 제약 — 변경 불가)
```

### GPU 활용 (a005, a012, a025) — 2026-06-15 추가
```
진단: 학습 중 GPU 사용률이 train 구간만 89~90%, eval 구간 ~18초 동안 4~5%로 idle
원인: eval_epoch이 val 전체(~3,500만 포인트)를 sklearn으로 처리
      → average_precision_score/roc_auc_score/precision_recall_curve 3개 모두 CPU 정렬(O(n log n))
```

무손실 가속 3종 적용:
1. **TF32 (a005)**: `torch.backends.cuda.matmul.allow_tf32=True` + `cudnn.allow_tf32` + `cudnn.benchmark`
   → LSTM/GRU/Linear 행렬곱 가속, FP32 대비 사실상 무손실
2. **텐서 GPU 상주 (a012)**: `TENSORS[split][key].to(DEVICE).float()` → 배치마다 H2D 전송·`.float()` 제거
   → `pin_memory=False` (이미 GPU 상주), `num_workers=0` (GPU 텐서는 멀티워커 불가)
3. **eval 지표 GPU화 (a025)**: sklearn → `torchmetrics.functional`
   (`binary_average_precision` / `binary_auroc` / `binary_precision_recall_curve`, 모두 `thresholds=None`)
   + F1/Precision/Recall은 TP/FP/FN 직접 계산 (CPU sklearn 호출 완전 제거)
   → `eval_epoch`이 예측을 GPU 텐서로 `torch.cat` (`cpu().numpy()` 제거)
   → `save_preds=True`(test 1회)에서만 numpy 변환 (a033b 등 후처리 호환)

```
무손실 검증: 합성 데이터(1:20 불균형)로 sklearn vs torchmetrics 전 지표(36개) 대조
            → 최대 차이 4.28e-07 (부동소수점 한계, 사실상 완전 일치)
가속:       분류 정렬 단독 약 20x (500만 포인트 3.67s → 0.16s)
의존성:     torchmetrics 1.9.0 (pip install torchmetrics)
AMP(FP16):  미적용 — 회귀 타겟(MAE~0.003)이 작아 정밀도 리스크 → 무손실 원칙상 제외
```

> **다지역 확장 시**: compute_metrics(a025)는 입력 shape (samples, N_r, T_out)만 맞으면
> 노드 수가 달라도 그대로 동작. torchmetrics 기반이라 추가 의존성 없음.

---

## 12. 2차 실험 결과 (전체 기간, QUICK_RUN=False)

| 모델 | F1 | RMSE | AUPRC | 분류 | 안정성 |
|------|-----|------|-------|------|--------|
| Baseline-LSTM | ~0.828 | ~0.0036 (1위) | ~0.876 | 정상 | 안정 |
| GC-GRU | ~0.800 | ~0.0056 (5위) | ~0.851 | POS=60 적용 | 불안정 |
| Hetero-GAT-GRU | ~0.829 | ~0.0035 (1~2위) | ~0.874 | POS=60 적용 | 안정 |
| DCRNN | ~0.827 | ~0.0036 (2~3위) | **~0.877 (1위)** | 정상 | 안정 |
| STGCN | — | — | — | **제외** | — |

### 핵심 발견
1. **공간 이득 불명확**: 전체 기간 재학습 후에도 Baseline-LSTM이 GNN과 RMSE 기준 동등하거나 우세
2. **분류 불균형 대응 필수**: GNN 모델은 POS_WEIGHT 기본값(19.75)에서 Recall≈0 붕괴 → 60.0으로 상향
3. **DCRNN 최우수 GNN**: AUPRC 1위, 회귀·분류 균형, 학습 안정, 배치 벡터화 후 속도도 개선

---

## 13. 침수 구간 분리 MAE (a033b 셀)

공간 이득 정밀 검증을 위한 분석 셀이 추가됨.

```python
# flood_mask: 예측 윈도우 내 최소 1개 (노드×스텝)에서 flood_flag=1
flood_mask = (y_cls_true > 0.5).any(axis=(1, 2))

# 침수 구간 MAE가 낮을수록 공간 이득 존재
mae_flood  = np.mean(np.abs(y_reg_true[flood_mask]  - y_reg_pred[flood_mask]))
mae_normal = np.mean(np.abs(y_reg_true[~flood_mask] - y_reg_pred[~flood_mask]))

# 공간 이득 비교: Baseline-LSTM 대비 각 GNN 모델의 침수 MAE 개선율(%)
```

---

## 14. 미결 과제 및 다음 단계

| 우선순위 | 항목 | 상태 |
|---------|------|------|
| 🔴 | 침수 구간 분리 MAE 결과 확인 → 공간 이득 정량화 | 셀 추가 완료, 재학습 후 확인 필요 |
| 🟡 | 그래프 구조 개선: 상류-하류 방향성 반영 | 미착수 |
| 🟡 | 강우량 피처 추가 (AWS 데이터): F_SEWER 9→10 | rainfall_norm 자리 확보됨, 데이터 미확보 |
| 🟢 | GC-GRU 이종 구조 전환 또는 제거 | 동종 근사 한계 확인됨 |
| 🟢 | STGCN T_OUT=6 단기 전용 재설계 후 재포함 | 현재 비교 대상 제외 |
| 🟢 | Hetero-GAT-GRU forward 루프 벡터화 | Python for t in range(T) 제거 |

---

## 15. 전처리 노트북 주요 사항

### preprocessing_rain_join.ipynb — `add_rainfall_norm()`
```python
# 청크 단위 처리 (메모리 크래시 방지)
CHUNK_ROWS = 1_000_000
for batch in pf.iter_batches(batch_size=CHUNK_ROWS):
    ...
writer.close()
tmp_path.replace(path)  # 원자적 교체

# 멱등성 체크 (KeyError 방지)
if 'rainfall_norm' not in old_schema.names:
    new_schema = old_schema.append(pa.field('rainfall_norm', pa.float32()))
```

---

## 16. 주요 참조 문서

| 문서 | 내용 |
|------|------|
| `EVALUATION_FRAMEWORK.md` | 성능 지표 체계 (MAE/RMSE/AUPRC/CSI/운영지표) |
| `PROJECT_ANALYSIS.md` | 전처리 파이프라인 (Step 01~11 상세) |
| `parquet_preprocessing_guide.md` | Parquet 파일 처리 가이드 |
