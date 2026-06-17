# 전처리 기준 및 개선사항 정리

이 문서는 현재 정리된 도심 침수 예측 GNN 전처리 기준, 처리 과정, 그리고 추가 개선사항을 요약한다.

> 주의: 이 문서는 전처리 코드와 설정 기준으로 작성했으며, parquet/csv 내부 데이터 내용은 열람하지 않았다.

## 0. 실행 순서

### A. 전체 파이프라인 (처음부터)

```text
1. preprocessing.ipynb  — Step 01~10
   원본 변환 → 스키마 통일 → 이상치/보간 → 제원표 조인 → 교차상관
   → 파생 피처 → 인접행렬 → 공통기간 분리 → 정규화 → train/val/test 분리
2. preprocessing_rain.ipynb
   서울시 1분 강수량 → 관측소별 10분 합산 → rainfall_norm
3. preprocessing_rain_join.ipynb
   하수관로 센서별 최근접 관측소 매핑
   → sewer_normalized.parquet 및 train/val/test split에 rainfall_norm 추가
4. preprocessing.ipynb  — Step 11
   gnn_config.json 생성 (피처 목록, 그래프 정보, split 기간, 클래스 가중치)
5. model_comparison.ipynb  — 맨 위 셀부터 순차 실행
```

> 의존성: 강우 조인(3)은 정규화/분리(Step 09~10) 이후에만 가능하다.
> `preprocessing_rain`(2)은 관측소 데이터만 쓰므로 1과 병행 가능하다.

### B. 피처 변경(season 제거 / flood_stage 입력 제외)만 반영 — 재전처리 불필요

`gnn_config.json`은 이미 갱신되어 있고, `load_tensor()`는 config의 `feat_cols`만
읽으므로 parquet을 다시 만들지 않아도 모델만 재실행하면 된다.
(season은 읽지 않고, flood_stage는 입력에서 제외됨)

```text
1. (필요시) 텐서 캐시 무효화
   rm dataset/processed/tensor_cache/*.pt
   ※ feat_cols 변경으로 캐시 키가 달라져 a010 재실행 시 자동 재생성됨
2. VSCode: 두 노트북 탭 우클릭 → "Revert File"  (외부 편집 반영)
3. model_comparison.ipynb — 커널 재시작 후 a005부터 순차 실행
   a005(TF32) → a006(CFG, F_SEWER=9 / F_ROAD=8 자동) → a008(그래프)
   → a010(텐서·새 캐시) → a012(Dataset/Loader) → a015~a023(모델)
   → a025(metrics) → a026(run_experiment) → a028(학습) → a030~(결과)
```

### C. season을 parquet에서도 물리적으로 제거 (선택)

모델 동작에는 영향이 없지만 parquet 컬럼까지 없애려면
`preprocessing.ipynb` Step 06~11을 재실행한다 (B의 캐시 무효화 포함).

## 1. 현재 전처리 기준

### 1-1. 기준 산출물 경로

현재 전처리 산출물의 기준 경로는 `dataset/features/`이다.

```text
dataset/features/
├── sewer/
├── road/
├── rain/
├── overlap/
├── adjacency_expanded.parquet
├── sewer_sewer_edges.parquet
├── sewer_node_index.parquet
├── road_node_index.parquet
└── gnn_config.json
```

### 1-2. 학습 기간 기준

현재 공통 학습 기간은 도로노면 수위와 하수관로 수위가 함께 존재하는 구간을 기준으로 한다.

```text
전체 공통 기간: 2024-01-01 ~ 2025-08-31

Train: 2024-01-01 ~ 2024-12-31
Val  : 2025-01-01 ~ 2025-05-31
Test : 2025-06-01 ~ 2025-08-31
```

### 1-3. 시간 해상도

모델 입력 데이터는 10분 단위로 정렬한다.

```text
해상도: 10분
T_in : 6 step = 과거 60분
T_out: 18 step = 미래 180분
```

`gnn_config.json`의 `output_steps`와 모델 노트북의 `T_OUT`은 **18로 통일**되었다.

## 2. 전처리 과정

### 2-1. 원본 변환

하수관로 CSV와 도로노면 TXT를 parquet 형식으로 변환한다.

### 2-2. 스키마 통일

한글/영문 컬럼명을 통일하고 기본 컬럼을 정리한다.

```text
sensor_id
timestamp
level
```

### 2-3. 이상치 처리 및 보간

하수관로 수위:

- 음수 수위 제거
- 센서별 측정범위 초과값 제거
- 10분 이하 결측은 선형 보간

도로노면 수위:

- 오류코드 제거
- 10분 이하 결측은 선형 보간

### 2-4. 노드 메타데이터 조인

제원표를 이용해 센서별 좌표, 배수구역, 관규격, 측정범위 등을 조인한다.

### 2-5. 그래프 후보 생성

하수관로 센서와 도로노면 센서 쌍을 배수구역과 거리 기준으로 구성한다.

### 2-6. 교차상관 분석

그래프 구조에 test 기간 정보가 섞이지 않도록, 교차상관 분석은 train 기간 내부 장마철만 사용한다.

```text
교차상관 분석 기간: 2024-06-01 ~ 2024-10-01
```

이 기준은 test 기간인 `2025-06-01 ~ 2025-08-31`의 정보 누수를 막기 위한 것이다.

### 2-7. 그래프 엣지 생성

생성되는 주요 엣지는 다음과 같다.

- 하수관로 → 도로노면 엣지
- 하수관로 → 하수관로 엣지
- 고립 도로 센서에 대한 fallback 엣지

거리 기반 Gaussian weight를 사용하며, 고립 도로 센서는 2km 이내 최근접 하수관로 센서와 연결한다.

### 2-8. 피처 생성

하수관로 피처 (F_SEWER=9):

```text
level_norm
level_diff_norm
fill_rate
hour_sin
hour_cos
month_sin
month_cos
is_weekend
rainfall_norm
```

도로노면 입력 피처 (F_ROAD=8):

```text
level_norm
level_diff_norm
flood_flag
hour_sin
hour_cos
month_sin
month_cos
is_weekend
```

> `season`은 `month`와 중복되어 sewer/road 모두에서 제거했다.
> `flood_stage`는 parquet 컬럼과 멀티클래스 타겟(`road_multiclass`)으로 보존하되,
> `level_norm`/`flood_flag`와 중복인 **입력 피처에서는 제외**했다.

### 2-9. 정규화

정규화 파라미터는 train 기간만 사용해 계산한다.

Val/Test에는 train에서 계산한 정규화 파라미터를 그대로 적용한다.

### 2-10. 강우 전처리

서울시 1분 강수량 데이터를 관측소별 10분 단위로 합산한다.

```text
1분 강수량
→ 관측소별 10분 합산
→ rainfall_mm
→ rainfall_norm
```

현재 강우 정규화 기준은 다음과 같다.

```text
rainfall_norm = clip(rainfall_mm * 6 / 100, 0, 1)
```

즉, 10분 강수량을 시간당 강우강도로 환산한 뒤 `100mm/hr` 기준으로 0~1 범위에 맞춘다.

### 2-11. 강우-하수관로 조인

하수관로 센서별 최근접 강우 관측소를 매핑하고, 하수관로 split 파일에 `rainfall_norm`을 추가한다.

수정된 기준에서는 split 파일뿐 아니라 아래 파일도 split 산출물 기준으로 다시 재구성한다.

```text
dataset/features/overlap/sewer_normalized.parquet
```

### 2-12. 최종 설정

`dataset/features/gnn_config.json`에 feature 목록, 그래프 정보, split 기간, 강우 피처 상태를 기록한다.

## 3. 최근 정리된 사항

최근 전처리 기준에서 정리한 내용은 다음과 같다.

```text
1. dataset/features 경로로 통일
2. 그래프 교차상관 분석에서 test 기간 제거
3. 강우 조인 후 sewer_normalized.parquet도 재구성
4. parquet 내부 내용을 출력하는 검증 셀 제거
5. 전처리 노트북의 과거 실행 출력 제거
6. season 피처 제거 (month와 중복) — sewer/road 모두
7. flood_stage 입력 피처에서 제외 (parquet·멀티클래스 타겟은 보존)
8. gnn_config.json output_steps를 18로 통일 (T_OUT과 일치)
```

## 4. 추가 개선사항

### 4-1. 강우 결측 mask 추가

현재 강우 결측과 무강우가 모두 `0.0`으로 들어갈 수 있다.

개선안:

```text
rainfall_norm
rain_missing
```

`rain_missing=1`이면 관측소/시간 매칭 실패 또는 원자료 결측, `0`이면 정상 관측으로 구분한다.

### 4-2. 누적 강우 피처 추가

침수는 순간 강우보다 누적 강우의 영향을 크게 받을 수 있다.

추천 피처:

```text
rain_10min
rain_30min_sum
rain_60min_sum
rain_lag_10min
rain_lag_30min
```

### 4-3. 강우 지연 반응 반영

하수 수위는 강우보다 몇 분에서 수십 분 늦게 반응할 수 있다.

따라서 현재 시점 강우뿐 아니라 과거 강우 lag 피처를 함께 넣는 것이 좋다.

### 4-4. Road feature leakage 점검

도로 feature에는 `flood_flag`, `flood_stage`가 포함되어 있다.

과거 입력창 안의 값이면 사용할 수 있지만, 미래 target window와 겹치면 leakage가 된다. Dataset slicing에서 다음 구조가 유지되는지 확인해야 한다.

```text
입력: road[t : t + T_in]
타겟: road[t + T_in : t + T_in + T_out]
```

### 4-5. Split 설정 단일화

문서, 전처리 노트북, 모델 실험 노트북에 split 기준이 섞이지 않도록 설정 파일을 하나로 분리하는 것이 좋다.

예:

```text
config/preprocessing_config.json
```

포함할 기준:

```text
데이터 경로
train/val/test 기간
시간 해상도
T_in
T_out
강우 피처 사용 여부
그래프 생성 기준
```

### 4-6. 전처리 manifest 생성

산출물마다 재현성 정보를 남기는 manifest를 생성하는 것이 좋다.

기록할 항목:

```text
생성 시각
입력 기간
split 기간
feature 목록
정규화 기준
강우 조인 여부
그래프 생성 기준
코드 버전/hash
```

### 4-7. 노트북에서 스크립트로 분리

현재 파이프라인은 노트북 셀 실행 순서에 민감하다.

장기적으로는 핵심 전처리를 스크립트로 분리하는 것이 안정적이다.

예:

```text
scripts/01_build_clean.py
scripts/02_build_graph.py
scripts/03_build_rain.py
scripts/04_join_rain.py
scripts/05_build_config.py
```

## 5. 분석된 개선사항 종합

> 아래는 전처리 코드와 설정 기준 분석을 통해 도출된 개선사항이다. parquet/csv 내부 데이터 내용은 열람하지 않았다.

### 5-1. 데이터 누수 (Leakage) 관련

| 항목 | 문제 | 현재 상태 | 권장사항 |
|------|------|------|---------|
| 교차상관 분석 | test 기간 정보가 섞일 위험 | ✅ train 기간 내부 장마철(2024-06~2024-10)만 사용 | 유지 |
| Road feature leakage | `flood_flag`, `flood_stage`가 미래 target window와 겹칠 수 있음 | 현재 Dataset slicing은 입력/타겟이 분리된 구조 | 운영 목적에 따라 road 과거 상태 사용 여부를 실험군으로 분리 |

### 5-2. 강우 피처 관련 (가장 시급)

| 항목 | 현재 상태 | 권장 사항 | 우선순위 |
|------|----------|------|-------|
| rain_missing | 미구현 | 관측소/시간 매칭 실패 또는 원자료 결측을 `1`로 마스킹 | 🔴 1순위 |
| 누적 강우 피처 | 미구현 | `rain_30min_sum`, `rain_60min_sum` 추가 | 🔴 1순위 |
| 강우 지연 반응 | 미고려 | `rain_lag_10min`, `rain_lag_30min` 추가 | 🟡 2순위 |

**권장 추가 피처:**

```text
rainfall_norm       # 현재 정규화 강우
rain_missing        # 결측 마스킹 (새로 추가)
rain_30min_sum      # 최근 30분 누적 강우 (새로 추가)
rain_60min_sum      # 최근 60분 누적 강우 (새로 추가)
rain_lag_10min      # 10분 전 강우 (새로 추가)
rain_lag_30min      # 30분 전 강우 (새로 추가)
```

### 5-3. 설정/분할 불일치 점검

| 항목 | preprocessing_summary.md | PREPROCESSING_CURRENT_STANDARD.md | 상태 |
|------|------|------|------|
| 학습 시작일 | 2022-01 | 2024-01 | ⚠️ 불일치 |
| 학습 종료일 | 2025-09 | 2025-08-31 | ⚠️ 불일치 |
| T_out | 미정 | 18 | ✅ output_steps=18 통일 |

**권장 조치:**
1. 실제 사용 기간을 확인하고 문서 통일
2. ~~`gnn_config.json`의 `output_steps`와 모델 노트북의 `T_OUT`값 일치시킴~~ → 완료 (18)

### 5-4. 구조적 개선 (장기)

| 항목 | 권장사항 | 기대효과 |
|------|---------|------|
| 설정 파일 분리 | `config/preprocessing_config.json`에 모든 split/해상도/피처 설정 통합 | 설정 관리 용이 |
| manifest 생성 | 산출물마다 재현성 정보 남김 | 디버깅/검증 용이 |
| 노트북 → 스크립트 | 핵심 전처리를 스크립트로 분리 | 재현성/안정성 향상 |

### 5-5. 개선 우선순위 요약

```
최우선 정리:
   - 학습 기간/T_out 설정 불일치 확인 및 통일

🔴 즉시 반영:
   - rain_missing 피처 추가
   - 누적 강우 피처 (rain_30min_sum, rain_60min_sum) 추가

🟡 다음 단계:
   - 강우 lag 피처 추가
   - Road 입력 feature 실험군 분리

🟢 장기 과제:
   - 설정 파일 단일화
   - 전처리 manifest 생성
   - 노트북 → 스크립트 분리
```

## 6. 타당성 검토 및 조정 의견

다른 분석가가 정리한 추가 개선사항은 전반적으로 타당하다. 특히 강우 결측 마스킹, 누적 강우 피처, 설정 불일치 정리는 현재 전처리 안정성과 모델 해석력을 높이는 데 직접적으로 도움이 된다.

다만 실제 작업 순서는 다음처럼 조정하는 것이 더 안전하다.

```text
0순위: T_out, split 기간, 기준 문서 통일
1순위: rain_missing 추가
2순위: rain_30min_sum, rain_60min_sum 추가
3순위: rain_lag_10min, rain_lag_30min 추가
4순위: road 입력 feature 실험군 분리
5순위: config/manifest/script화
```

Road feature leakage의 경우 현재 모델 Dataset 구조는 입력창과 타겟창을 분리한다.

```text
입력: road[t : t + T_in]
타겟: road[t + T_in : t + T_in + T_out]
```

따라서 `flood_flag`, `flood_stage`가 곧바로 미래 leakage를 일으키는 구조는 아니다. 다만 운영 목표가 도로 센서 반응 전 조기예측이라면, 과거 도로 상태를 사용하는 모델과 사용하지 않는 모델을 별도 실험군으로 분리해야 한다.

강우 누적 피처는 모델이 60분 입력 시퀀스에서 이론적으로 학습할 수 있는 정보이지만, 명시적인 누적 피처로 제공하면 학습 안정성과 해석 가능성이 좋아질 수 있다. 반면 lag 피처는 누적 강우와 목적이 다르므로 1순위 누적 피처와 분리해 2순위로 두는 것이 적절하다.
