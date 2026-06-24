# EDA 기반 전처리 계획

> ⚠️ **상태(2026-06-24 기준)**: 이 문서는 작성 시점 기준 기록이다. **현재 프로젝트 방향·결론의 정본은 `CLAUDE.md` · `reports/progress_report.html`(§1~18) · `docs/PROGRESS_LOG.md`**. EDA 사실·전처리 방법은 유효하나, 방향은 **"GNN 침수예측 → 위험수위 분석 → 데이터 확보(고객=KICT)"**로 이동했고 **관악 침수=국지 호우 표면류**(하수 만관·하천역류 아님)로 규명됨.


> 작성일: 2026-06-17  
> 기준 자료: `eda_raw.ipynb`, `eda_raw_report.html`, `DATA_QUALITY_FINDINGS.md`  
> 목적: 기존 `flood_flag = level > 0` 전제를 폐기하고, 원본 측정값 기반의 방어 가능한 침수 라벨과 모델 입력 데이터를 다시 구성한다.

---

## 0. 결론 요약

EDA 결과상 전처리의 핵심은 단순 결측 보간이나 스케일링이 아니라 다음 3가지다.

1. 도로 수위 센서의 baseline, stuck value, saturation 값을 분리한다.
2. 침수 라벨을 `level > 0`이 아니라 `수위 임계값 + 강우 검증 + 품질 필터` 기준으로 재정의한다.
3. 도로, 하수, 강우를 10분 기준 패널 데이터로 통합한다.

우선 권장 라벨 기본안은 아래와 같다.

```text
flood_candidate_t6 =
    road_level >= 6cm
    AND road_level != 96
    AND road_level < 999
    AND recent_rainfall_sum > 0
```

단, 운영 침수 기준 `T`는 고객 기준이 필요하므로 `T=2`, `T=6`, `T=10` 후보를 모두 생성해 비교한다.

---

## 1. 전처리 원칙

### 1-1. 원본 보존

원본 측정값은 가능한 한 수정하지 않는다. 값 삭제, 보간, 라벨 재정의는 별도 산출물에서 수행한다.

```text
원본 보존 레이어:
- road_raw: sensor_id, timestamp, level
- sewer_raw: sensor_id, timestamp, level, comm_status, 위치정보
- rain_raw: station_id, timestamp, rainfall_mm
```

권장 방식:

- 원본 parquet은 그대로 유지한다.
- 정제 결과는 `cleaned`, `features`, `labels`, `panel` 계층으로 분리한다.
- 제거 대상 값도 가능하면 먼저 flag로 남기고, 학습용 view에서 제외한다.

### 1-2. 삭제보다 플래그 우선

센서 데이터는 추후 기준 변경 가능성이 크다. 따라서 전처리 초반에는 데이터를 바로 삭제하지 않고 품질 플래그를 붙인다.

예시:

```text
is_saturated_96
is_extreme_ge_999
is_baseline_noise
is_stuck_segment
is_missing_window
is_training_usable
```

최종 모델 입력 테이블에서는 이 플래그를 기준으로 제외 또는 마스킹한다.

---

## 2. 권장 산출물 구조

새 전처리 결과는 기존 산출물과 충돌하지 않도록 별도 디렉터리에 생성한다.

```text
dataset/processed/eda_based/
├── road_cleaned.parquet
├── road_sensor_quality.parquet
├── road_label_candidates.parquet
├── sewer_cleaned.parquet
├── sewer_features_10min.parquet
├── rain_features_10min.parquet
├── training_panel_10min.parquet
├── label_summary.parquet
└── preprocessing_config.json
```

각 산출물의 역할:

| 파일 | 역할 |
|------|------|
| `road_cleaned.parquet` | 도로 원본 수위에 품질 플래그와 baseline 보정값을 추가 |
| `road_sensor_quality.parquet` | 센서별 결측률, stuck 비율, saturation 비율, 사용 가능 여부 |
| `road_label_candidates.parquet` | `T=2`, `T=6`, `T=10` 기준 라벨 후보 |
| `sewer_cleaned.parquet` | 하수 수위 품질 플래그 및 10분 집계 전 중간 결과 |
| `sewer_features_10min.parquet` | 하수 10분 단위 평균, 최대, 변화량 피처 |
| `rain_features_10min.parquet` | 강우 10분 값 및 rolling sum 피처 |
| `training_panel_10min.parquet` | 도로-하수-강우-라벨을 결합한 모델용 패널 |
| `label_summary.parquet` | 라벨 기준별 건수, 센서 수, 월별 분포, 강우 동반율 |
| `preprocessing_config.json` | 임계값, rolling window, 필터 기준 기록 |

---

## 3. Step 1. 도로 수위 정제

### 3-1. 기본 컬럼

입력:

```text
dataset/processed/raw_parquet/road/*.parquet
```

필수 컬럼:

```text
sensor_id
timestamp
level
```

추가할 컬럼:

```text
level_raw
level_clean
sensor_baseline
level_adj
is_zero
is_level_one
is_saturated_96
is_extreme_ge_999
is_invalid_level
```

### 3-2. 도로 level 처리 규칙

| 조건 | 처리 | 이유 |
|------|------|------|
| `level == 0` | 정상 dry 후보 | 도로 관측치의 대부분 |
| `level == 1` | `is_level_one=1`, 침수 라벨 제외 | baseline noise 가능성 높음 |
| `level == 96` | `is_saturated_96=1`, 학습 라벨 제외 | 측정범위 천장 saturation 오류로 판단 |
| `level >= 999` | `is_extreme_ge_999=1`, 학습 라벨 제외 | 물리적으로 비정상적인 극단값 |
| `level < 0` | `is_invalid_level=1`, 제외 | 물리적으로 불가능 |

`level_clean` 권장 정의:

```text
level_clean =
    null if is_saturated_96 or is_extreme_ge_999 or level < 0
    else level_raw
```

단, 원본 추적을 위해 `level_raw`는 항상 유지한다.

### 3-3. 센서별 baseline

센서별 baseline은 학습 구간에서 계산한다. 기본안은 센서별 최빈값 또는 하위 분위수다.

```text
sensor_baseline_mode = mode(level_raw where valid)
sensor_baseline_q10  = quantile(level_raw where valid, 0.10)
sensor_baseline      = min(sensor_baseline_mode, sensor_baseline_q10)
level_adj            = max(level_clean - sensor_baseline, 0)
```

주의:

- `level == 96`, `level >= 999`는 baseline 계산에서 제외한다.
- baseline이 1cm 이하인 센서는 대부분 정상으로 볼 수 있으나, `level == 1` 비율이 큰 센서는 별도 품질 플래그를 둔다.

---

## 4. Step 2. 도로 센서 품질 테이블

센서별로 아래 지표를 계산한다.

```text
n_obs
t0
t1
active_days
missing_pct_1min
zero_pct
level_one_pct
saturated_96_pct
extreme_ge_999_pct
unique_level_count
top1_value
top1_pct
top3_pct
max_level
p99_level
is_stuck_sensor
is_label_usable
is_feature_usable
```

### 4-1. stuck 센서 판정 후보

초기 기준:

```text
is_stuck_sensor =
    top3_pct >= 0.80
    AND unique_level_count <= 5
    AND nonzero_pct is high
```

단, 도로 센서 자체가 이산 단계형일 수 있으므로 stuck 판정은 라벨 제외에 우선 사용하고, 입력 피처 제외는 더 보수적으로 결정한다.

### 4-2. 사용 가능성 구분

센서를 3그룹으로 나눈다.

| 그룹 | 의미 | 사용 방식 |
|------|------|-----------|
| A | 라벨과 입력 모두 사용 가능 | 모델 학습/평가에 사용 |
| B | 입력 피처로는 가능하나 라벨은 불안 | 타겟 생성에서는 제외, 보조 피처로 검토 |
| C | 품질 불량 | 학습에서 제외 또는 별도 분석 |

초기 판정안:

```text
A: missing_pct <= 50%, saturated_96_pct 낮음, extreme_ge_999_pct 낮음, stuck 아님
B: missing_pct <= 50%이나 stuck 또는 baseline noise 의심
C: missing_pct > 50% 또는 extreme/saturation 비율 과다
```

---

## 5. Step 3. 강우 피처 생성

입력:

```text
dataset/features/rain/*/rain_*.parquet
```

기준 해상도:

```text
10분
```

기본 컬럼:

```text
station_id
timestamp
rainfall_mm
```

생성 피처:

```text
rain_10m
rain_30m_sum
rain_1h_sum
rain_3h_sum
rain_6h_sum
rain_12h_sum
rain_1h_max
rain_3h_max
rain_6h_max
is_raining_10m
is_recent_rain_1h
is_recent_rain_3h
is_recent_rain_6h
```

권장 정의:

```text
is_recent_rain_6h = rain_6h_sum >= 1.0
```

라벨 검증에는 우선 `rain_6h_sum >= 1mm`를 사용하고, 민감도 분석에서 `rain_1h_sum`, `rain_3h_sum`도 비교한다.

---

## 6. Step 4. 도로-강우 매핑

도로 센서별 가장 가까운 강우 관측소를 연결한다.

필요 산출물:

```text
road_rain_mapping.parquet
```

필수 컬럼:

```text
road_sensor_id
station_id
distance_km
```

처리 원칙:

- 기존 매핑이 있으면 재사용한다.
- 매핑 거리가 큰 센서는 `rain_mapping_quality`를 낮게 둔다.
- 국소 강우 과소평가 가능성을 감안해, 라벨 조건에서 강우는 hard filter와 soft flag 두 버전을 모두 만든다.

라벨 후보:

```text
flood_t6_hard_rain = level >= 6 AND valid_level AND rain_6h_sum >= 1
flood_t6_soft_rain = level >= 6 AND valid_level
```

---

## 7. Step 5. 침수 라벨 후보 생성

기존 라벨:

```text
flood_flag = level > 0
```

위 정의는 폐기한다. 대신 여러 기준을 동시에 생성한다.

### 7-1. 절대 수위 기준

```text
flood_abs_t2  = level_clean >= 2
flood_abs_t6  = level_clean >= 6
flood_abs_t10 = level_clean >= 10
```

### 7-2. baseline 보정 기준

```text
flood_adj_t2  = level_adj >= 2
flood_adj_t5  = level_adj >= 5
flood_adj_t10 = level_adj >= 10
```

### 7-3. 강우 검증 기준

```text
flood_abs_t6_rain6h =
    flood_abs_t6
    AND rain_6h_sum >= 1
    AND NOT is_saturated_96
    AND NOT is_extreme_ge_999

flood_adj_t5_rain6h =
    flood_adj_t5
    AND rain_6h_sum >= 1
    AND NOT is_saturated_96
    AND NOT is_extreme_ge_999
```

### 7-4. 기본 추천 라벨

초기 모델링 기본 라벨:

```text
target_primary = flood_abs_t6_rain6h
```

비교 라벨:

```text
target_loose  = flood_abs_t2_rain6h
target_strict = flood_abs_t10_rain6h
target_adj    = flood_adj_t5_rain6h
```

---

## 8. Step 6. 시간 해상도 통일

모델 입력 기준 시간 해상도는 10분으로 둔다.

### 8-1. 도로 집계

도로는 침수 peak가 중요하므로 평균보다 최대값을 우선한다.

```text
road_level_max_10m
road_level_mean_10m
road_level_last_10m
road_valid_obs_count_10m
road_missing_flag_10m
```

라벨은 기본적으로 10분 window의 최대 수위 기준으로 생성한다.

```text
road_label_level = max(level_clean within 10min)
```

### 8-2. 하수 집계

하수는 연속 수위 흐름을 보조 피처로 사용한다.

```text
sewer_level_mean_10m
sewer_level_max_10m
sewer_level_min_10m
sewer_level_last_10m
sewer_level_diff_10m
sewer_level_roll_1h_mean
sewer_level_roll_3h_mean
sewer_level_roll_6h_mean
```

하수 `level > 0`은 정상 기저유량일 수 있으므로 라벨로 사용하지 않는다.

---

## 9. Step 7. 모델용 패널 생성

최종 모델용 테이블은 10분 단위로 구성한다.

권장 키:

```text
timestamp_10m
road_sensor_id
```

기본 컬럼 그룹:

```text
도로 피처:
- road_level_max_10m
- road_level_adj_max_10m
- road_level_mean_10m
- road_quality_flags

강우 피처:
- nearest_station_id
- rain_10m
- rain_1h_sum
- rain_3h_sum
- rain_6h_sum
- rain_6h_max

하수 피처:
- 인접 또는 매핑된 하수 센서의 level_mean/max/diff
- 필요 시 복수 하수 센서 집계값

라벨:
- target_primary
- target_loose
- target_strict
- target_adj
```

### 9-1. 하수 결합 방식

하수는 도로 침수의 강한 선행 지표로 확인되지 않았으므로, 초기 모델에서는 보조 피처로만 사용한다.

권장 순서:

1. 강우 + 도로 자기이력 baseline 모델
2. 강우 + 도로 자기이력 + 하수 피처 모델
3. 하수 추가가 성능을 실제로 올리는지 비교

---

## 10. Step 8. Split 기준

현재 공통 기간 기준을 유지한다.

```text
Train: 2024-01-01 ~ 2024-12-31
Val  : 2025-01-01 ~ 2025-05-31
Test : 2025-06-01 ~ 2025-08-31
```

주의:

- 라벨이 매우 희소하므로 split별 양성 이벤트 수를 반드시 확인한다.
- 센서 단위 누수가 없도록 baseline 계산은 train 기준을 우선한다.
- 정규화 파라미터도 train에서만 fit하고 val/test에 적용한다.

---

## 11. Step 9. 검증 리포트

전처리 후 반드시 아래 지표를 산출한다.

### 11-1. 라벨 검증

| 항목 | 확인 내용 |
|------|-----------|
| 기준별 양성 수 | `T=2`, `T=6`, `T=10`, baseline 보정 기준 비교 |
| 월별 분포 | 여름철 집중도가 물리적으로 타당한지 확인 |
| 센서별 분포 | 특정 센서가 라벨을 독점하는지 확인 |
| 강우 동반율 | 라벨 양성 시 직전 강우가 존재하는지 확인 |
| saturation 포함 여부 | `96`, `999`, `1000`이 라벨에 들어가지 않는지 확인 |

### 11-2. 품질 검증

```text
road_sensor_quality_summary
road_label_summary_by_month
road_label_summary_by_sensor
rain_label_coupling_summary
split_positive_counts
```

### 11-3. 통과 기준 초안

```text
- target_primary에서 level == 96, level >= 999 포함 0건
- split별 양성 이벤트가 최소 1개 이상 존재
- 상위 1개 센서가 전체 양성의 50% 이상을 독점하지 않음
- 양성 라벨의 rain_6h_sum >= 1mm 비율 100%
- target_loose/primary/strict 간 건수 차이가 리포트됨
```

---

## 12. 구현 순서

우선순위 기준 실행 순서:

```text
1. road_cleaned.parquet 생성
2. road_sensor_quality.parquet 생성
3. rain_features_10min.parquet 생성
4. road_rain_mapping 재사용 또는 검증
5. road_label_candidates.parquet 생성
6. label_summary.parquet 생성
7. sewer_features_10min.parquet 생성
8. training_panel_10min.parquet 생성
9. split별 정규화 파라미터와 config 생성
10. 모델 학습 전 검증 리포트 생성
```

최소 1차 목표:

```text
road_cleaned.parquet
road_sensor_quality.parquet
rain_features_10min.parquet
road_label_candidates.parquet
label_summary.parquet
```

이 5개가 만들어지면, 실제 모델링 전에 라벨 기준이 물리적으로 타당한지 먼저 판단할 수 있다.

---

## 13. 결정이 필요한 항목

아래 항목은 전처리 구현 중 config로 관리하고, 실험 결과를 보고 확정한다.

| 항목 | 기본값 | 후보 |
|------|--------|------|
| 도로 침수 절대 임계값 | `6cm` | `2cm`, `6cm`, `10cm` |
| baseline 보정 임계값 | `5cm` | `2cm`, `5cm`, `10cm` |
| 강우 검증 시간창 | `6h` | `1h`, `3h`, `6h`, `12h` |
| 강우 검증 임계값 | `1mm` | `>0mm`, `1mm`, `5mm` |
| 시간 해상도 | `10min` | `5min`, `10min`, `30min` |
| `level == 96` 처리 | 라벨 제외 | 삭제, flag only |
| `level >= 999` 처리 | 라벨 제외 | 삭제, flag only |
| stuck 센서 처리 | 라벨 제외 우선 | 전체 제외, 입력만 사용 |

---

## 14. 모델링 전 합의안

현재 EDA 기준으로는 다음 전처리 합의안을 기본값으로 둔다.

```text
1. 원본 parquet은 보존한다.
2. 도로 `level > 0` 라벨은 사용하지 않는다.
3. 도로 `level == 1`은 baseline noise 후보로 보고 침수 라벨에서 제외한다.
4. 도로 `level == 96`, `level >= 999`는 비정상값으로 보고 라벨에서 제외한다.
5. 기본 침수 라벨은 `level >= 6cm AND rain_6h_sum >= 1mm`로 시작한다.
6. 비교를 위해 `T=2`, `T=10`, baseline 보정 라벨도 함께 만든다.
7. 모든 입력은 10분 단위로 통일한다.
8. 도로는 10분 max를 라벨 기준으로 사용한다.
9. 하수는 라벨이 아니라 보조 피처로 사용한다.
10. split별 양성 수와 월별/센서별 라벨 분포를 확인한 뒤 모델 학습에 들어간다.
```

