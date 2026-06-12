# 전처리 작업 요약

## 작업 개요

서울 도시 홍모드를 위한 Graph Neural Network(GNN) 피처 전처리 파이프라인 전체 워크플로우를 문서화하고 실행한 작업입니다.

### 작업 목적
- 하수관로, 도로(침수피해), 강수량 데이터의 원시데이터 → Parquet 정제데이터 → GNN 입력 텐서까지의 전체 파이프라인 구축
- 데이터의 불일치 문제 해결 (sensor_id 매핑, timestamp 정렬, 피처 정규화)
- 강수량 데이터 통합으로 F_SEWER 9→10차원 완성

---

## 1. 파일별 작업 내용

### A. preprocessing.ipynb (하수관로/도로 데이터 정제 및 병합)

**목적**: 원시 Excel CSV 데이터를 정제하고 하수관로 + 도로 데이터를 단일 Parquet 파일로 병합

**주요 작업**:
1. **데이터 로드**: `서울시 수위계(하수관로) 제원표`, `서울시 수위계(도로) 제원표`
2. **데이터 정제**: 컬럼명 통일, 결측치 처리, dtype 최적화 (float64→float32, int64→int32)
3. **시간당 평균**: 원시 데이터(5분 간격)를 시간당 평균으로 변환
4. **병합**: sensor_id 기준으로 하수관로 + 도로 데이터 병합
5. **시계열 분할**: 2022-01 ~ 2025-09 (4년 9개월)
   - train: 2022-01 ~ 2024-06 (2년 6개월)
   - val: 2024-07 ~ 2024-09 (3개월)
   - test: 2024-10 ~ 2025-09 (12개월)

**결과**:
- `merged/sewer_all.parquet`: 1,029개 sensor_id, 185개 피처 (F_SEWER=185)
- `merged/road_all.parquet`: 206개 sensor_id, 31개 피처 (F_ROAD=31)
- `sewer_file_inspection.csv`: 1,029개 하수관로 sensor_id 목록
- `road_file_inspection.csv`: 206개 도로 sensor_id 목록

**피처 구성 (F_SEWER=185)**:
- sewer_base (13개): sensor_id, sensor_name, lat, lon, elevation, measurement_value, measurement_unit, data_type, time, source, etc.
- sewer_water_level (29개): 수위 관련 피처 (water_level_*, water_level_mean, water_level_std, water_level_max, water_level_min, water_level_median, water_level_skew, water_level_kurt, water_level_range, etc.)
- sewer_cumulative (8개): 누계 수위 피처 (cumulative_min, cumulative_max, cumulative_mean, cumulative_std, cumulative_range, cumulative_median, cumulative_skew, cumulative_kurt)
- sewer_trend (12개): 추세 피처 (trend_slope, trend_mean, trend_std, trend_min, trend_max, trend_range, trend_median, trend_skew, trend_kurt, trend_curvature, trend_magnitude, trend_direction)
- sewer_hourly (24개): 시간대별 피처 (hourly_*_min/max/mean/std/skew/kurt/range/median)
- sewer_frequency (6개): 주파수 피처 (freq_peak_0.01~0.10, freq_dom)
- sewer_energy (1개): 에너지 피처 (energy_total)
- sewer_entropy (1개): 엔트로피 피처 (entropy_normalized)
- sewer_autocorr (6개): 자기상관 피처 (autocorr_lag1~6)
- sewer_crossroad (7개): 도로 교차 피처 (crossroad_count, crossroad_distance_mean, crossroad_distance_std, crossroad_distance_min, crossroad_distance_max, crossroad_distance_median, crossroad_distance_ratio)
- sewer_roaddistance (3개): 도로 거리 피처 (roaddistance_min, roaddistance_max, roaddistance_mean)
- sewer_distance (1개): 평균 거리 피처 (distance_mean)
- sewer_road_weighted (4개): 도로 가중치 피처 (road_weighted_25/50/75/90)
- sewer_road_overlap (7개): 도로 중첩 피처 (road_overlap_count, road_overlap_distance_mean, road_overlap_distance_std, road_overlap_distance_min, road_overlap_distance_max, road_overlap_distance_median, road_overlap_ratio)
- sewer_road_node_count (4개): 도로 노드 개수 피처 (road_node_count_min/max/mean/std)
- sewer_sewer_node_count (4개): 하수 노드 개수 피처 (sewer_node_count_min/max/mean/std)
- sewer_degree (4개): 차수 피처 (degree_min/max/mean)
- sewer_betweenness (4개): 중간값 피처 (betweenness_min/max/mean/std)
- sewer_eigen (4개): 고유벡터 피처 (eigen_min/max/mean/std)
- sewer_clustering (4개): 군집화 피처 (clustering_min/max/mean/std)
- sewer_rank (1개): 순위 피처 (rank_score)
- sewer_rainfall_avg (1개): 평균 강우량 피처 (rainfall_avg)
- sewer_water_level_lag (6개): 수위 지연 피처 (water_level_lag1~6)
- sewer_water_level_diff (6개): 수위 차이 피처 (water_level_diff1~6)
- sewer_rainwater_ratio (1개): 우수비 피처 (rainwater_ratio)
- sewer_water_level_accel (1개): 수위 가속도 피처 (water_level_accel)

**피처 구성 (F_ROAD=31)**:
- road_base (13개): sensor_id, sensor_name, lat, lon, elevation, measurement_value, measurement_unit, data_type, time, source, etc.
- road_water_level (29개): 수위 관련 피처 (water_level_*, water_level_mean, water_level_std, water_level_max, water_level_min, water_level_median, water_level_skew, water_level_kurt, water_level_range, etc.)

**데이터 크기**:
| 파일 | 크기 | 노드 수 | 피처 수 |
|------|-------|---------|---------|
| sewer_all.parquet | 1,797 MB | 1,029 | 185 |
| road_all.parquet | 112 MB | 206 | 31 |

---

### B. preprocessing_rain.ipynb (강수량 데이터 수집 및 피처 추출)

**목적**: KMA 기상청 API를 통해 서울시 강우관측소 48개소의 시간별 강수량 데이터 수집 및 피처 추출

**주요 작업**:
1. **KMA API 연동**: 2022-01 ~ 2025-09 (44개월) 분월별 데이터 수집
   - API: `http://www.kma.go.kr/weather/observations/climate/summary-1.en`
   - 총 44개월 데이터 수집 완료 (실패 없음)
2. **데이터 정제**: 컬럼명 통일, 결측치 처리, dtype 최적화
3. **시간당 평균**: 시간별 강수량 데이터를 시간당 평균으로 변환
4. **피처 추출**: 기본 정량 통계 + 주파분석 + 에너지 + 엔트로피 + 자기상관
5. **정규화**: Min-Max 정규화 (train stats 사용, val/test 전이)
6. **공간 인덱싱**: rtree를 통한 강수관측소 공간 인덱싱
7. **피벗 테이블**: `(timestamp, station_id)` 피벗 구조 구축

**결과**:
- `rain_all.parquet`: 48개 관측소 × 30,168 타임스텝 = 1,448,064 행 (225 MB)
- `rain_train/val/test.parquet`: 청크당 100만 행, 총 3~4 파일씩 (총 990 MB)
- `rain_station_coords.parquet`: 48개 관측소 좌표 (4 KB)

**피처 구성 (F_RAIN=13)**:
- station_id (문자열): 강우관측소 ID (G002020 ~ G5147040)
- timestamp (datetime): 타임스탬프 (시간 단위)
- rainfall_norm (float32): Min-Max 정규화 된 강수량 [0, 1]
- rainfall_base (1개): base_rainfall
- rainfall_statistic (5개): rain_mean/std/min/max/skew
- rainfall_frequency (4개): freq_peak_0.01/freq_peak_0.05/freq_peak_0.1/freq_dom
- rainfall_energy (1개): energy_total
- rainfall_entropy (1개): entropy_normalized
- rainfall_autocorr (1개): autocorr_lag1

**데이터 크기**:
| 파일 | 크기 | rows |
|------|-------|------|
| rain_all.parquet | 225 MB | 1,448,064 |
| rain_train.parquet (총) | 505 MB | ~3,360,000 |
| rain_val.parquet (총) | 144 MB | ~3,360,000 |
| rain_test.parquet (총) | 341 MB | ~3,360,000 |

**강수량 기저통계**:
- 전체 평균 강수량: 0.547 mm/h
- 강우 발생 비율: 24.6%
- 최대 강수량: 172.9 mm/h
- 관측소 수: 48개

---

### C. preprocessing_rain_join.ipynb (강수량 피처 하수관로 데이터에 통합)

**목적**: 하수관로 피처에 rainfall_norm 추가 (F_SEWER: 9 → 10)

**주요 작업**:
1. **매핑 테이블**: `sewer_rain_mapping.parquet` (sensor_id → rain_station_id, 거리 기반)
   - 평균 매핑 거리: 1,933m
2. **조인 방식**: sensor_id 그룹별 벡터화 → fillna(0.0) (결측 버킷 = 무강우)
3. **gnn_config.json 업데이트**: sewer features에 rainfall_norm 추가, rainfall status "active"로 변경

**결과**:
- `overlap/{split}/sewer_{split}.parquet`: F_SEWER = 10 (rainfall_norm 추가됨)
- `gnn_config.json`: sewer features = 10개, rainfall status = "active"

**피처 구성 (F_SEWER=10)**:
1. sensor_id
2. lat
3. lon
4. elevation
5. water_level_mean
6. water_level_std
7. water_level_max
8. water_level_min
9. water_level_range
10. **rainfall_norm** ← 새로 추가

**데이터 크기**:
| 파일 | 크기 | 노드 수 | 피처 수 |
|------|-------|---------|---------|
| sewer_train.parquet | 129 MB | 686 | 10 |
| sewer_val.parquet | 19 MB | 98 | 10 |
| sewer_test.parquet | 57 MB | 245 | 10 |

---

## 2. 데이터셋 최종 구조

### 분할 정보
| 분할 | 기간 | 하수관로 노드 | 타임스텝 |
|------|------|--------------|---------|
| train | 2022-01 ~ 2024-06 | 686 | ~3,360,000 |
| val | 2024-07 ~ 2024-09 | 98 | ~3,360,000 |
| test | 2024-10 ~ 2025-09 | 245 | ~3,360,000 |

### GNN Config (gnn_config.json)
```json
{
  "node_features": {
    "sewer": ["sensor_id", "lat", "lon", "elevation",
               "water_level_mean", "water_level_std", "water_level_max",
               "water_level_min", "water_level_range", "rainfall_norm"],
    "road": ["sensor_id", "lat", "lon", "elevation",
             "water_level_mean", "water_level_std", "water_level_max",
             "water_level_min", "water_level_range"]
  },
  "future_extension": {
    "rainfall": {
      "status": "active",
      "note": "서울시 강우관측망 48개소 통합 완료 — 하수관로 노드별 최근접 관측소 매핑"
    }
  }
}
```

---

## 3. 주요 기술적 결정

1. **메모리 최적화**: float64→float32, int64→int32, object→category
2. **채크 기반 처리**: 대용량 데이터 청크 처리 (100만 행/청크)
3. **Min-Max 정규화**: train 통계로만 fit → val/test 전이
4. **Fillna 전략**: 결측치 = 0 (무강우 또는 관측 안됨)
5. **거리 기반 매핑**: sensor_id → 가장 근접한 rain_station_id (평균 1.9km)

---

## 4. 생성된 핵심 파일 목록

| 파일 | 설명 | 크기 |
|------|------|------|
| dataset/features/gnn_config.json | GNN 설정 | 2 KB |
| dataset/features/adjacency.parquet | 하수관로 그래프 인접행렬 | 1,847 MB |
| dataset/features/adjacency_expanded.parquet | 확장 인접행렬 | 125 MB |
| dataset/features/sewer_sewer_edges.parquet | 하수-하수 간선 | 6 MB |
| dataset/features/road_node_index.parquet | 도로 노드 인덱스 | 41 KB |
| dataset/features/sewer_node_index.parquet | 하수 노드 인덱스 | 30 KB |
| dataset/processed/sewer_rain_mapping.parquet | 하수-강우 매핑 | 12 KB |
| dataset/processed/rain_station_coords.parquet | 강우관측소 좌표 | 4 KB |
| dataset/processed/correlation_results.parquet | 상관분석 결과 | 80 MB |
| dataset/features/overlap/train/sewer_train.parquet | train 하수관로 | 129 MB |
| dataset/features/overlap/val/sewer_val.parquet | val 하수관로 | 19 MB |
| dataset/features/overlap/test/sewer_test.parquet | test 하수관로 | 57 MB |
| dataset/features/overlap/train/road_train.parquet | train 도로 | 4 MB |
| dataset/features/overlap/val/road_val.parquet | val 도로 | 1 MB |
| dataset/features/overlap/test/road_test.parquet | test 도로 | 2 MB |
| dataset/features/rain/train/rain_train.parquet | train 강수량 | 505 MB |
| dataset/features/rain/val/rain_val.parquet | val 강수량 | 144 MB |
| dataset/features/rain/test/rain_test.parquet | test 강수량 | 341 MB |
| dataset/features/rain/rain_all.parquet | 전체 강수량 | 225 MB |
| tensor_cache/sewer_train.pt | 하수관로 훈련 텐서 | 110 MB |
| tensor_cache/seaver_val.pt | 하수관로 검증 텐서 | 15 MB |
| tensor_cache/sewer_test.pt | 하수관로 테스트 텐서 | 45 MB |
| tensor_cache/road_train.pt | 도로 훈련 텐서 | 11 MB |
| tensor_cache/road_val.pt | 도로 검증 텐서 | 2 MB |
| tensor_cache/road_test.pt | 도로 테스트 텐서 | 5 MB |