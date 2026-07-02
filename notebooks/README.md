# 노트북 가이드

> ⚠️ **실행은 프로젝트 루트(`/home/namjun/city_flood`)를 작업폴더로** — 경로는 루트 기준 상대경로.
> 신규 노트북은 첫 셀에 `import os; os.chdir("/home/namjun/city_flood")` 부트스트랩 권장.

## 01_preprocessing — 원천 → 정제 산출물
| 노트북 | 내용 |
|---|---|
| `eda_raw` | 원천 데이터 최초 탐색 |
| `preprocessing` | 1차 전처리(레거시) |
| `preprocessing_eda_based` | ★현행 표준 정제 — `road_panel_10min`·`sewer_features_10min` 등 생성 |
| `preprocessing_rain` | AWS 강우 1분 → 10분 피처 |
| `preprocessing_rain_join` | 강우 ↔ 센서 매핑·조인 |

## 02_analysis — EDA / 라벨 품질
| 노트북 | 내용 |
|---|---|
| `correlation_road_sewer` | 도로↔하수 상관·선후(결합 약함 corr~0.05 확인) |
| `label_quality_audit` | 도로 침수 라벨 진위(93% 아티팩트) → `road_flood_sensor_trust` |
| `recurrent_flood_report` | 상습 침수지점 리포트 |
| `flood_event_catalog` | 하수 다중센서 동시반응 침수 사건 카탈로그(2022~2025, fill≥0.6+±90분 병합) → `flood_event_catalog` |

## 03_surcharge — 하수 만관(surcharge) 가설 검증
| 노트북 | 내용 |
|---|---|
| `sewer_surcharge_audit` | 만관 플래그 진위감사(강우 lift) → 확정 만관센서 |
| `sewer_capacity_recalibration` | capacity(관높이) 재교정 → `sewer_capacity_reliability`(datum 불일치 29 식별) |
| `sewer_aws_reverify` | AWS 1분강우 재검증 → 확정 만관 3→5 (16-0017·18-0018 추가) |
| `sewer_road_chain` | 시흥동 사슬(강우→만관→침수) 시공간 동조 |
| `zone_matching` | 제원표 배수구역 기반 하수↔도로 교차매칭 |

## 04_feasibility — GNN 학습셋 가용성
| 노트북 | 내용 |
|---|---|
| `gnn_feasibility` | 현재 데이터로 침수예측 학습셋 불가 판정 + 필요데이터 → `gnn_data_needs` |

## 05_modeling — 모델
| 노트북 | 내용 |
|---|---|
| `nowcast_baseline` | persistence vs 자기이력 nowcast 베이스라인 |
| `model_comparison` | 모델 비교(레거시) |

## 06_gnn — 하수 수위 예측 GNN (진행 중)
관악 GIS 관망 유향 그래프 기반 하수 수위 예측. Phase 0 베이스라인부터.

## _archive — 실험/폐기
`test` 등.

---
**스크립트**는 `scripts/`(데이터 다운로드·빌드·데모), 실행은 루트에서 `python scripts/xxx.py`.
