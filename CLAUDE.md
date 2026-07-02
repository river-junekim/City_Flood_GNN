# 도심 침수 예측 — Claude 작업 가이드

> 작업 디렉토리: `/home/namjun/city_flood` · 마지막 업데이트 2026-07-02
> **상세 일자별 진행로그 → `docs/PROGRESS_LOG.md`** · 레거시 GNN 5모델 프로젝트 → `docs/LEGACY_GNN_PROJECT.md`
> 설계 문서 → `docs/TIMESERIES_PREPROCESSING_DESIGN.md` ★, `docs/DATA_QUALITY_FINDINGS.md` 등

## 0. 한 줄 요약
**★범위 확정(2026-06-24): 모든 기준=관악구. 서울 전역 폐기**(서울 결과는 맥락 참고만) [[gwanak-only-scope]].
**원 목표=관악구 배수지역 침수예측 GNN**(관악 데이터 부족→서울 확장했으나 폐기). 서울 **도로노면(123)·하수관로(484) 수위 센서** 사용. EDA에서 **침수 양성을 만드는 도로 센서의 ~93%가 아티팩트형**(비영값 고착·사각파·겨울지배; *마른 도로 0 지배는 정상*)·**관악 침수≠하수만관(표면류형)** 판명 → **현재 데이터로 침수예측 supervised 학습 불가**(깨끗한 양성 부족 + 희소 양성 오염). 단 **03_GIS 관악 관망 토폴로지 확보** → 라벨 불필요한 **하수 수위 예측 GNN**으로 진행 중.
**★★전환(2026-07-02): 라벨 병목 해소 개시.** 재조사 결과 **하수 다중센서가 실제 침수를 포착**하고 있었고("양성 0"의 원인은 센서결함이 아니라 관측창 한계=도로센서 2024-06설치·AWS강우 여름만), **서울시 침수흔적도(ground-truth) 확보→2022-08-08 하수 7센서가 실제 침수구역과 공간(≤102m)+시간(21~23시) 양면 일치 검증**. **침수흔적도 폴리곤→GNN 맨홀 노드 13,272 공간조인으로 노드 침수라벨 생성**. 상세=`docs/REPORT_2차_검증발견.md`·`reports/report_2nd_findings.html`. 남은 입력 병목=레이더 강우.

## 1. 작업 규율 (반드시 준수)
- **추측·단정 금지** — 임계·관계는 데이터로 검증해 정한다. **과장 금지** — 확정 vs 미확정 구분해 보고.
- 작업은 **`.ipynb` 노트북**(셀+주석+시각화)으로 정리. bash는 빠른 확인용.
- **시각화 라벨은 한글**(영어 X). 첫 코드셀에 `import sys; sys.path.insert(0,"scripts"); from krfont import set_korean; set_korean()` (NanumGothic·`axes.unicode_minus=False`). 라벨에 유니코드 −(U+2212) 말고 ASCII '-' 사용.
- **보고서 요청 시 `reports/progress_report.html` 수정·확장**(새 파일 X, 그림은 base64 임베드).
- **KMA 인증키는 `.env`의 `KMA_AUTH_KEY`** — gitignore됨, 절대 커밋·채팅·메모리 금지.
- **EDA raw 데이터·GIS 내용은 영구 메모리에 안 남김**(세션 한정 — 사용자 요청). 데이터 직접 접근은 허용됨.
- **강우는 절대 기준 아님** — 양성 확증·위험순위 도구(비강우 침수: 상수도·하수역류·하천역류 + 관측소 ~1.2km 거리누락). 강우 동반=강한 양성, 무강우=약한 음성.

## 2. 확립된 핵심 결론
- **침수 양성을 만드는 도로 센서의 ~93%가 아티팩트형**(비영값 고착·사각파·겨울지배). ⚠️*마른 도로가 0인 것은 정상(고장 아님)* — 문제는 **희소 양성의 오염**. 진짜 침수는 정상적으로 극희소(<0.5%). 증거: 봉천동 911-14(아티팩트) 가짜 1511건 vs 조원로 5-6(진짜) 30건 — 양성 수가 실제와 반비례.
- **도로↔하수 결합 약함**(차분 corr ~0.05)·하수 약간 후행 — 해상도(1/10분)·페어링(거리/동일위치) 무관하게 안정. 튄 강결합은 검증하면 대부분 허상. → **센서간 예측 엣지 의존 말 것.**
- **단독 nowcast: persistence가 강한 바.** 자기이력 모델 수위회귀 못 이김, 분류 점수 ~20%가 stuck 거품.
- **신뢰 도로침수 = 강우확인 6지점**: 시흥동882-61·성산로494-30·신영동165·대림동862-5·둔촌동218-6·월계동9-2.
- **하수 surcharge(만관, fill_rate≥1)**: 강우→fill 실재(corr **0.21**). surcharge 플래그 194센서 중 **진짜 강우성 만관 확정 5**(20-0012·16-0017·18-0018·11-0004·06-0009; AWS 1분강우 재검증으로 +2). 나머지=stuck천장77·미검증79·비강우모호29. **capacity "에러" 35는 센서불량 아닌 기준불일치 29**(수위>관높이; 20~25구역, 제원표 자체 기왕최대도 관높이 초과).
- **사슬 검증**: 확정만관 5 중 도로침수지점과 co-located는 **시흥동(18-0018↔882-61, 575m) 1쌍뿐**. 거기선 강우→만관→침수 또렷(도로침수 95%가 직전6h 만관동반). 배수구역 매칭해도 새 쌍 없음(공식 배수구역 15 basin이 coarse).
- **관악 침수≠하수만관**: 관악 하수 13센서 **만관 0건**(호우때 최대 fill 0.69), 도로침수 시각 하수 fill 0.13 → **표면류형 추정**. 마포·반포·불광·홍제도 도로침수 있으나 만관 0(동일).
- **GNN 가용성 판정**: 진짜신호 희소·구마다 1개씩 흩어짐(공존=금천1) → supervised 침수예측 학습셋 불가. 병목=**독립 침수정답·다년사건(라벨)**.
- **03_GIS 관악 관망 확보**: 하수관로 17,286(유향 엣지 92.9%)·맨홀 12,184·**소배수구역 40**·물받이 27,447·토구·**레이더격자 1km**. 필요데이터 ③④⑤ 충족. → 토폴로지 병목 해소(관악 한정).
- **관악 GNN 실측(06_gnn 01~06)**: Phase0~2(베이스라인·CSI·유향그래프·상류근사)에 더해 **Phase3 실제 시공간 GNN(GRU+유향GraphSAGE+ablation, 위험수위 fill≥0.5)**: persistence 0.72/0.57/0.51 ≫ GNN-off 0.39/0.29/0.25 > **GNN-on 0.17/0.15/0.08**(그래프가 오히려 더 나쁨=전파신호 부재). → 관악 데이터로 GNN 침수예측 불가(데이터 한계). 파이프라인은 완성(레이더 꽂으면 재측정).
- **위험수위 재정의(만관→fill≥0.8)**: 서울 **도달 329센서(만관 194의 1.7배)**·100%강우동반·90%여름 → 만관 기준이 신호 가렸음. 강우성 확정 6(신규 02-0001·10-0008·15-0006·01-0005). 단 깔때기 병목=**AWS강우 커버리지**(검증가능 37)+stuck123.
- **A-1 서울전역 위험수위 예측(06_gnn 05)**: 실제 도달센서서도 **강우+이력이 persistence를 CSI로 못 넘음**(0.67/0.52/0.45 vs 0.64/0.50/0.40). 이유=강우-수위 거의 동시반응, **리드타임 부재**.
- **onset 심화(06_gnn 07)**: 진입(현재 비위험→위험) 조기예측. persistence 구조적 0인데 모델은 **0.06~0.09**(약신호 존재하나 포착10%·오경보80% 실무미달). → **의미있는 조기경보엔 공간·이동 강우(레이더) 필수**. 이 값이 레이더가 넘어야 할 기준선.

## 3. 현재 방향 / 다음 작업
**★GNN 구조 확정(2026-06-24 지시): 노드=맨홀(관악 12,184), 엣지=상류→하류 유향 관거(관저고 기반)** = 맨홀-노드 하수 라우팅 GNN. 17-센서 그래프는 stopgap. 관측은 수위계 snap 맨홀만(13→신설29 확보시 42, semi-supervised). [[gnn-feasibility-verdict]]

### 2026-07-02 완료 (라벨 확보 단계)
- **하수 드롭아웃 전처리 규칙화**: 고수위→바닥값(0.01m)→복귀=센서 드롭아웃(4,356건·10,977bins·0.021%·100/485센서) 감지·선형보간·`is_dropout` 플래그(원자료 1분 보존). 노트북 `01_preprocessing/sewer_dropout_clean`·튜닝 `sewer_dropout_tuning`(임계 FLOOR0.02·HIGH0.3·MAXLEN12).
- **침수흔적도(ground-truth) 확보**: OA-15636 **2010~2025 전 연도** zip(`03_GIS/침수흔적도_shp/`). 관악 폴리곤=2010(1227)·2011(1686)·2022(3572)·2023(5)·2024(1)·2025(0=관악 미해당). 2010·2011은 센서 이전이나 저지대 노드라벨로 가치. 2025는 스키마 상이(GU_NAM 없이 ADM_CD).
- **2022-08-08 검증**: 하수 7센서(21-0001~7)가 실제 침수구역과 **공간(≤102m·중앙0m) + 시간(흔적도 저녁20~23시 72%↔센서피크 20:50~23:00) 양면 일치**. 원인 "배수용량초과"=하수만관 1002폴리곤. 노트북 `02_analysis/flood_trace_crosscheck`·`floodtrace_timing_align`. 카탈로그 E01 "흔적도 검증(공간+시각)" 승격.
- **GNN 노드 침수라벨**: 흔적도 폴리곤×맨홀노드 13,272 공간조인 → `gnn_manhole_flood_labels_2022.parquet`(2022-08-08 내부1229·≤50m 6709·침수심·원인, 센서42노드 전부라벨). 노트북 `02_analysis/floodtrace_manhole_join`. → 라벨병목 해소, 하수 라우팅 GNN에 실제 노드정답 확보.

### 다음
1. **레이더 강우**(입력 병목·최우선) → 격자↔노드 매핑 → onset/위험수위/노드분류 재측정. 출처=기상청 포털(data.kma.go.kr)/API허브(apihub.kma.go.kr), 03_GIS 1km 격자 shp. onset CSI 0.06~0.09가 넘어야 할 기준선. [[todo-radar-rain-acquisition]]
2. **과거 강우(2022·2023)** — AWS 2022 여름 받아 E01 강우 직접확인(현재 외부기록 141.5mm로만 뒷받침). KMA 일일용량 제한으로 분할·재개(2026-07-02 2023-02-23까지 받고 403 중단, 2022는 미착수). `python scripts/download_aws_seoul.py 202208010000 202209010000`.
3. **전 기간 노드 라벨링** — 2010·2011 관악 흔적도로 저지대 정적라벨 추가(다운로드 무관·즉시 가능).
4. **1차/2차 보고서 병합 결정**(미정) · **하천 수위**(한강홍수통제소, 도림천) 하천역류 재검증 · **DEM** 집수면적/표면류.
- 보류: 옛 figure 6개(03_surcharge·04) 한글화.

## 4. 데이터·파일 핵심
- **정제 산출물** `dataset/processed/eda_based/`: `road_cleaned`(1분), `road_panel_10min`(5.69M, 라벨/split/차분), `sewer_features_10min`(51.6M, **capacity교정·fill_rate·surcharge·is_dropout**), `*_sensor_quality`, `road_flood_sensor_trust`(판정_final), `recurrent_flood_report`, `sewer_capacity_v2`.
- **드롭아웃 정정(2026-07-02)**: 하수 센서 일부가 고수위 중 순간 바닥값(0.01m) 급락→복귀(센서 드롭아웃, 4,356건·10,977bins·0.021%·100/485센서). 고수위 사이 낀 바닥값 런만 선형보간+`is_dropout` 플래그(원자료 1분 보존). 노트북 `01_preprocessing/sewer_dropout_clean.ipynb`. 예: 21-0005 2022-08-08 21:00~21:30.
- **추가 산출물**: `sewer_capacity_reliability`(신뢰455/datum불일치29), `sewer_surcharge_audit`(최종판정·lift_aws), `surcharge_road_pairing`·`road_surcharge_zone_pairing`(매칭), `gnn_data_needs`.
- **노트북**(`notebooks/`, 폴더정리·README有, ★루트에서 실행): `01_preprocessing`·`02_analysis`·`03_surcharge`(만관 검증 5종)·`04_feasibility`·`05_modeling`·`06_gnn`(하수 수위 GNN 진행). 스크립트는 `scripts/`(다운로드·빌드·`demo_client.py`), 실행 `python scripts/xxx.py`.
- **제원표(원천)**: `dataset/processed/서울시 수위계(하수관로/도로) 제원표_20260310.xlsx` — 관규격·배수구역이 정제데이터와 100% 일치(우리 capacity·좌표 원천).
- **GIS**: `03_GIS/관악구_하수관로_맨홀_shp/`(sb001관로·sb101맨홀·sb503물받이·sb104토구, EPSG:5181, 속성 cp949) · `03_GIS/레이더격자_shp/`(1km) · `derived/`. ⚠️GIS 내용 영구메모리 금지.
- **침수흔적도(ground-truth, 2026-07-02 확보)**: `03_GIS/침수흔적도_shp/{연도}_서울시_침수흔적도.zip` — **2010~2025 전 연도**(2015·2021 침수없음 제외). OA-15636, EPSG:5179, 속성=구(GU_NAM)·침수심(F_SHIM)·발생일자/시각(F_SAT_YMD/TM)·원인(F_RSN_DTL)·주소·면적. ⚠️2025는 스키마 상이(GU_NAM 없이 ADM_CD·PNU). 관악 폴리곤=2010(1227)·2011(1686)·2022(3572)·2023(5)·2024(1)·2025(0). 관악필터본 `03_GIS/derived/침수흔적도_2022_관악.gpkg`. 재다운로드 seq: 2010~2020=4~13(2015·2021없음), 2022=30·2023=31·2024=32·2025=103, POST `infId=OA-15636&seq=NN&infSeq=1` → `datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do`.
- **GNN 노드 침수라벨(2026-07-02)**: `dataset/processed/eda_based/gnn_manhole_flood_labels_2022.parquet`(13,272노드, flood_in/25m/50m·depth·cause, 2022-08-08+2022전체) · `floodtrace_sensor_crosscheck_2022.csv` · 사건 카탈로그 `flood_event_catalog.parquet`(8사건, E01 흔적도검증). 노트북 `02_analysis/flood_trace_crosscheck·floodtrace_manhole_join·floodtrace_timing_align·flood_event_catalog`.
- **AWS 강우**: `download_aws_seoul.py` → `data/aws_seoul/win/` → `build_aws_rain.py` → `data/aws_seoul_rain_10min.parquet`(빌드본=2024-06~2025-09, 43지점). 매핑 `aws_sewer_mapping_v2`. ※win/에 2023-01~02-23 추가수집분 있음(403 중단, build 재실행 전). 2022는 미수집. 커버리지 밖 기간은 전량 fetch(스크립트 버그픽스 반영).
- **보고서**: `reports/progress_report.html`(§1~11 자체완결) · **`reports/report_2nd_findings.html`(2차 검증발견, 2026-07-02)** · `reports/data_request_report.html`(KICT 요청) · `reports/feedback_report.html`(1차 피드백) · `reports/figures_sewer/`·`reports/figures_demo/`.
- ⚠️ **gitignore 변경(2026-07-02, private repo)**: 이제 `scripts/*.py`·`data/`·`*.png`·`CLAUDE.md`·`03_GIS/`가 **추적됨**(커밋됨). 여전히 제외=**`.env`(비밀키)·`dataset/`(12GB·100MB초과)·100MB초과 dbf 2개**. AWS win 파일은 다운로드 진행 중이라 수동 선별 커밋.

## 5. 자주 쓰는 명령
```bash
set -a; source .env; set +a                                    # KMA 키 로드
python scripts/download_aws_seoul.py 202406010000 202410010000  # 2024 여름(재개)
python scripts/download_aws_seoul.py 202506010000 202510010000  # 2025 여름
python scripts/build_aws_rain.py                                # 1분창 → 10분 강우피처
python scripts/demo_client.py                                   # 고객설명 라이브 데모
```
