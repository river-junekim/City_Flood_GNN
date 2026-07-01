"""데이터 요청 사유·근거 보고서 — KICT 앞.

★ 기준: 관악구 · 여름 우기(6~9월). 서울 전역·전체기간 폐기(2026-06-24).
요청 데이터: ⓪관악 신설 하수센서 시계열 ①침수 정답 라벨 ②DEM ③레이더 강우.
실제 산출물(+GIS)에서 수치를 읽어 [요청/사유/근거/증거그림]을 출력.
실행:  python scripts/data_request_rationale.py
"""
from __future__ import annotations
import os
import pandas as pd

ROOT = "/home/namjun/city_flood"
EB = os.path.join(ROOT, "dataset/processed/eda_based")
WET = [6, 7, 8, 9]
os.chdir(ROOT)


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def head(n, t):
    print("\n" + "=" * 72 + f"\n  근거{n}. {t}\n" + "=" * 72)


def block(요청, 사유, 근거, 그림):
    print(f"  ▸ 요청: {요청}")
    print(f"  ▸ 사유: {사유}")
    print("  ▸ 근거(관악·여름 우기 실측):")
    for g in 근거:
        print(f"      - {g}")
    print(f"  ▸ 증거 시각화: {그림}")


def _coverage():
    """GIS 등록 센서 vs 우리 시계열 보유 (관악)."""
    import geopandas as gpd
    import warnings
    warnings.filterwarnings("ignore")
    base = f"{ROOT}/03_GIS/하수관로_노면_수위계_shp"
    s = gpd.read_file(f"{base}/도시침수계(관로)_서울시 2026.shp")
    r = gpd.read_file(f"{base}/도시침수계(노면)_서울시 2026.shp")
    s_gw = s[s["자치구"].astype(str).str.contains("관악")]
    r_gw = r[r["자치구"].astype(str).str.contains("관악")]
    have = set(pd.read_parquet(f"{EB}/../cleaned/sewer_node.parquet", columns=["sensor_id"]).sensor_id)
    n_have = int(s_gw["수위계번호"].astype(str).isin(have).sum())
    return dict(sew_reg=len(s_gw), sew_have=n_have, sew_new=len(s_gw) - n_have, road=len(r_gw))


# ───────────────────────── ④ 신설 하수센서 시계열 ─────────────────────────
def item_sensor():
    head(4, "관악 신설 하수 수위계 시계열 (낮은 비용·2026 여름 대비)")
    cov = _try(_coverage)
    근거 = []
    if cov:
        근거.append(f"GIS 등록 센서: 하수 {cov['sew_reg']} + 도로 {cov['road']} = {cov['sew_reg']+cov['road']}개")
        근거.append(f"우리 보유 시계열: 하수 {cov['sew_have']} + 도로 {cov['road']} = {cov['sew_have']+cov['road']}개")
        근거.append(f"→ 하수 {cov['sew_new']}개(21-0014~0042)는 2025-08 신설로 데이터 없음 (관망 위엔 이미 존재)")
        근거.append(f"확보 시 관악 하수 노드 {cov['sew_have']}→{cov['sew_reg']} (3배↑) = 맨홀-노드 GNN의 관측 노드 밀집")
        근거.append("우리 데이터는 2025-08-31까지 → 신설분엔 2025-09(조원로 16건 침수 활성월) 등 우리에게 없는 기간 포함")
    block(
        "관악 신설 하수 수위계 29개(21-0014~0042, 2025-08~) 및 이후 최신 기간 시계열",
        "즉시 침수 라벨 가치는 낮으나(신설·여름 놓침), 비용 0(이미 수집 중)·2026 여름 실시간 수확(전달 리드타임 고려 지금 요청)·GNN 그래프 밀도 확보.",
        근거,
        "reports/figures_demo/req_sensor_coverage.png",
    )


# ───────────────────────── ① 침수 정답 라벨 ─────────────────────────
def item1():
    head(1, "침수 정답 라벨")
    근거 = []
    rn = _try(lambda: pd.read_parquet(f"{EB}/../cleaned/road_node.parquet", columns=["sensor_id", "자치구"]))
    rp = _try(lambda: pd.read_parquet(f"{EB}/road_panel_10min.parquet", columns=["sensor_id", "ts10", "flood_t6"]))
    if rn is not None and rp is not None:
        gw = rn[rn.자치구 == "관악구청"].sensor_id.tolist()
        s = rp[rp.sensor_id.isin(gw) & rp.ts10.dt.month.isin(WET)]
        jw = s[s.sensor_id == "조원로 5-6"]
        npos = int((jw.flood_t6 == 1).sum()); nobs = len(jw)
        if nobs:
            근거.append(f"관악 도로 {len(gw)}센서 중 신뢰 침수 양성은 조원로 5-6 1지점·{npos}건뿐 (나머지 4센서 ~0)")
            근거.append(f"극심한 불균형 1 : {(nobs-npos)//max(npos,1):,} (조원로 정상 {nobs-npos:,} vs 침수 {npos})")
        근거.append("봉천동 911-14 가짜 1,511건은 '겨울' 아티팩트 → 침수철엔 거의 안 찍힘 (오염은 비수기 현상)")
    근거.append("하수 위험수위는 대리지표 — 도로 침수 시각 하수 fill 0.13 (실제 침수 ≠ 하수)")
    block(
        "KICT 침수 검증(ground-truth) 데이터 · 관악 침수흔적도 · 120/119 침수신고",
        "납품물=AI 예측 모델 → supervised 학습엔 검증된 정답지 필수. 관악은 깨끗한 양성이 1지점·소수라 학습 불가.",
        근거,
        "reports/figures_demo/req1_label_contamination.png · req_why_infeasible.png",
    )


# ───────────────────────── ② DEM ─────────────────────────
def item2():
    head(3, "DEM (수치표고) — 또는 KICT 하수관망 수리모델(SWMM)")
    근거 = [
        "노드 입력 = 맨홀 유입량 = 강우 × 집수면적 (합리식 Q=C·i·A) — 강우만으론 유입 부피가 안 나옴, 면적이 곱해져야 함",
        "같은 비라도 집수면적 큰 맨홀이 먼저 위험수위·월류 → 공간 위험 구분에 면적 가중 필수",
        "가설 A 하수 만관 기각(위험수위 도달 0.064%·만관 0%)·가설 B 하천역류 기각(침수 90%가 도림천 평상시) → 관악 침수=국지 표면류 = 큰 집수면적+저지대",
        "GIS 소유역(집수단위)은 40개로 coarse·이미 활용 → 맨홀별 정밀 집수면적은 DEM/SWMM 필요",
        "관망 골격은 공식 연결(시작→끝맨홀)로 구축·검증(차수 정확 80%·±1 91%·공간정합 77%); 완전 검증판은 SWMM",
    ]
    block(
        "고해상 DEM/LiDAR 또는 KICT 하수관망 수리모델(SWMM 등, 집수면적 포함) 중 하나",
        "모델 노드 입력(맨홀 유입)=강우×집수면적. 강우(②)만으론 부족하고 집수면적이 곱해져야 실제 유입량. DEM에서 도출하거나 SWMM이 있으면 그대로 사용.",
        근거,
        "reports/figures_demo/req2_dem_rationale.png · req_sewer_reach.png",
    )


# ───────────────────────── ③ 레이더 강우 ─────────────────────────
def item3():
    head(2, "레이더 강우")
    근거 = []
    gm = _try(lambda: pd.read_parquet(f"{EB}/gnn_model_csi.parquet"))
    if gm is not None:
        p = gm["persistence"].values
        gon = gm["GNN-on(유향전파)"].values if "GNN-on(유향전파)" in gm else gm.iloc[:, -1].values
        근거.append(f"관악 시공간 GNN: 강우+이력이 persistence를 CSI로 못 넘음 (10/30/60분 persistence {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f} vs GNN {gon[0]:.2f}/{gon[1]:.2f}/{gon[2]:.2f}) = 리드타임 부재")
    근거.append("국지(점) 강우는 수위와 거의 동시 반응 → 다가오는 호우를 못 봄 (조기경보 리드타임 0)")
    근거.append("맨홀-노드 GNN의 '맨홀별 강우 유입' 입력에 격자 강우 필수 (점 강우로는 공간 분포·이동 불가)")
    block(
        "기상청 레이더(HSR/HFC) 격자 강수 또는 KICT 보정 강수장",
        "조기경보엔 리드타임 필요한데 국지 강우는 동시 반응. 레이더(공간·이동)만이 수위를 앞서는 입력.",
        근거,
        "보고서 §19 그림 28(관악 GNN) · §17 그림 26(onset)",
    )


def main():
    print("\n" + "#" * 72)
    print("#  KICT 데이터 요청 — 사유·근거 (관악구 · 여름 우기 6~9월, 실측)  2026-06-24")
    print("#  ※ 시각화 근거는 별도(그래프), 본 스크립트는 텍스트 근거")
    print("#  ※ 우선순위: ①라벨 → ②레이더 → ③DEM → ④신설센서(낮은비용·2026여름 대비)")
    print("#" * 72)
    item1(); item3(); item2(); item_sensor()
    print("\n  시각화 자료(reports/figures_demo/): req_sensor_coverage · req1_label_contamination ·")
    print("    req_why_infeasible · req2_dem_rationale · req_sewer_reach")
    print("  요청서 본문: docs/DATA_REQUEST.md · 상세: reports/progress_report.html\n")


if __name__ == "__main__":
    main()
