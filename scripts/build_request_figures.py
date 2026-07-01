"""KICT 요청 보고서용 그림 정본 생성 — 관악·여름 우기(6~9월).

침수 정의 = **서울시 침수 예보 기준: 도로수위계 침수심 ≥ 15cm**
 (출처: 서울시, news.seoul.go.kr/env/archives/522983 — 자치구 단위 침수예보 발령 기준의 수위 항)
 ※ 기존 임의 ≥6cm 폐기. 강우/15분강우 항은 OR 조건이나 본 모듈은 수위 항을 침수 양성으로 사용.

생성물(reports/figures_demo/):
  req_why_infeasible.png   가용성 진단(임계별 양성·불균형·하수 도달율)
  req1_label_contamination.png  도로센서별 양성 + 오작동 근거
  req_river_useless.png    하천역류 기각(15cm 침수시각 기준)
  req_leadtime_jowonro.png 조원로 강우↔수위 동시반응(리드타임 0)

실행:  python scripts/build_request_figures.py
"""
from __future__ import annotations
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys; sys.path.insert(0, "scripts")
from krfont import set_korean; set_korean()
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, matplotlib.pyplot as plt

EB = "dataset/processed/eda_based/"
FD = "reports/figures_demo"
os.makedirs(FD, exist_ok=True)
WET = [6, 7, 8, 9]
FLOOD_CM = 15  # 서울시 침수예보 기준 (도로수위계 침수심)
PERIOD = "2024-06~2025-09 · 여름 우기(6~9월)"

# ── 공통 데이터 ───────────────────────────────────────────────
rn = pd.read_parquet("dataset/processed/cleaned/road_node.parquet", columns=["sensor_id", "자치구"])
GW = rn[rn.자치구.astype(str).str.contains("관악")].sensor_id.tolist()
rp = pd.read_parquet(EB + "road_panel_10min.parquet", columns=["sensor_id", "ts10", "road_max", "month"])
gw = rp[rp.sensor_id.isin(GW)].copy()
summer = gw[gw.month.isin(WET)]
JO = "조원로 5-6"

# 관악 하수 fill 도달율(여름) — sewer_features_10min에서 산출(threshold-독립, 검증치)
SEWER_REACH = {0.5: 0.457, 0.6: 0.381, 0.7: 0.147, 0.8: 0.064, 0.9: 0.0, 1.0: 0.0}


def fig_why_infeasible():
    cnts = {t: int((summer.road_max >= t).sum()) for t in [6, 10, 15, 30]}
    jo = summer[summer.sensor_id == JO]
    n_pos = int((jo.road_max >= FLOOD_CM).sum()); n_neg = len(jo) - n_pos
    imbal = n_neg // max(n_pos, 1)

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.2))
    # A: 임계별 관악 도로 양성
    ts = [6, 10, 15, 30]; vals = [cnts[t] for t in ts]
    bars = ax[0].bar([f"≥{t}cm" for t in ts], vals,
                     color=["#bbb", "#bbb", "#c44e52", "#8c3b3e"])
    for b, v in zip(bars, vals):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 0.4, str(v), ha="center", fontsize=11, fontweight="bold")
    ax[0].set_ylim(0, max(vals) * 1.18)
    ax[0].set_title("(A) 임계별 관악 도로 침수 양성 수")
    ax[0].set_ylabel("양성 10분구간 수")
    ax[0].annotate("서울시 침수예보 기준", xy=(2, cnts[15]), xytext=(0.7, max(vals) * 0.7),
                   fontsize=9, color="#c44e52", arrowprops=dict(arrowstyle="->", color="#c44e52"))
    ax[0].text(0.5, -0.28, "≥15cm = ≥30cm → 남는 양성은 전부 깊은 침수", transform=ax[0].transAxes,
               ha="center", fontsize=8, color="#555")
    # B: 불균형 (조원로)
    bars = ax[1].bar(["정상", "침수"], [n_neg, n_pos], color=["#4c72b0", "#c44e52"])
    ax[1].set_yscale("log"); ax[1].set_title(f"(B) 조원로 불균형  1 : {imbal:,}")
    for b, v in zip(bars, [n_neg, n_pos]):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
    ax[1].set_ylabel("10분구간 수(로그)"); ax[1].set_ylim(top=n_neg * 4)
    # C: 하수 위험수위 도달율
    ks = list(SEWER_REACH); vs = [SEWER_REACH[k] for k in ks]
    bars = ax[2].bar([f"{k:.1f}" for k in ks], vs, color=["#6fa8dc"] * 4 + ["#cccccc"] * 2)
    ax[2].axvline(3, ls="--", c="red", lw=1)  # fill 0.8 위치
    for b, v in zip(bars, vs):
        ax[2].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}%", ha="center", fontsize=8)
    ax[2].set_ylim(0, max(vs) * 1.22); ax[2].set_title("(C) 관악 하수 충전율 도달율"); ax[2].set_xlabel("충전율(수위/관높이)")
    ax[2].set_ylabel("도달율(%)")
    ax[2].text(0.5, -0.28, "위험수위(0.8) 0.064% · 만관(1.0) 0% = 하수 신호 부재", transform=ax[2].transAxes,
               ha="center", fontsize=8, color="#555")

    fig.suptitle(f"관악 도로 침수 학습 가용성  (침수 = 도로수위계 ≥{FLOOD_CM}cm · 서울시 침수예보 기준)\n[{PERIOD}]",
                 fontsize=11, fontweight="bold", y=1.07)
    plt.tight_layout(); out = f"{FD}/req_why_infeasible.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out, "| 양성", cnts, "| 불균형 1:", imbal)


def fig_label_contamination():
    # 센서별 ≥6 vs ≥15 양성 + 오작동 근거(trust)
    try:
        tr = pd.read_parquet(EB + "road_flood_sensor_trust.parquet").set_index("sensor_id")
    except Exception:
        tr = pd.DataFrame()
    rows = []
    for sid in GW:
        s = summer[summer.sensor_id == sid]
        rows.append((sid, int((s.road_max >= 6).sum()), int((s.road_max >= FLOOD_CM).sum())))
    d = pd.DataFrame(rows, columns=["sid", "ge6", "ge15"]).sort_values("ge6", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d.ge6, height=0.4, color="#d0b0b0", label="≥6cm(참고)")
    ax.barh(y + 0.2, d.ge15, height=0.4, color="#c44e52", label="≥15cm(서울시 기준)")
    ax.set_yticks(y); ax.set_yticklabels(d.sid)
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.text(r.ge6 + 0.3, yi - 0.2, str(r.ge6), va="center", fontsize=8)
        ax.text(r.ge15 + 0.3, yi + 0.2, str(r.ge15), va="center", fontsize=8, fontweight="bold")
    ax.set_xlabel("\"침수\" 양성 10분구간 수"); ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"관악 도로센서별 침수 양성  (≥15cm 서울시 기준 적용 시 조원로만 잔존)\n[{PERIOD}]",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(); out = f"{FD}/req1_label_contamination.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out)


def fig_river():
    riv = pd.read_parquet("data/river_level_10min.parquet")
    sl = riv[riv.obsnm == "신림5교"].set_index("ts").wl.sort_index()
    p95, p99, med = sl.quantile(.95), sl.quantile(.99), sl.median()
    fl = summer[(summer.sensor_id == JO) & (summer.road_max >= FLOOD_CM)].ts10
    fl = fl[(fl >= riv.ts.min()) & (fl <= riv.ts.max())].sort_values()
    wl_at = sl.reindex(fl, method="nearest")
    share_norm = (wl_at <= p95).mean() if len(wl_at) else np.nan

    fig, ax = plt.subplots(figsize=(11, 4.4))
    bins = np.linspace(sl.min(), sl.max(), 50)
    ax.hist(sl.values, bins=bins, density=True, color="lightsteelblue", label=f"전체 시간 (n={len(sl):,})")
    ax.axvline(med, ls=":", c="gray", lw=1.2, label=f"평상시 중앙값 {med:.2f}m")
    ax.axvline(p95, ls="--", c="orange", lw=1.2, label=f"p95 {p95:.2f}m")
    ax.axvline(p99, ls="--", c="red", lw=1.2, label=f"p99 {p99:.2f}m")
    for w in wl_at.values:
        ax.plot([w, w], [0, 9], c="crimson", lw=1.6, alpha=0.8)
    ax.plot([], [], c="crimson", lw=1.6, label=f"조원로 침수(≥{FLOOD_CM}cm) 시각 n={len(fl)}")
    ax.set_xlabel("도림천 수위(신림5교, m)"); ax.set_ylabel("밀도"); ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 10)
    ax.set_title(f"하천(도림천) 수위는 침수와 무관 — 침수 {len(fl)}회 중 {share_norm:.0%}가 평상시(p95 미만)\n[{PERIOD}]",
                 fontsize=11, fontweight="bold")
    ax.text(0.0, -0.22, f"침수 조건: 도로수위계 ≥{FLOOD_CM}cm(서울시 침수예보 기준) · 하천 출처: 한강홍수통제소(HRFCO) 도림천 신림5교, {PERIOD}",
            transform=ax.transAxes, fontsize=8, color="#555")
    plt.tight_layout(); out = f"{FD}/req_river_useless.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out, f"| 침수 {len(fl)}회 평상시 {share_norm:.0%} | 침수시 도림천 중앙 {wl_at.median():.2f} vs 전체 {med:.2f}")


def fig_leadtime():
    rain = pd.read_parquet("data/aws_seoul_rain_10min.parquet")
    r = rain[rain.stn == 410].set_index("ts10").sort_index()
    jo = rp[rp.sensor_id == JO].set_index("ts10").road_max.sort_index()
    # 2025-09-09 호우 (조원로 ≥15cm 최다·강우 커버)
    a, b = pd.Timestamp("2025-09-09 00:00"), pd.Timestamp("2025-09-09 12:00")
    rseg = r.loc[a:b]; jseg = jo.loc[a:b]
    col = "rn15m" if "rn15m" in r.columns else r.columns[1]

    fig, ax1 = plt.subplots(figsize=(11, 4.4))
    ax1.bar(rseg.index, rseg[col].values, width=0.006, color="#4c72b0", alpha=0.6, label="강우(AWS 기상청, 15분강우 mm)")
    ax1.set_ylabel("강우(mm/15분)", color="#4c72b0"); ax1.tick_params(axis="y", labelcolor="#4c72b0")
    ax2 = ax1.twinx()
    ax2.plot(jseg.index, jseg.values, c="crimson", lw=1.8, marker="o", ms=3, label="조원로 도로수위(cm)")
    ax2.axhline(FLOOD_CM, ls="--", c="crimson", lw=1, alpha=0.6)
    ax2.set_ylabel("조원로 도로수위(cm)", color="crimson"); ax2.tick_params(axis="y", labelcolor="crimson")
    ax1.set_title("조원로: 강우와 도로수위가 거의 동시 상승 → 리드타임 ≈ 0 (2025-09-09 호우)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("시각")
    import matplotlib.dates as mdates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper left")
    plt.tight_layout(); out = f"{FD}/req_leadtime_jowonro.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out)


def fig_sewer_reach():
    sn = pd.read_parquet("dataset/processed/cleaned/sewer_node.parquet", columns=["sensor_id", "자치구"])
    gw = set(sn[sn.자치구.astype(str).str.contains("관악")].sensor_id)
    sf = pd.read_parquet(EB + "sewer_features_10min.parquet", columns=["sewer_sensor_id", "fill_rate", "ts10"])
    sf = sf[sf.sewer_sensor_id.isin(gw) & sf.ts10.dt.month.isin(WET)].dropna(subset=["fill_rate"])
    fmax = sf.groupby("sewer_sensor_id").fill_rate.max().sort_values(ascending=False)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.2]})
    ks = list(SEWER_REACH); vs = [SEWER_REACH[k] for k in ks]
    bars = axA.bar([f"{k:.1f}" for k in ks], vs, color=["#6fa8dc"] * 4 + ["#cccccc"] * 2)
    axA.axvline(3, ls="--", c="red", lw=1)
    for b, v in zip(bars, vs):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}%", ha="center", fontsize=8)
    axA.set_title("(A) 관악 하수 충전율 도달율"); axA.set_xlabel("충전율(수위/관높이)"); axA.set_ylabel("도달율(%)")
    y = np.arange(len(fmax))
    axB.barh(y, fmax.values, color=["#c44e52" if v >= 0.8 else "#4c72b0" for v in fmax.values])
    axB.axvline(0.8, ls="--", c="red", lw=1, label="위험수위 0.8")
    axB.set_yticks(y); axB.set_yticklabels(list(fmax.index), fontsize=7)
    for yi, v in zip(y, fmax.values):
        axB.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=7)
    axB.set_xlabel("센서별 최고 충전율"); axB.set_title("(B) 센서별 최고 도달"); axB.legend(fontsize=8); axB.set_xlim(0, 1.05)
    fig.suptitle(f"관악 하수관로 충전율 도달 분포  (위험수위 0.8 도달 극소·만관 1.0 0개)\n[{PERIOD}]",
                 fontsize=11, fontweight="bold", y=1.07)
    plt.tight_layout(); out = f"{FD}/req_sewer_reach.png"; plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out)


def fig_dem_rationale():
    riv = pd.read_parquet("data/river_level_10min.parquet")
    sl = riv[riv.obsnm == "신림5교"].set_index("ts").wl.sort_index()
    p95 = sl.quantile(.95)
    jo = summer[(summer.sensor_id == JO) & (summer.road_max >= FLOOD_CM)][["ts10", "road_max"]].sort_values("ts10")
    sf = pd.read_parquet(EB + "sewer_features_10min.parquet", columns=["sewer_sensor_id", "ts10", "fill_rate"])
    s2 = sf[sf.sewer_sensor_id == "21-0002"].set_index("ts10").fill_rate.sort_index()
    times = jo.ts10.tolist()
    road = jo.road_max.values
    sewer = [float(s2.reindex([t], method="nearest").iloc[0]) if len(s2) else np.nan for t in times]
    river = [float(sl.reindex([t], method="nearest").iloc[0]) for t in times]
    n_norm = sum(1 for r in river if r <= p95)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(len(times))
    ax.bar(x, road, width=0.5, color="crimson", label="조원로 도로수위(cm)")
    ax.axhline(FLOOD_CM, ls="--", c="crimson", lw=0.8, alpha=0.6)
    ax.set_ylabel("조원로 도로수위(cm)", color="crimson"); ax.tick_params(axis="y", labelcolor="crimson")
    ax.set_xticks(x); ax.set_xticklabels([t.strftime("%y-%m-%d\n%H:%M") for t in times], fontsize=7)
    ax2 = ax.twinx()
    ax2.plot(x, sewer, "o-", c="#4c72b0", label="인근 하수 충전율(조원547)")
    ax2.axhline(0.8, ls=":", c="#4c72b0", lw=0.8, alpha=0.6)
    ax2.set_ylabel("인근 하수 충전율 (0~1)", color="#4c72b0"); ax2.tick_params(axis="y", labelcolor="#4c72b0"); ax2.set_ylim(0, 1)
    ax.set_title(f"조원로 침수 시각: 도로수위 높으나 인근 하수·하천은 정상 → 표면류\n[{PERIOD}]", fontsize=11, fontweight="bold")
    ax.text(0.0, -0.32, f"침수 조건: 도로수위계 ≥{FLOOD_CM}cm(서울시) · 하천(도림천 신림5교, 한강홍수통제소): 침수 {len(times)}회 중 {n_norm}회가 평상시(p95 미만) · 인근 하수 충전율 위험수위(0.8) 미달",
            transform=ax.transAxes, fontsize=8, color="#555")
    l1, lab1 = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="upper right")
    plt.tight_layout(); out = f"{FD}/req2_dem_rationale.png"; plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", out)


if __name__ == "__main__":
    fig_why_infeasible()
    fig_label_contamination()
    fig_river()
    fig_leadtime()
    fig_sewer_reach()
    fig_dem_rationale()
    print("\n완료 — reports/figures_demo/ 6개 갱신")
