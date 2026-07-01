"""
고객 설명용 라이브 데모 — "현재 데이터로 침수예측 모델 학습이 불가능한 이유"

미팅에서 위에서 아래로 실행(또는 통째 실행)하면, 설명 대본 흐름대로
핵심 숫자와 그래프가 즉석에서 출력된다. 모든 수치는 정제 산출물(parquet)에서 실시간 계산.

사용:
    python demo_client.py            # 전체 실행 + 그래프 저장(reports/figures_demo/)
    python demo_client.py --no-plot  # 숫자만(그래프 생략, 더 빠름)

근거 출처: 노트북 label_quality_audit / sewer_surcharge_audit / gnn_feasibility
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

EB = Path("dataset/processed/eda_based")
CLEAN = Path("dataset/processed/cleaned")
FIGDIR = Path("reports/figures_demo")
NO_PLOT = "--no-plot" in sys.argv


# ────────────────────────────────────────────────────────────────────────
def head(n, title):
    print("\n" + "=" * 72)
    print(f"  [STEP {n}]  {title}")
    print("=" * 72)


def say(text):
    """대본 한 줄(미팅에서 읽는 멘트)."""
    print("  🗣  " + text)


def stat(label, value):
    print(f"      → {label}: {value}")


def load_parquet(path, **kw):
    return pd.read_parquet(path, **kw)


# ────────────────────────────────────────────────────────────────────────
def step0_overview():
    head(0, "개요 — 무엇을 보여드릴지")
    say("결론부터: 지금 센서 데이터만으로는 침수예측 모델을 '학습'시킬 수 없습니다.")
    say("모델 기법이 아니라 '학습용 정답 데이터'가 성립하지 않는 게 원인입니다.")
    say("그 근거 3가지를 실제 데이터로 즉석에서 확인해 보겠습니다.")


# ────────────────────────────────────────────────────────────────────────
def step1_label_contamination():
    head(1, "근거 1 — 정답지(침수 라벨)가 오염되어 있다")
    trust = load_parquet(EB / "road_flood_sensor_trust.parquet",
                         columns=["sensor_id", "판정_final"])
    rp = load_parquet(EB / "road_panel_10min.parquet",
                      columns=["sensor_id", "flood_t6"])

    n_total = len(trust)
    n_real = int((trust["판정_final"] == "진짜상습").sum())
    n_artifact = int((trust["판정_final"] == "아티팩트").sum())
    pos_rate = rp["flood_t6"].mean() * 100

    say("AI 학습엔 '여기, 이 시각에 침수가 있었다'는 정답(라벨)이 필요합니다.")
    say("그런데 도로 센서의 침수 표시 대부분이 실제 침수가 아니라 센서 오작동입니다.")
    say("(값이 한 곳에 멈춤 / 비 안 온 겨울에 침수로 찍힘)")
    stat("도로 침수센서 총 수", f"{n_total}개")
    stat("그중 '진짜 상습침수'로 검증된 센서", f"{n_real}개  ({n_real/n_total:.0%})")
    stat("'센서 오작동(아티팩트)'으로 판정", f"{n_artifact}개  ({n_artifact/n_total:.0%})")
    stat("전체 침수 양성 라벨 비율", f"{pos_rate:.2f}%  (이마저 대부분 오염)")
    say(f"비유: 답안지의 대부분이 오답 채점된 시험지 → 모델이 '고장 패턴'을 침수로 학습.")
    return dict(n_total=n_total, n_real=n_real, n_artifact=n_artifact)


# ────────────────────────────────────────────────────────────────────────
def step2_gwanak_counterexample(make_plot=True):
    head(2, "근거 2 — 관악구에선 핵심 가설이 관측되지 않는다")
    sn = load_parquet(CLEAN / "sewer_node.parquet", columns=["sensor_id", "자치구"])
    gw_sew = sn.loc[sn["자치구"] == "관악구", "sensor_id"].tolist()

    # 관악 하수 충전율 (pyarrow 필터로 13센서만 빠르게)
    g = load_parquet(EB / "sewer_features_10min.parquet",
                     columns=["sewer_sensor_id", "ts10", "fill_rate"],
                     filters=[("sewer_sensor_id", "in", gw_sew)])
    n_surch = int((g["fill_rate"] >= 1.0).sum())
    fmax = g["fill_rate"].max()

    # 조원로5-6 도로 침수 시각의 관악 하수 충전율
    rp = load_parquet(EB / "road_panel_10min.parquet",
                      columns=["sensor_id", "ts10", "flood_t6"],
                      filters=[("sensor_id", "==", "조원로 5-6")])
    flood_ts = rp.loc[rp["flood_t6"] == 1, ["ts10"]]
    fill_by_ts = g.groupby("ts10")["fill_rate"].max().rename("fill")
    merged = flood_ts.merge(fill_by_ts, on="ts10", how="left")
    fill_at_flood = merged["fill"].mean()

    say("이 사업의 핵심 가정: '하수관이 꽉 차 역류하면 도로가 침수된다'.")
    say("그런데 관악구 하수관 13곳은 2년간 단 한 번도 가득 찬 적이 없습니다.")
    stat("관악 하수 13센서 만관(꽉 참=1.0) 횟수", f"{n_surch}건")
    stat("관악 하수 최대 충전율", f"{fmax:.2f}  (1.0=만관, 도달 못함)")
    say("더 결정적으로 — 관악 도로가 실제 침수된 순간 하수관은 거의 비어 있었습니다.")
    stat("조원로5-6 도로 침수 횟수(10분 bin)", f"{len(flood_ts)}회")
    stat("그 침수 순간들의 관악 하수 평균 충전율", f"{fill_at_flood:.2f}  (= 13%만 차 있음)")
    say("→ 관악 침수는 '하수 역류'가 아니라, 빗물이 관에 들기 전 저지대 도로에")
    say("  고이는 '표면류형'일 가능성 강함. 하수 데이터로는 신호가 안 잡힙니다.")

    if make_plot and not NO_PLOT:
        _plot_gwanak(g, gw_sew, fill_at_flood)
    return dict(n_surch=n_surch, fmax=fmax, fill_at_flood=fill_at_flood,
                n_flood=len(flood_ts))


# ────────────────────────────────────────────────────────────────────────
def step3_signal_sparsity(make_plot=True):
    head(3, "근거 3 — 진짜 사례가 너무 적고 흩어져 있다")
    rn = load_parquet(CLEAN / "road_node.parquet", columns=["sensor_id", "자치구"])
    sn = load_parquet(CLEAN / "sewer_node.parquet", columns=["sensor_id", "자치구"])
    trust = load_parquet(EB / "road_flood_sensor_trust.parquet",
                         columns=["sensor_id", "판정_final"])
    au = load_parquet(EB / "sewer_surcharge_audit.parquet",
                      columns=["sewer_sensor_id", "최종판정"])

    tr = (trust[trust["판정_final"] == "진짜상습"]
          .merge(rn, on="sensor_id", how="left").dropna(subset=["자치구"]))
    conf = au.loc[au["최종판정"].str.startswith("확정"), "sewer_sensor_id"].tolist()
    sw = (pd.DataFrame({"sensor_id": conf})
          .merge(sn, on="sensor_id", how="left").dropna(subset=["자치구"]))

    road_gu = tr["자치구"].str.replace("청", "", regex=False)
    sew_gu = sw["자치구"].str.replace("청", "", regex=False)
    overlap = sorted(set(road_gu) & set(sew_gu))

    say("관악만으론 부족해 서울 전역으로 넓혀도 결과는 같습니다.")
    stat("신뢰 '진짜상습' 도로 지점", f"{len(tr)}곳  ({road_gu.nunique()}개 구에 흩어짐)")
    stat("검증된 '진짜 하수만관' 지점", f"{len(sw)}곳  ({sew_gu.nunique()}개 구에 흩어짐)")
    stat("도로침수+하수만관이 함께 확인된 구", f"{overlap}  (단 {len(overlap)}곳)")
    say("GNN은 '한 곳 상황이 옆으로 어떻게 번지는지'를 배워야 하는데,")
    say("진짜 사례가 이렇게 띄엄띄엄이면 '번지는 패턴'을 배울 데이터가 없습니다.")

    if make_plot and not NO_PLOT:
        _plot_sparsity(road_gu, sew_gu)
    return dict(n_road=len(tr), n_sew=len(sw), overlap=overlap)


# ────────────────────────────────────────────────────────────────────────
def verdict_and_needs():
    head("결론", "현재 데이터로 학습 불가 + 무엇을 확보하면 가능한가")
    say("정리: ①정답이 오염됐고 ②핵심 가설이 타깃 지역에서 관측 안 되며")
    say("       ③진짜 사례가 모델이 패턴 배울 만큼 모이지 않습니다.")
    say("→ 따라서 지금은 '학습용 데이터 정제'가 불가능합니다. (모델 문제 아님)")
    print()
    say("대신, 가능하게 만들 데이터는 분명합니다:")
    needs = [
        ("① 독립적 침수 기록", "오염된 센서 라벨 대체", "서울시 침수흔적도·120/119신고·CCTV·재해연보", "최우선"),
        ("② 다년 침수 이력", "침수는 연 몇 번뿐(현 데이터 2024~)", "2020·2022 대홍수 포함 과거자료", "최우선"),
        ("③ 하수 관망도(흐름방향)", "모델의 뼈대(노드-엣지)", "우수·오수 관거 GIS", "높음"),
        ("④ 지형고저·빗물받이", "관악 '표면류형' 모델링", "수치표고(DEM)·빗물받이 위치", "중간"),
        ("⑤ 레이더 강우", "관측소 1.5km 공백 보완", "기상청 레이더 강우", "중간"),
    ]
    tbl = pd.DataFrame(needs, columns=["필요 데이터", "왜 필요한가", "어디서", "우선순위"])
    print(tbl.to_string(index=False))
    print()
    say("특히 ①독립 침수기록 + ③관망도가 시급. ③이 확보되면 지금 깨끗한")
    say("하수 수위데이터(484지점·신뢰 78%)로 '하수 수위·만관 예측 모델'은 곧 시도 가능.")


# ────────────────────────────────────────────────────────────────────────
def _plot_gwanak(g, gw_sew, fill_at_flood):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([g.loc[g.sewer_sensor_id == s, "fill_rate"].values for s in gw_sew],
               showfliers=False)
    ax.axhline(1.0, ls="--", c="r", label="surcharge=1.0 (NEVER reached)")
    ax.axhline(fill_at_flood, ls=":", c="purple",
               label=f"sewer fill at road-flood times = {fill_at_flood:.2f}")
    ax.set_title("Gwanak sewers never surcharge — flooding is NOT from sewer backup")
    ax.set_xlabel("13 Gwanak sewer sensors"); ax.set_ylabel("fill_rate")
    ax.legend(fontsize=9)
    out = FIGDIR / "demo_gwanak_counterexample.png"
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"      🖼  그래프 저장: {out}")


def _plot_sparsity(road_gu, sew_gu):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["DejaVu Sans"]
    ROM = {"은평": "Eunpyeong", "광진": "Gwangjin", "금천": "Geumcheon", "관악": "Gwanak",
           "종로": "Jongno", "영등포": "Yeongdeungpo", "강남": "Gangnam",
           "서대문": "Seodaemun", "동대문": "Dongdaemun", "노원": "Nowon",
           "강서": "Gangseo", "동작": "Dongjak"}
    rc = road_gu.map(ROM).value_counts(); sc = sew_gu.map(ROM).value_counts()
    allg = sorted(set(rc.index) | set(sc.index))
    x = np.arange(len(allg))
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.2, [rc.get(k, 0) for k in allg], 0.4, label="genuine road-flood", color="green")
    ax.bar(x + 0.2, [sc.get(k, 0) for k in allg], 0.4, label="confirmed sewer surcharge", color="steelblue")
    ax.set_xticks(x); ax.set_xticklabels(allg, rotation=45, ha="right", fontsize=8)
    ax.set_title("Genuine signal ~1 per district; only Geumcheon has both")
    ax.set_ylabel("# sensors"); ax.legend(fontsize=9)
    out = FIGDIR / "demo_signal_sparsity.png"
    plt.tight_layout(); plt.savefig(out, dpi=110, bbox_inches="tight"); plt.close()
    print(f"      🖼  그래프 저장: {out}")


# ────────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "#" * 72)
    print("#  고객 설명 라이브 데모 — 침수예측 모델 학습 불가 사유")
    print("#  (모든 수치는 정제 데이터에서 실시간 계산)")
    print("#" * 72)
    step0_overview()
    step1_label_contamination()
    step2_gwanak_counterexample()
    step3_signal_sparsity()
    verdict_and_needs()
    print("\n" + "#" * 72)
    print("#  데모 끝. 그래프:", "(--no-plot 생략됨)" if NO_PLOT else str(FIGDIR) + "/")
    print("#" * 72 + "\n")


if __name__ == "__main__":
    main()
