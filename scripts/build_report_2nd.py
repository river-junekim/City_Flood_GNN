"""2차 보고서(검증된 발견) — 그림 생성 + HTML 작성.

내용 = docs/REPORT_2차_검증발견.md 를 오늘 자로 HTML 화. 핵심 그림:
  (1) 2022-08-08 관악 하수 7센서 동시 반응 수문곡선  (★ 실제 침수 포착 증거)
  (2) 강우 교차검증 5사건 요약 표(텍스트 표)
기준: 관악구 · 도로 침수 = 도로수위계 ≥15cm(서울시) · 하수 충전율은 별개 신호.
실행:  python scripts/build_report_2nd.py
"""
from __future__ import annotations
import os, sys, base64
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")
from krfont import set_korean
set_korean()

import pandas as pd
import matplotlib.pyplot as plt

FD = "reports/figures_demo"
OUT = "reports/report_2nd_findings.html"
SEWER = "dataset/processed/eda_based/sewer_features_10min.parquet"
TODAY = "2026-07-02"

# 2022-08-08 당시 운영 중이던 관악 하수 7센서 (id → 지점명)
SENSORS_2022 = {
    "21-0001": "삼성808",
    "21-0004": "낙성대",
    "21-0007": "신림1450",
    "21-0005": "보라매968",
    "21-0003": "미성",
    "21-0006": "행운",
    "21-0002": "조원547",
}


def fig_2022_hydrograph():
    """2022-08-08 저녁 관악 하수 7센서 fill_rate 수문곡선."""
    df = pd.read_parquet(SEWER, columns=["sewer_sensor_id", "ts10", "fill_rate"])
    df["ts10"] = pd.to_datetime(df["ts10"])
    d = df[(df["ts10"] >= "2022-08-08 17:00") & (df["ts10"] < "2022-08-09 03:00")
           & (df["sewer_sensor_id"].isin(SENSORS_2022))]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    for sid, name in SENSORS_2022.items():
        s = d[d["sewer_sensor_id"] == sid].sort_values("ts10")
        if len(s) == 0:
            continue
        pk = s["fill_rate"].max()
        ax.plot(s["ts10"], s["fill_rate"], marker="o", ms=3, lw=1.6,
                label=f"{sid} {name} (피크 {pk:.2f})")
    ax.axhline(0.6, color="#888", ls="--", lw=1, label="fill 0.6 (동시반응 기준)")
    ax.set_ylabel("하수관 충전율 fill_rate")
    ax.set_xlabel("2022-08-08 시각")
    ax.set_ylim(0, 1.0)
    ax.set_title("2022-08-08 관악 하수 7센서 동시 반응 — 상승→피크→하강 수문곡선\n"
                 "[관악 대침수일 · 10분 집계 · fill_rate = 수위/관높이]", fontsize=11, y=1.06)
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    p = f"{FD}/req_2022_hydrograph.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("그림:", p)
    return p


def embed(path, cap=""):
    if not os.path.exists(path):
        return f'<p class="cap">[그림 없음: {path}]</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    c = f'<div class="cap">{cap}</div>' if cap else ""
    return f'<img src="data:image/png;base64,{b}">{c}'


CSS = """
body{font-family:'Malgun Gothic','Apple SD Gothic Neo','NanumSquareRound',sans-serif;max-width:1000px;margin:24px auto;padding:0 18px;color:#222;line-height:1.7}
h1{border-bottom:3px solid #2c7;padding-bottom:8px}
h2{border-left:5px solid #2c7;padding-left:10px;margin-top:30px}
.meta{color:#666;font-size:13px}
.box{border:1px solid #ddd;border-left:4px solid #2c7;border-radius:6px;padding:12px 16px;margin:14px 0;background:#f7fbf8}
.star{color:#c0392b;font-weight:bold}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:center}
th{background:#eef7f0}
td.l{text-align:left}
img{max-width:100%;border:1px solid #ccc;border-radius:4px;margin-top:8px}
.cap{font-size:12px;color:#666;margin-top:4px}
.warn{background:#fff3cd;border:1px solid #ffe08a;border-radius:4px;padding:8px 12px;font-size:14px}
ul{margin:6px 0}
"""


def build(hydro_png, map_png):
    ev_rows = "".join(f"<tr><td>{d}</td><td>{n}</td><td>{r}</td><td>{v}</td></tr>" for d, n, r, v in [
        ("2025-05-16", "5", "36.0mm", "✅ 강우 15:40 급증 → 센서 15:54~16:07 반응"),
        ("2025-08-30", "3", "41.5mm", "✅"),
        ("2025-07-19", "2", "41.0mm", "✅"),
        ("2025-07-17", "3", "31.5mm", "✅"),
        ("2024-07-22", "3", "30.5mm", "✅"),
    ])
    s7_rows = "".join(f"<tr><td>{i}</td><td>{n}</td><td>{p}</td><td>{d}</td><td>{t}</td></tr>" for i, n, p, d, t in [
        ("21-0001", "삼성808", "0.88", "160분", "21:00"),
        ("21-0004", "낙성대", "0.82", "150분", "22:10"),
        ("21-0007", "신림1450", "0.78", "190분", "23:00"),
        ("21-0005", "보라매968", "0.74", "130분", "21:50"),
        ("21-0003", "미성", "0.74", "150분", "20:50"),
        ("21-0006", "행운", "0.71", "80분", "21:00"),
        ("21-0002", "조원547", "0.68", "20분", "21:50"),
    ])
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>2차 보고서 — 검증된 발견</title><style>{CSS}</style></head><body>
<h1>2차 보고서 — 검증된 발견: 하수 다중센서가 실제 침수를 포착한다</h1>
<p class="meta">기준: 관악구 · 도로 침수 = 도로수위계 ≥15cm(서울시 침수예보 발령 기준) · 하수 충전율은 별개 신호로 분리 · 작성 {TODAY} · 1차 보고서(노드-링크 상관관계 분석)의 후속</p>

<div class="box"><b>핵심 전환 한 줄</b> — 1차에서 "관악 도로센서 신뢰 침수 양성 ≈ 0"으로 보았으나,
<b>재조사 결과 하수센서가 실제 침수 사건을 다수 포착</b>하고 있었다. "양성 0"의 원인은
<b>센서 결함이 아니라 관측창 한계</b>(도로센서 2024-06 설치·AWS 강우 여름만 보유)였다.</div>

<h2>1. 침수 판단 기준 (검증된 결정)</h2>
<ul>
<li><b>도로 침수 = 도로수위계 침수심 ≥ 15cm</b> (출처: 서울시 침수예보 발령 기준, news.seoul.go.kr/env). 기존 임의 임계(≥6cm) 폐기.</li>
<li><b>하수 충전율(fill_rate, 만관/위험수위)은 침수 기준이 아니라 별개 신호로 분리</b> — 도로 표면 침수 ≠ 하수관 만관.</li>
</ul>

<h2>2. <span class="star">★</span> 2022-08-08 (관악 대침수일) — 하수 7센서 동시 포착</h2>
<p>당시 운영 중이던 관악 하수센서 <b>7개 전부</b>가 그날 저녁(20:50~23:00) 동시 반응. 정제 데이터(10분 집계)로 직접 재현:</p>
{embed(hydro_png, "2022-08-08 관악 하수 7센서 수문곡선 — 스파이크·고착이 아니라 상승→피크→하강. 피크 시각을 조금씩 달리하며 동시 반응(광역 침수의 교과서적 신호). ※ 21-0005의 21:00~21:30 센서 드롭아웃(바닥값 0.01m 급락 후 복귀) 구간은 전처리에서 선형보간으로 정정(is_dropout 플래그, 노트북 sewer_dropout_clean).")}
<table>
<tr><th>센서</th><th>지점</th><th>peak fill</th><th>fill≥0.6 지속</th><th>피크시각</th></tr>
{s7_rows}
</table>
<ul>
<li>7센서가 <b>같은 날 저녁</b>에, <b>수십~수백 분 지속</b>으로, 피크 시각을 조금씩 달리하며 동시 반응.</li>
<li>강우: 우리 AWS 강우창(2024-06~) 밖이라 <b>직접 확인은 진행 중</b>(과거 강우 추가 수집 중). 외부 기록으로 <b>2022-08-08 시간당 141.5mm·일 381.5mm</b>(신대방동 관측)가 뒷받침.</li>
<li><b>전처리 정정</b>: 하수 센서 일부에서 고수위 중 순간 바닥값(0.01m) 급락 후 복귀하는 <b>센서 드롭아웃</b>이 발견돼(데이터셋 전체 4,356건·0.021%·100센서), 고수위 사이에 낀 바닥값 런만 선형보간으로 정정(`is_dropout` 플래그, 노트북 `sewer_dropout_clean`). 원자료(1분)는 보존. 위 7센서 수문곡선은 정정 후 기준.</li>
</ul>

<div class="box"><b><span class="star">★</span> 침수흔적도 대조 — ground-truth 확인 완료</b><br>
서울시 공식 <b>침수흔적도 2022</b>(서울 열린데이터광장 OA-15636)를 확보해 위 7센서와 공간·시간 대조한 결과:
<ul>
<li>관악 2022-08-08 침수 폴리곤 <b>2,120개</b>(총 105만 m²) — 원인에 <b>"배수용량초과"(하수 만관/월류) 1,002개</b> 포함(우리 하수센서가 포착할 침수 유형과 일치).</li>
<li><b>동시반응 7센서 전부가 실제 침수 폴리곤 위/인접(최근접 ≤102m·중앙값 0m)</b>, 300m내 침수 폴리곤 39~201개.</li>
<li>→ <b>센서 신호(동시반응) = 실제 침수 위치·날짜(2022-08-08)와 일치</b>. 하수 다중센서 사건이 <b>검증된 침수 라벨</b>임이 공식 ground-truth로 확인됨.</li>
</ul>
{embed(map_png, "관악 2022-08-08 침수흔적도(파랑) × 하수 7센서(빨간 별) — 7센서 전부 실제 침수구역 위/인접. 출처: 서울시 침수흔적도 2022(서울 열린데이터광장 OA-15636, 공공누리 1유형). 노트북 flood_trace_crosscheck.")}
</div>

<h2>3. <span class="star">★</span> 강우 교차검증 — 확인 가능한 다중센서 사건은 전부 강우와 일치</h2>
<p>AWS 강우를 보유/추가수집해 확인 가능한 다중센서 사건 5개가 <b>모두 실제 강우(30~42mm)와 시각 일치</b>:</p>
<table>
<tr><th>날짜</th><th>동시반응 센서</th><th>관악 최대 시간강우</th><th>검증</th></tr>
{ev_rows}
</table>
<p>→ <b>강우를 확인할 수 있는 경우, 하수 다중센서 반응은 예외 없이 강우와 맞물린다.</b></p>

<h2>4. AWS 강우 커버리지 갭 발견·보완 (검증의 전제)</h2>
<ul>
<li>기존 AWS 강우 = <b>여름(6~9월) 2024·2025만</b> 다운로드돼 있었음 → 비여름·과거 사건의 "강우데이터없음"은 <b>무강우가 아니라 미수집</b>.</li>
<li>2025-05 보완 완료(→ 2025-05-16 사건 검증). <b>2023~ 과거 강우 추가 수집 진행 중</b>(용량 제한으로 다회 분할·재개, {TODAY} 기준 재개 실행 중).</li>
</ul>

<h2>5. 재해석 — "양성 0"의 진짜 원인 (관측창 한계)</h2>
<ul>
<li><b>도로센서는 2024-06 설치</b> → 2022·2023 대형 침수를 데이터에 담지 못함.</li>
<li><b>AWS 강우는 여름만 보유</b> → 비여름·과거 강우 검증 불가.</li>
<li>따라서 1차의 "현 데이터로 supervised 학습 불가"는 <i>센서가 못 쓴다</i>가 아니라 <b>우리 분석 창이 큰 사건을 비껴갔다</b>가 정확한 진술.</li>
</ul>

<h2>6. 모델 학습 관점 의의 + 데이터 요청 연결</h2>
<ul>
<li><b>하수 다중센서 동시반응 = 침수 정답 라벨 후보</b> (지속·동시·수문곡선·강우동반으로 신뢰도 평가 가능).</li>
<li>단 <b>깨끗한 다년 라벨</b>에는 (a) 과거 강우(2022·2023, 수집 중), (b) 침수흔적도·120/119 신고가 필요 → <b>데이터 요청 ①(검증된 침수 라벨)과 직접 연결.</b></li>
<li>관측 밀도: 2022 당시 7센서 → 현재 13센서 → 신설 29 확보 시 42센서. 사건 포착력 향상.</li>
<li><b>노드 라벨 생성 완료(2026-07-02)</b>: 침수흔적도 폴리곤을 <b>GNN 맨홀 노드 13,272개</b>에 공간조인 → 2022-08-08 기준 노드별 침수 라벨(내부 1,229·≤50m 6,709·침수심·원인). 센서 42노드 전부 라벨 부여(사건 7센서 중 6/7이 50m내 침수). → 라벨 부재로 막혔던 하수 라우팅 GNN에 <b>실제 침수 정답(node label)</b> 확보. 노트북 `02_analysis/floodtrace_manhole_join`, 산출 `gnn_manhole_flood_labels_2022.parquet`.</li>
</ul>

<h2>7. 외부 자료 — 침수흔적도(ground-truth)·서울시 2026 대책</h2>
<p><b>(1) 침수흔적도 = ground-truth 침수 라벨 — 확보·대조 완료(2026-07-02)</b></p>
<ul>
<li>서울 열린데이터광장 <b>OA-15636</b>(서울시 침수흔적도, 2010~2025년 연도별·공공누리 1유형)에서 <b>2022년분 확보</b>(`03_GIS/침수흔적도_shp/`). safemap.go.kr(2002~2023)·safecity.seoul.go.kr도 제공.</li>
<li>속성: 구·침수심·발생일자/시각·<b>원인(하수역류·배수용량초과 등)</b>·주소·면적·폴리곤 → 데이터 요청 ①(검증된 침수 라벨)의 상당 부분을 <b>공공데이터로 즉시 확보 가능</b>(§2 대조 참조).</li>
<li><b>대조 결과(§2)</b>: 2022-08-08 동시반응 7센서 전부가 실제 침수구역 위/인접 → 센서 사건의 ground-truth 검증 완료. <b>다음</b>: 나머지 연도(2010~2020·2024·2025) 확보 → 전 기간 라벨링.</li>
</ul>
<p><b>(2) 서울시 2026 풍수해 안전대책 = 본 접근의 타당성 공식 입증</b></p>
<ul>
<li>서울시 2026: <b>"AI 기반 침수 예측 — 과거 강우량 + 도로·하수관로 수위 데이터 학습"</b>(강남역 등 15개소) = 본 프로젝트 접근을 서울시가 공식 채택.</li>
<li><b>관악구 = 반지하 밀집 → 소형 레이더 수위계 30개 추가</b>('25년 15→'26년 45) = 데이터 요청 ④(신설 센서)와 직결.</li>
<li>출처: safemap.go.kr · safecity.seoul.go.kr.</li>
</ul>

<h2>미확정 / 다음</h2>
<div class="warn">
<ul>
<li>[ ] 2022·2023 강우 수집 완료 후 2022-08-08·2023-08-11 등 강우 <b>재검증</b> ({TODAY} 다운로드 재개, 완료 대기).</li>
<li>[ ] 신뢰 침수 사건 카탈로그 정식 구축(센서수·지속·피크·강우·shape 기준).</li>
<li>[x] <s>침수흔적도 확보 → 2022-08-08 하수 7센서 사건과 위치·날짜 대조</s> — <b>완료(2026-07-02, §2·§7)</b>. 다음: 나머지 연도 확보.</li>
<li>[ ] 형태가 다른 일자(2024-03-14 1분 분산 스파이크)는 카탈로그에서 제외(사용자 지시).</li>
</ul>
</div>
</body></html>"""


MAP_PNG = f"{FD}/floodtrace_2022_gwanak.png"  # flood_trace_crosscheck 노트북 산출

if __name__ == "__main__":
    hydro = fig_2022_hydrograph()
    open(OUT, "w").write(build(hydro, MAP_PNG))
    print("작성:", OUT)
