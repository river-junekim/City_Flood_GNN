"""KICT 데이터 요청 보고서(HTML) 생성 — 관악구 침수 예측 AI 모델.

그림(reports/figures_demo, figures_gnn)을 base64로 임베드해 자체완결 HTML 생성.
내용 수정은 아래 데이터 구조(SECTIONS/REQUESTS)만 고치면 됨.
실행:  python scripts/build_data_request_report.py
산출:  reports/data_request_report.html
"""
from __future__ import annotations
import base64
import os

os.chdir("/home/namjun/city_flood")
OUT = "reports/data_request_report.html"

CSS = """
body{font-family:'Malgun Gothic','NanumGothic',sans-serif;max-width:980px;margin:24px auto;padding:0 18px;color:#222;line-height:1.65}
h1{font-size:25px;border-bottom:3px solid #1565c0;padding-bottom:10px}
h2{font-size:20px;margin-top:34px;border-left:6px solid #1565c0;padding-left:10px}
h3{font-size:16px;margin-top:22px;color:#0d47a1}
figure{margin:16px 0;text-align:center} img{max-width:100%;border:1px solid #ddd;border-radius:6px}
figcaption{font-size:12.5px;color:#666;margin-top:6px}
.lead{background:#e3f2fd;padding:12px 14px;border-radius:6px}
.key{background:#e8f5e9;padding:10px 14px;border-left:5px solid #2e7d32;border-radius:4px;margin:10px 0}
.warn{background:#fff3e0;padding:10px 14px;border-left:5px solid #ef6c00;border-radius:4px;margin:10px 0}
.bad{background:#ffebee;padding:10px 14px;border-left:5px solid #c62828;border-radius:4px;margin:10px 0}
.note{background:#f6f8fb;padding:10px 14px;border-left:5px solid #607d8b;border-radius:4px;margin:10px 0}
.draft{background:#fffde7;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}
.draft p{margin:8px 0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:14px}
th,td{border:1px solid #ccc;padding:7px 9px;text-align:left} th{background:#f0f4f8}
.req{background:#fafafa;border:1px solid #ddd;border-radius:8px;padding:6px 16px;margin:14px 0}
.tag{display:inline-block;background:#1565c0;color:#fff;font-size:12px;padding:2px 8px;border-radius:10px}
.meta{color:#888;font-size:12.5px}
"""

FD = "reports/figures_demo"
FG = "reports/figures_gnn"

# 요청 항목 (우선순위순). 내용 수정은 여기만.
REQUESTS = [
    dict(title="② 위험기상 예측정보 노드 = 격자(레이더형) 강우", tag="노드 입력(강우) · 리드타임의 본체",
         요청="이동하는 강우장을 담은 <b>격자 강수자료</b> — 기상청 레이더 합성(HSR 등)·KICT 보정 강수장 등 가용한 것이면 됨",
         명세="<b>기능 요건</b>: 점이 아닌 공간 격자 · 관악을 해상하기 충분한 해상도(예: 1km 이하) · 시간 5~10분급 · 상류 포함 영역(이동 관측) · 2024~현재 · 좌표계/격자ID/강수단위/결측코드. <b>※기상청 HSR(500m·5분)은 충족 예시일 뿐 필수 규격 아님</b> — 동등한 격자자료면 가능",
         사유="리드타임은 모델이 만드는 게 아니라 '미래를 담은 입력'에서 나옴. 두 개의 시계 중 <b>(가) 기상 시계</b>(비가 아직 관악에 안 옴)만 10~60분 선행을 줌 = 레이더. 관악 <b>(나) 수문 시계</b>(유출 지연)는 거의 0(강우-수위 동시반응, persistence 0.72 ≫ GNN 0.17)이라 그래프만으론 몇 분뿐 → <b>레이더가 리드타임의 본체</b>. 점 강우는 리드타임 0(빗방울이 강우계 때리는 순간 이미 온 것). [정직한 한계] 관악 침수=국지 대류 호우는 nowcasting이 가장 약한 유형(이동뿐 아니라 생성·소멸)이라 리드타임 천장은 보수적(현실 10~30분). 구조: 레이더 강우장 → nowcast → 예측 강우장 → ×집수면적 → 맨홀 유입 → 유향 GNN(2단 분리).",
         figs=[(f"{FG}/08_stgnn.png", "그림 3. 관악 시공간 GNN. 현재 입력(수위 이력·점강우)만으로는 persistence(0.72/0.57/0.51)를 넘지 못함 = 조기 입력 리드타임 부재 → 레이더 필요.")]),
    dict(title="③ DEM (수치표고) — 또는 KICT 하수관망 수리모델(SWMM)", tag="모델: 노드 입력(집수면적)",
         요청="고해상 DEM/LiDAR <b>또는</b> KICT 보유 하수관망 수리모델(SWMM 등, 소배수구역·맨홀별 집수면적 포함) — 둘 중 하나",
         명세="1순위: SWMM .inp/수리모델(소배수구역·맨홀별 집수면적 포함) · 대안: DEM/LiDAR 1m급 또는 가용 최고해상",
         사유="모델 노드 입력 = 맨홀 유입량 = <b>강우 × 집수면적</b>(합리식 Q=C·i·A). 강우(②)만으론 유입 부피가 안 나오고, 면적이 곱해져야 '실제 들어오는 물의 양'이 됨. 같은 비라도 집수면적 큰 맨홀이 먼저 위험 → 공간 위험 구분에 면적 가중 필수. <b>GIS 소유역은 40개로 coarse</b>(이미 활용) → 맨홀별 정밀 집수면적은 DEM(흐름누적·저지대)에서 도출하거나 SWMM이 있으면 그대로 사용.",
         figs=[(f"{FD}/req2_dem_rationale.png", "그림 4. 조원로 침수 시각 하수·하천 수위(여름). 침수 때 하수(위험수위 미달)·하천(평상시) 모두 정상 → 표면류."),
               (f"{FD}/req_sewer_reach.png", "그림 5. 관악 하수관로 충전율 도달 분포(여름). 위험수위(0.8) 도달 극소·만관(1.0) 0개 = 하수 신호 부재(=하수 만관 가설 기각)."),
               (f"{FD}/req_river_useless.png", "그림 6. 하천역류 가설 기각. 조원로 침수 30회 중 93%가 도림천 평상시 수위(p95 미만)이고 침수시 중앙(2.46m)이 전체 중앙(2.39m)과 사실상 동일 → 하천 고수위(역류) 동반이 거의 없어 하천 수위로는 침수를 전혀 구분할 수 없음. 만관 기각(그림 5)과 함께 소거법으로 '관악 침수 = 국지 호우 표면류' 확정 → DEM(저지대·집수면적) 필요.")]),
    dict(title="④ 관악 신설 하수 수위계 시계열", tag="모델: 관측 노드 밀집",
         요청="관악 신설 하수 수위계 29개(21-0014~0042, 2025-08~) 및 이후 최신 기간 시계열",
         명세="수위계 21-0014~0042 · 1·10분 수위 · 2025-09~현재",
         사유="2026년 우기 실시간 검증과 semi-supervised 관측 밀도 확보에 필요. 관측 노드가 13→42(3배)로 늘며, 이미 수집 중인 시계열이라 제공 비용이 낮음. 우리 데이터는 2025-08-31까지라 2025-09 이후 신설분도 미보유.",
         figs=[], extra='<p class="meta">(그림 0b 참조)</p>'),
    dict(title="① 침수 정답 라벨", tag="후순위: 예측 확장·검증용",
         요청="KICT 침수 검증(ground-truth) 데이터 · 관악 침수흔적도 · 120/119 침수신고",
         명세="침수흔적도 SHP(발생일자 포함) · 120/119 신고(비식별화 가능: 일시·좌표/주소·침수심/피해유형) · 가능 연도(2020·2022 대홍수 포함 희망)",
         사유="<b>상관관계 분석 모듈의 선결조건은 아님(후순위)</b>. 단 모듈을 <i>예측</i>으로 확장·검증하려면 검증된 정답지 필수 — 관악 깨끗한 양성이 1지점·24건이라 현 데이터로는 예측 검증 불가.",
         figs=[(f"{FD}/req1_label_contamination.png", "그림 2. 관악 도로센서 침수 양성(여름). 실제 침수=조원로 24, 나머지는 센서 오작동/미판정.")]),
]


def embed(path: str, cap: str) -> str:
    if not os.path.exists(path):
        return f'<p style="color:#b00">[그림 없음: {path}]</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}"/><figcaption>{cap}</figcaption></figure>'


def req_block(r: dict) -> str:
    figs = "".join(embed(p, c) for p, c in r["figs"]) + r.get("extra", "")
    명세 = f'<br><b>명세</b>: <span class="meta">{r["명세"]}</span>' if r.get("명세") else ""
    return (f'<div class="req"><h3>{r["title"]} <span class="tag">{r["tag"]}</span></h3>'
            f'<p><b>요청</b>: {r["요청"]}{명세}<br><b>사유</b>: {r["사유"]}</p>{figs}</div>')


def build() -> str:
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>KICT 데이터 요청 보고서 — 관악 노드-링크 상관관계 분석 모듈</title>
<style>{CSS}</style></head><body>

<h1>데이터 요청 보고서 — 관악 노드-링크 상관관계 분석 모듈</h1>
<p class="meta">수신: 한국건설기술연구원(KICT) · 작성 2026-06-24 · 기준: <b>관악구(테스트베드) · 여름 우기(6~9월)</b> · 과업: <b>노드(맨홀)-링크(관로) 상관관계 분석 모듈</b>(위험기상 예측정보 노드 포함 · KICT 개발 알고리즘 기반)</p>
<p class="lead">관악 관망 GIS + 수위 센서로 <b>노드(맨홀)-링크(관로) 그래프를 구축·검증하고 상관관계 분석 모듈을 시제 구현</b>했으나, <b>노드 입력(위험기상 예측 × 집수면적)과 관측 밀도가 비어</b> 노드-링크 상관/전파 신호가 약했습니다. 본 보고서는 <b>그 진행 경과와 실패 사유를 정리하고, 모듈 완성에 필요한 데이터를 요청</b>합니다.</p>

<h2>0. 과업 그래프 구조와 데이터의 관계</h2>
<p>과업 = <b>노드-링크 상관관계 분석 모듈</b>(테스트베드 관악) — <b>노드=맨홀</b>(원자료 12,184개·그래프 노드 13,272개), <b>링크=관로</b>(상류→하류 유향, 관저고 기반, GIS 보유), <b>위험기상 예측정보 노드</b>(강우 예측을 맨홀별 유입으로 구조화), <b>분석=KICT 개발 알고리즘 기반</b> 네트워크에서 노드(맨홀수위)-링크(관로흐름) 상관관계 산출. 침수 라벨은 후속 <i>예측</i> 확장·검증용(선결조건 아님).</p>
<div class="note"><b>숫자 기준</b>: 원자료 맨홀 시설물 <b>12,184개</b>와 그래프 노드 <b>13,272개</b>는 서로 다른 기준입니다. 그래프 노드는 관거의 시작·끝 맨홀 ID를 기준으로 구성한 모델 입력 단위입니다. 관거 레코드 중 엣지화 가능한 항목은 <b>16,059개</b>이고, 중복·정리 후 NetworkX 유향 그래프 기준 엣지는 <b>15,217개</b>입니다.</div>
{embed(f"{FG}/09_roadmap.png", "그림 0. 맨홀-노드 GNN 구성요소와 필요 데이터. 관망·맨홀·빗물받이(녹색)는 GIS로 확보, 요청 ①~④(주황·빨강)가 나머지를 채운다.")}
<p><b>골격은 개념이 아니라 이미 구축·검증됐습니다.</b> KICT 관악 관망 GIS의 <b>공식 연결(시작맨홀→끝맨홀)</b>로 그래프 노드 <b>13,272개</b>와 엣지화 가능 관거 레코드 <b>16,059개</b>를 구성했습니다. 이 중 중복·정리 후 NetworkX 유향 그래프 기준 엣지는 <b>15,217개</b>입니다. 공식 <b>접속관수로 교차검증(차수 정확 80%·±1 91%)</b>·<b>공간정합 77%</b>(나머지 플래그, 완전검증판=SWMM)했고, 공식 경사·소유역(집수단위 40)도 부착했습니다. 비어있는 건 입력·정답뿐입니다.</p>
{embed(f"{FG}/10_manhole_graph.png", "그림 0b. 관악 하수 관망 그래프 + 수위센서(통합). 노드=맨홀 13,272·엣지=유향 관거 16,059(차수검증 80%·±1 91%·공간정합 77%). <b>각 색=1 소유역(집수단위 40, coarse 집수면적=③ 근거)</b>. 센서: 하수 데이터보유 13(검은 원)·신설 29(빨간 X, 2025-08)·도로 5(녹색 삼각) — <b>등록 47 vs 보유 18</b>.")}
<p>이 그래프 위에 <b>맨홀-노드 STGNN을 실제 구현·시험</b>했습니다. 다만 <b>13,272개 맨홀 중 관측은 13개뿐</b>이라 나머지는 입력이 비어 있는 통과(relay) 노드로 남고, 그래서 그래프 전파 효과가 아직 미미합니다 — relay 노드를 채울 <b>맨홀별 유입(②레이더 강우 × ③집수면적)</b>이 그래프가 실제로 신호를 전파하기 위한 직접 조건입니다.</p>
<div class="key">관망·맨홀(노드-링크 골격)은 이미 확보·구축·검증됨. <b>빈 곳은 ②위험기상 예측정보 ③집수면적 ④관측 노드</b>(상관관계 분석 모듈 입력) · <b>①침수 라벨은 후속 예측 확장용</b> → 아래 요청이 각각을 채웁니다.</div>

<h2>1. 진행 경과와 미완(실패) 사유 (관악·여름)</h2>
{embed(f"{FD}/req_why_infeasible.png", "그림 1. 관악 도로 침수 supervised 학습 가용성. 신뢰 양성 조원로 24건(1지점)·불균형 1:1,208·하수 위험수위 도달 0.064%.")}
<div class="bad"><b>그래프 골격은 구축·검증됐으나 노드-링크 상관/전파 신호가 약함</b>: 도로↔하수 결합 corr ~0.05 · 센서간 전파 부재(강우+이력 GNN 0.17 ≪ persistence 0.72) · 하수 만관 0%·위험수위 0.064% · 하천역류 기각(침수 90%가 도림천 평상시). <b>원인 = 노드 입력(위험기상 예측 × 집수면적)·관측 밀도가 비어 상관관계가 드러날 입력 자체가 부재.</b></div>
<div class="warn">센서 인프라는 충분하나 데이터가 부족: <b>등록 47(하수 42+도로 5) vs 데이터 보유 18</b> — 하수 29개(2025-08 신설)는 관망 위에 있으나 데이터 없음(그림 0b 참조).</div>

<h2>2. 요청 데이터</h2>
<p class="meta"><b>상관관계 분석 모듈 기준 우선순위: ②·③(노드 입력) → ④(관측 밀도) → ①(예측 확장·검증)</b></p>
<table>
<tr><th>우선순위</th><th>요청 데이터</th><th>채우는 구성요소</th><th>없을 때 병목</th><th>최소 제공 형태</th></tr>
<tr><td>②</td><td>위험기상 예측정보(격자 강우)</td><td>노드 입력(강우)</td><td>점강우·수위이력은 리드타임 부족</td><td>격자ID·시간·강수량·좌표계·결측코드</td></tr>
<tr><td>③</td><td>SWMM/DEM</td><td>노드 입력(집수면적)</td><td>강우를 맨홀별 유입량으로 변환 불가</td><td>SWMM .inp 또는 DEM/LiDAR</td></tr>
<tr><td>④</td><td>신설 하수 수위계</td><td>관측 노드 밀집</td><td>관측 노드 13개로 공간 검증 제한</td><td>21-0014~0042, 1·10분 수위, 2025-09~현재</td></tr>
<tr><td>①</td><td>침수 정답 라벨</td><td>예측 확장·검증(후순위)</td><td>모듈을 예측으로 확장 시 정답지 부재</td><td>발생일시·위치·침수심/유형(비식별 가능)</td></tr>
</table>
{''.join(req_block(r) for r in REQUESTS)}

<h2>3. 데이터 수신 후 활용 계획</h2>
<table>
<tr><th>단계</th><th>작업</th><th>산출물</th></tr>
<tr><td>1</td><td>좌표계·시간축·센서 ID를 기존 관망 그래프와 정합</td><td>정합 테이블, 품질 점검표</td></tr>
<tr><td>2</td><td>위험기상 예측정보와 집수면적을 결합해 맨홀별 유입(노드 입력) 생성</td><td>맨홀별 유입 시계열</td></tr>
<tr><td>3</td><td>KICT 알고리즘 기반 네트워크에서 노드-링크 상관관계 분석 모듈 산출</td><td>노드-링크 상관/전파 지표</td></tr>
<tr><td>4</td><td>(예측 확장 시) 침수 라벨로 persistence 대비 성능 검증</td><td>CSI·POD·FAR·리드타임 비교</td></tr>
</table>
<div class="warn"><b>부분 제공 시 우선순위</b>: 상관관계 분석 모듈은 ②위험기상 예측정보·③집수면적(노드 입력)과 ④관측 밀도가 먼저 채워져야 노드-링크 신호가 드러납니다. ①침수 라벨은 이후 예측 확장·검증 단계에서 필요합니다.</div>

<h2>4. 회신 요청 (방향 정렬)</h2>
<ol>
<li><b>KICT 개발 알고리즘</b>(네트워크 설계 근거) 명세·문서 공유 — 맨홀-노드 유향그래프와의 정합 확인.</li>
<li><b>위험기상 예측정보</b>의 정의·제공 형식(레이더 격자 / 초단기예측 / 특보 중 무엇인지).</li>
<li>KICT ground-truth 침수 검증 보유 여부(후속 <i>예측</i> 확장·모듈 검증용 — 이미 보유 시 ① 해소).</li>
<li>관악 신설 29센서 제공 가능 여부·형식 · 하수관망 수리모델(SWMM)/고해상 DEM 보유 여부.</li>
</ol>
<div class="key">관망·맨홀(노드-링크 골격)은 확보·검증됨. <b>②위험기상 예측정보·③집수면적·④관측 밀도가 상관관계 분석 모듈을 완성</b>하고, ①침수 라벨은 이후 <i>예측</i> 확장의 정답지입니다.</div>

<h2>5. 별도 정리용 문장 모음</h2>
<div class="draft">
<p><b>요약</b>: 관악 관망 GIS와 수위 센서로 노드(맨홀)-링크(관로) 그래프를 구축·검증하고 상관관계 분석 모듈을 시제 구현했으나, 노드 입력(위험기상 예측 × 집수면적)과 관측 밀도가 비어 노드-링크 상관/전파 신호가 약했습니다. 따라서 본 요청은 추가 자료 수집이 아니라, 이미 구축된 노드-링크 그래프의 빈 입력을 채워 상관관계 분석 모듈을 완성하기 위한 최소 데이터 요청입니다.</p>
<p><b>숫자 기준</b>: 본 보고서의 맨홀 수와 그래프 노드 수는 서로 다른 기준입니다. 원자료 맨홀 시설물은 12,184개이며, 관거의 시작·끝 맨홀 ID를 기준으로 그래프화하면 노드 13,272개가 생성됩니다. 관거 레코드 중 엣지화 가능한 항목은 16,059개이고, 중복·정리 후 NetworkX 유향 그래프 기준 엣지는 15,217개입니다.</p>
<p><b>침수 라벨</b>: 침수 정답 라벨은 본 과업의 최우선 요청 데이터입니다. 현재 관악구 도로센서 기준으로 신뢰 가능한 침수 양성은 조원로 1지점 24건 수준에 그쳐, supervised 학습을 수행하면 실제 침수가 아니라 센서 오류나 하수 수위 대리지표를 학습할 위험이 큽니다.</p>
<p><b>레이더 강우</b>: 리드타임은 모델이 만들어내는 것이 아니라, 미래를 담은 입력에서 나옵니다. 침수 사건의 시간 여유는 두 개의 시계로 나뉩니다 — <b>기상 시계</b>(비가 아직 관악에 도달하지 않음)와 <b>수문 시계</b>(비는 내렸으나 아직 유출·월류 전). 관악은 도시 소유역이 작아 수문 시계가 거의 0이며(강우-수위가 동시 반응, persistence 0.72 ≫ GNN 0.17), 따라서 유향 그래프만으로 버는 리드타임은 몇 분에 불과합니다. 의미 있는 10~60분 리드타임의 본체는 <b>기상 시계 = 레이더</b>입니다. 레이더는 공간적으로 이동하는 강우장을 상류에서 먼저 관측하므로 맨홀별 유입을 시간적으로 앞서 입력합니다. 점 강우계는 빗방울이 계기를 때리는 순간 이미 비가 온 것이라 리드타임이 0입니다. 다만 정직하게, 관악 침수형인 국지 대류 호우는 셀이 이동만 하는 것이 아니라 그 자리에서 생성·소멸하므로 nowcasting이 가장 어려운 유형이며, 리드타임 천장은 보수적으로 10~30분 수준으로 봅니다. 구현은 [레이더 강우장 → nowcast → 예측 강우장 → ×집수면적 → 맨홀 유입 → 유향 GNN]의 2단 구조로 분리하여, 강우 예측 단과 침수 라우팅 단을 독립적으로 검증·교체할 수 있게 설계합니다.</p>
<p><b>SWMM/DEM</b>: 맨홀-노드 GNN의 노드 입력은 단순 강우량이 아니라 맨홀별 유입량입니다. 유입량은 강우강도에 집수면적이 곱해져야 산정되므로, 같은 비가 내려도 집수면적이 큰 맨홀은 더 빨리 위험 상태에 도달할 수 있습니다. 따라서 SWMM 등 수리모델이 있으면 가장 직접적으로 활용 가능하며, 없을 경우 고해상 DEM/LiDAR를 통해 맨홀별 집수면적을 추정해야 합니다.</p>
<p><b>신설 센서</b>: 관악구 신설 하수 수위계 29개는 즉시 침수 라벨을 대체하지는 않지만, 2026년 우기 검증과 semi-supervised 관측 노드 확대에 중요합니다. 제공 시 관측 가능한 하수 노드가 13개에서 42개로 약 3배 확대되어, 관망 위 수위 전파와 국지 반응 검증의 공간 해상도가 개선됩니다.</p>
<p><b>부분 제공</b>: 모든 자료를 동시에 제공하기 어렵다면, 침수 정답 라벨과 신설 수위계 시계열을 우선 제공받는 것만으로도 학습 정답과 관측 노드 병목을 일부 해소할 수 있습니다. 이후 레이더 강우와 SWMM/DEM이 연결되면 조기예측 모델로 확장할 수 있습니다.</p>
<p><b>실행 계획</b>: 데이터 수신 후에는 먼저 좌표계·시간축·센서 ID를 기존 관망 그래프와 정합하고, 침수 라벨을 맨홀/도로 단위 학습 정답으로 변환합니다. 이후 레이더 강우와 집수면적을 결합해 맨홀별 유입량 입력을 만들고, 기존 GNN 하네스에서 persistence 대비 CSI·POD·FAR·리드타임 개선 여부를 재평가하겠습니다.</p>
<p><b>비식별 제공</b>: 120/119 신고 등 개인정보가 포함될 수 있는 자료는 개인 식별정보를 제거한 형태로도 충분히 활용 가능합니다. 모델 학습에는 신고자 정보가 필요하지 않으며, 일시·공간 위치·침수 유형 또는 침수심 정보만 있으면 정답 라벨로 변환할 수 있습니다.</p>
</div>
<p class="meta">근거 수치 재현: <code>python scripts/data_request_rationale.py</code></p>
</body></html>"""


def main():
    html = build()
    open(OUT, "w").write(html)
    n_fig = html.count("data:image/png;base64")
    miss = html.count("그림 없음")
    print(f"작성: {OUT} | {len(html):,} bytes | 임베드 그림 {n_fig} | 누락 {miss}")


if __name__ == "__main__":
    main()
