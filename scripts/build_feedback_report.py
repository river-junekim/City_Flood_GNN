"""1차 보고서 피드백 반영 내용만 모은 HTML 생성.

피드백 7개(① 침수조건+출처 · ② 도로 오작동 근거 · ③ 조원로 리드타임 · ④ 해상도·강우장 근거
· ⑤ DEM 근거(SWMM 배제) · ⑥ 그림 기간 표기 · ⑦ 그림6 하천 출처)에 대한 반영 내용을
원본 보고서와 분리해 한 파일로 정리. 그림은 base64 임베드.
기준: 1차 보고서(2차 발견 전) · 도로 침수 = 도로수위계 ≥15cm(서울시).
실행:  python scripts/build_feedback_report.py
"""
from __future__ import annotations
import os, base64
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FD = "reports/figures_demo"
OUT = "reports/feedback_report.html"

CSS = """
body{font-family:'Malgun Gothic','Apple SD Gothic Neo',sans-serif;max-width:1000px;margin:24px auto;padding:0 18px;color:#222;line-height:1.7}
h1{border-bottom:3px solid #c0392b;padding-bottom:8px}
.meta{color:#666;font-size:13px}
.item{border:1px solid #ddd;border-left:4px solid #c0392b;border-radius:6px;padding:14px 18px;margin:18px 0;background:#fafafa}
.item h3{margin:0 0 6px}
.fb{background:#fff3cd;border:1px solid #ffe08a;border-radius:4px;padding:8px 12px;font-size:14px;margin-bottom:10px}
.done{display:inline-block;background:#27ae60;color:#fff;font-size:12px;padding:1px 8px;border-radius:10px;margin-left:6px}
img{max-width:100%;border:1px solid #ccc;border-radius:4px;margin-top:8px}
.cap{font-size:12px;color:#666;margin-top:4px}
ul{margin:6px 0}
"""


def embed(path, cap=""):
    if not os.path.exists(path):
        return f'<p class="cap">[그림 없음: {path}]</p>'
    b = base64.b64encode(open(path, "rb").read()).decode()
    c = f'<div class="cap">{cap}</div>' if cap else ""
    return f'<img src="data:image/png;base64,{b}">{c}'


ITEMS = [
    dict(no="①", title="침수 조건 설명 추가 (+ 출처)",
         fb="진행 경과/실패 사유 — 침수 조건 설명 추가 (어떤 출처 기반인지 명시)",
         body="""<p><b>침수 양성 = 도로수위계 침수심 ≥ 15cm</b> (임의 임계가 아니라 <b>서울시 침수 예보 발령 기준</b>).</p>
<ul>
<li><b>출처</b>: 서울시 (news.seoul.go.kr/env/archives/522983) — <i>"①시간당 강우량 55mm 초과 ②15분당 강우량 20mm 초과 <b>③도로수위계 기준 침수심 15cm 초과</b> 중 어느 하나라도 해당되면 자치구 단위 '침수 예보' 발령."</i></li>
<li>세 조건 중 <b>③ 도로수위계 수위 항(15cm)</b>을 침수 양성 정의로 채택. 강우 항은 별도 입력으로 사용.</li>
<li><b>하수 충전율(만관·위험수위)은 침수 기준이 아니라 별개 신호로 분리</b> (도로 표면 침수 ≠ 하수관 만관).</li>
</ul>""",
         fig=(f"{FD}/req_why_infeasible.png", "도로수위계 ≥15cm(서울시 기준) 적용 시 관악 도로 침수 양성 가용성. 모든 그림에 측정기간 표기(⑥).")),

    dict(no="②", title="도로 노면 오작동·미판정 근거 제시",
         fb="도로 노면 오작동 및 미판정에 대하여 오작동에 대한 근거 제시 필요",
         body="""<p>관악 도로센서별 침수 양성을 <b>≥6cm(참고) vs ≥15cm(서울시 기준)</b>로 비교하고, 오작동 의심 센서엔 정량 근거(겨울 집중·비영값 고착·고유값)를 부기.</p>
<ul>
<li><b>봉천동 911-14</b>: 겨울 100%·고착 71%·고유값 3 = 겨울 사각파형 신호 → 오작동 근거 명확.</li>
<li>≥15cm(서울시 기준) 적용 시 오작동·저수위 센서는 0으로 탈락, 조원로 5-6만 잔존.</li>
</ul>""",
         fig=(f"{FD}/req1_label_contamination.png", "관악 도로센서별 양성(≥6 vs ≥15cm) + 오작동 근거(겨울%·고착%·고유값).")),

    dict(no="③", title="강우-수위 동시반응(리드타임 0) — 조원로 기준 상세",
         fb="강우-수위 동시 반응으로 리드타임 0인 부분을 조원로 기준으로 자세한 설명 필요",
         body="""<p><b>조원로 5-6에서 강우-수위가 거의 동시 반응 → 리드타임 ≈ 0.</b></p>
<ul>
<li>조원로 일대 소유역은 작고(평균 ≈0.74km²)·불투수율이 높아 <b>도달시간(time of concentration)이 분 단위</b> → 비가 내리면 도로 수위가 거의 동시에 상승.</li>
<li>조원로의 강우계(점 강우)는 빗방울이 계기에 닿는 순간 이미 비가 온 것 → <b>수위를 앞설 수 없음(리드타임 0)</b>.</li>
<li>상류에서 다가오는 강우장을 먼저 보는 <b>격자 강우(레이더)</b>만이 조원로 수위를 시간적으로 앞서는 입력이 된다.</li>
</ul>""",
         fig=None),

    dict(no="④", title="해상도·강우장 데이터 요구조건 근거",
         fb="해상도 및 강우장 데이터 요구 조건에 대한 근거 제시 필요",
         body="""<ul>
<li><b>공간 ≤1km</b>: 관악 면적 ≈ 29.57km², 소유역 40개 → 평균 소유역 ≈ 0.74km². 1km 격자(1km²)는 소유역당 채 1셀이 안 돼 <b>분해 최소선이 1km</b>, 500m면 소유역당 ~3셀로 맨홀별 강우 차등 가능. (관악 전역 = 1km ~30셀 / 500m ~120셀)</li>
<li><b>시간 5~10분</b>: 도시 소유역(평균 0.74km²·불투수)의 도달시간이 분~십수 분 → 반응이 빨라 10분보다 성긴 자료는 첨두를 놓침.</li>
<li><b>상류 포함</b>: 리드타임 = 강우장의 상류→관악 이동시간. 이류속도 ~30~50km/h → 30분 리드타임엔 상류 <b>~15~25km</b> 강우장 필요.</li>
</ul>""",
         fig=None),

    dict(no="⑤", title="DEM 필요성 근거 (SWMM 현 단계 배제)",
         fb="SWMM은 초기 모델 복잡도 때문에 현재는 배제. DEM이 노드-엣지 연결에 정말 필요한지 구체적 근거로 필요성 어필",
         body="""<p><b>DEM은 노드-엣지(맨홀-관로) 연결용이 아니다</b> — 연결은 KICT GIS 관저고로 이미 확보·검증(차수 80%·공간정합 77%). DEM이 필요한 건 <b>노드 입력(맨홀별 유입)의 결손</b>을 채우기 위함:</p>
<ul>
<li><b>① 집수면적</b>: 맨홀 유입 = 강우 × 집수면적(Q=C·i·A). 강우만으론 부피가 안 나오고 면적이 곱해져야 유입량. 공식 소유역 40개는 맨홀 12,184개 대비 너무 coarse → 맨홀별 정밀 집수면적은 <b>DEM 흐름누적·저지대 분석으로만</b> 도출.</li>
<li><b>② 표면류 경로</b>: 관악 침수=표면류형 → 관망 밖 지표에서 물이 모임. DEM 저지대·흐름방향이 "어느 맨홀로 물이 모이나" 제공.</li>
<li><b>③ 노드 변별력</b>: 집수면적 없으면 강우가 모든 맨홀에 균일 입력 → 노드 차이·상관신호 안 뜸.</li>
<li><b>SWMM은 현 단계 배제</b> (모델 복잡도).</li>
</ul>""",
         fig=(f"{FD}/req2_dem_rationale.png", "조원로 침수 시각: 도로수위는 높으나 인근 하수·하천은 정상 → 표면류 = DEM(지형) 필요 근거.")),

    dict(no="⑥", title="각 시각화에 측정 년도·시기 표기",
         fb="각 시각화 자료에 측정 년도와 시기 동시 표기",
         body="""<p>요청서용 그림 <b>6개 전부</b>에 측정기간을 제목/캡션에 표기 완료: <b>[2024-06~2025-09 · 여름 우기(6~9월)]</b>.</p>
<ul><li>req_why_infeasible · req1_label_contamination · req_river_useless · req2_dem_rationale · req_sewer_reach (+ req_leadtime).</li></ul>""",
         fig=(f"{FD}/req_sewer_reach.png", "관악 하수 충전율 도달 분포 — 제목에 측정기간 표기(⑥ 적용 예시).")),

    dict(no="⑦", title="그림6(하천) 보충 + 하천 출처 표기",
         fb="그림 6도 침수 조건 설명에 따른 보충 설명을 하단 문구에 표기, 하천 데이터 출처도 표기",
         body="""<p>하천(하천역류 기각) 그림 하단에 <b>침수 조건(도로수위계 ≥15cm·서울시)</b>과 <b>하천 데이터 출처</b>를 명기.</p>
<ul><li><b>하천 출처</b>: 한강홍수통제소(HRFCO) 도림천 신림5교, 2024-06~2025-09.</li>
<li>침수 시각의 도림천 수위가 대부분 평상시(p95 미만)임을 표기 → 하천역류가 침수 원인이 아님(표면류).</li></ul>""",
         fig=(f"{FD}/req_river_useless.png", "하천(도림천) 수위 — 하단에 침수 조건 + 하천 출처(한강홍수통제소 신림5교) 표기.")),
]


def build():
    secs = ""
    for it in ITEMS:
        figs = embed(it["fig"][0], it["fig"][1]) if it["fig"] else ""
        secs += (f'<div class="item"><h3>{it["no"]} {it["title"]}<span class="done">반영 완료</span></h3>'
                 f'<div class="fb"><b>피드백</b>: {it["fb"]}</div>{it["body"]}{figs}</div>')
    return f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>1차 보고서 피드백 반영 내용</title><style>{CSS}</style></head><body>
<h1>1차 보고서 — 피드백 반영 내용 (별도 정리)</h1>
<p class="meta">기준: 관악구 · 도로 침수 = 도로수위계 ≥15cm(서울시) · 작성 2026-06-30 · 원본 보고서와 분리</p>
<p>아래는 1차 보고서에 받은 피드백 7건과 그 반영 내용만 모은 것입니다. (2차 발견 사항은 별도 문서로 분리)</p>
{secs}
</body></html>"""


if __name__ == "__main__":
    open(OUT, "w").write(build())
    print("작성:", OUT)
