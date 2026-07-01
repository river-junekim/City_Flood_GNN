"""아침 현황 브리핑 — 도심 침수 예측 프로젝트.

실제 산출물에서 수치를 읽어 ①현재 현황 ②핵심 결론 ③확보/필요 데이터를 출력.
실행(루트에서):  python scripts/morning_briefing.py
"""
from __future__ import annotations
import os
from datetime import date
import pandas as pd

ROOT = "/home/namjun/city_flood"
EB = os.path.join(ROOT, "dataset/processed/eda_based")
os.chdir(ROOT)


def _try(fn, default="-"):
    try:
        return fn()
    except Exception:
        return default


def head(t):
    print("\n" + "=" * 70 + f"\n  {t}\n" + "=" * 70)


def line(label, value):
    print(f"  {label:<34} {value}")


def sec1_summary():
    head("① 한 줄 요약 / 현재 위치")
    print("  원 목표: 관악구 배수지역 침수예측 GNN (데이터 부족→서울 확장).")
    print("  현재: 반복 확인된 병목은 모델이 아니라 '데이터'. 현 센서로는")
    print("        침수 supervised·CSI 우위 모델 불가 → 데이터 확보(B) 단계.")


def sec2_have():
    head("② 확보 데이터 현황 (HAVE)")
    sn = _try(lambda: len(pd.read_parquet(f"{EB}/../cleaned/sewer_node.parquet")))
    rn = _try(lambda: len(pd.read_parquet(f"{EB}/../cleaned/road_node.parquet")))
    line("하수관로 수위센서", f"{sn}개 (수위 78% A등급, 연속·깨끗)")
    line("도로노면 수위센서", f"{rn}개 (라벨 93% 아티팩트, 진짜상습 ~10)")
    line("AWS 강우", _try(lambda: f"{pd.read_parquet(f'{ROOT}/data/aws_seoul_rain_10min.parquet').stn.nunique()}지점 2024-06~2025-09"))
    line("하천 수위(도림천)", _try(lambda: f"{pd.read_parquet(f'{ROOT}/data/river_level_10min.parquet').obsnm.nunique()}개소(신림5교·신대방1교) ✅확보·검증완료"))
    line("관악 GIS 관망", "하수관로 17,286·맨홀 12,184·소배수구역 40·물받이 27,447 ✅")
    line("레이더 격자 shp", "1km 격자 보유(강수값 미수신) ⏳")


def sec3_conclusions():
    head("③ 오늘까지의 핵심 결론")
    msgs = [
        "도로 침수 라벨 ~93% 아티팩트 → supervised 타깃 부적합.",
        "관악 침수 ≠ 하수 만관(관악 하수 안 참, 침수시 fill 0.13).",
        "관악 침수 ≠ 도림천 하천역류(침수 90%가 하천 평상시 수위) — 오늘 기각.",
        "→ 소거법: 관악 침수 = 국지 호우 표면류(pluvial). 강우강도가 결정변수.",
        "위험수위 재정의(만관 fill≥1 → fill≥0.8): 서울 도달센서 194→329(1.7배).",
    ]
    # 실제 수치로 보강
    da = _try(lambda: pd.read_parquet(f"{EB}/sewer_danger_audit.parquet"))
    if isinstance(da, pd.DataFrame):
        msgs.append(f"위험수위 강우성 확정 {int((da.판정=='진짜_강우성위험').sum())}센서 (병목=AWS강우 커버리지+stuck).")
    csi = _try(lambda: pd.read_parquet(f"{EB}/seoul_danger_csi.parquet"))
    if isinstance(csi, pd.DataFrame):
        p = csi[csi.model.str.contains("persist")].set_index("horizon").CSI
        g = csi[~csi.model.str.contains("persist")].set_index("horizon").CSI
        msgs.append(f"A-1 위험수위 예측 CSI: persistence {p.get(10):.2f}/{p.get(30):.2f}/{p.get(60):.2f} vs 모델 {g.get(10):.2f}/{g.get(30):.2f}/{g.get(60):.2f} → 못 넘음(리드타임 부재).")
    on = _try(lambda: pd.read_parquet(f"{EB}/onset_csi.parquet"))
    if isinstance(on, pd.DataFrame):
        gb = on[on.model.str.contains("GBM")].CSI
        msgs.append(f"onset(진입) 조기예측: persistence 구조적 0, 모델 {gb.min():.2f}~{gb.max():.2f}(약신호·실무미달) → 레이더 필요.")
    for i, m in enumerate(msgs, 1):
        print(f"  {i}. {m}")


def sec4_need():
    head("④ 필요 데이터 (레이더 외 포함) — 우선순위순")
    rows = [
        ("1", "레이더 강우(HSR/HFC 격자)", "기상청 포털/API허브 (격자 shp 보유)",
         "⏳ 내일(용량리셋)", "onset 리드타임 + 관악 표면류 강우강도"),
        ("2", "침수 정답 라벨(침수흔적도·120/119)", "국가공간정보포털·서울 열린데이터·소방청",
         "❌ 미확보", "실제 침수 supervised 타깃(현재 1지점)"),
        ("3", "DEM 수치표고", "국토정보플랫폼(빗물받이는 03_GIS 보유)",
         "❌ 미확보", "저지대 표면류 위험지도·도로센서 그래프"),
        ("4", "다년 침수이력(2020·2022 신림)", "재해연보·침수흔적도 과거분",
         "❌ 미확보", "진짜 다년 양성표본(현 센서 2024~만)"),
        ("-", "하천 수위(도림천·안양천)", "한강홍수통제소 hrfco",
         "✅ 확보·검증완료", "관악 원인 소거(하천역류 기각)"),
    ]
    print(f"  {'#':<2}{'데이터':<26}{'상태':<14}{'해소/효과'}")
    print("  " + "-" * 66)
    for n, d, src, st, eff in rows:
        print(f"  {n:<2}{d:<26}{st:<14}{eff}")
        print(f"     └ 출처: {src}")


def sec5_next():
    head("⑤ 내일 액션")
    print("  1) 레이더 강우 수신 → 격자↔센서 매핑 → onset CSI 재측정(레이더 vs persistence).")
    print("  2) 동시에 관악 표면류: 레이더 국지 강우강도 ↔ 조원로 침수 동조 확인.")
    print("  3) 침수흔적도 입수 경로 타진(다년 라벨 일괄 해결 가능성).")
    print("  ※ 보류: 옛 figure 6개 한글화(노트북 첫셀 os.chdir 추가 후 재실행).")
    print("  ※ 키: .env의 KMA_AUTH_KEY(레이더)·HRFCO_AUTH_KEY(하천) 사용. 절대 노출 금지.")


def main():
    print("\n" + "#" * 70)
    print(f"#  도심 침수 예측 — 아침 현황 브리핑   ({date.today()})")
    print("#" * 70)
    sec1_summary()
    sec2_have()
    sec3_conclusions()
    sec4_need()
    sec5_next()
    print("\n  상세: reports/progress_report.html(§1~18) · docs/BRAINSTORM_2026-06-24.md\n")


if __name__ == "__main__":
    main()
