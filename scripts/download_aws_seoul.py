"""
기상청 API허브 AWS 매분자료 → 서울 지점만 받아 저장 (보정본).

핵심 제약(실측):
- stn 파라미터는 단일지점 또는 "0"(전체)만 지원(콤마 리스트 미지원).
- 응답 행수 상한 ~132,673행(전지점 기준 ~181분). 초과 요청 시 뒤쪽만 잘려 옴.
  → 안전하게 **180분(3시간) 창**으로 끊어 요청하면 완전한 데이터를 받음.
- 이력은 2023년~현재까지 가용.

전략: stn=0로 180분씩 받아 → 서울 44개 지점만 필터 → 강우 핵심컬럼만 창별 parquet로 저장(재개 가능).
저장이 서울만이라 용량은 작음(전송은 전지점이라 큼).

사용:
    set -a; source .env; set +a
    python download_aws_seoul.py 202406010000 202410010000   # 시작 끝 (YYYYMMDDHHMM)
    # 인자 없으면 아래 DEFAULT 사용
"""
from __future__ import annotations
import os, sys, time
from datetime import datetime, timedelta
from pathlib import Path
import requests
import pandas as pd
import numpy as np

URL = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-aws2_min"
WINDOW_MIN = 180                     # 캡 안전 창 크기
OUT_DIR = Path("data/aws_seoul/win") # 창별 저장(재개용)
COORDS = "data/aws_station_coords.parquet"
DEFAULT_START = "202406010000"       # 기본: 2024 여름~ (침수 시즌)
DEFAULT_END   = "202410010000"

# 헤더 컬럼 순서(고정)
COLS = ['tm','stn','WD1','WS1','WDS','WSS','WD10','WS10','TA','RE',
        'RN15m','RN60m','RN12H','RNDAY','HM','PA','PS','TD']
KEEP = ['tm','stn','RN15m','RN60m','RN12H','RNDAY','TA','WS10','HM']


def seoul_stations() -> set[int]:
    s = pd.read_parquet(COORDS)
    s = s[(s.lat.between(37.41, 37.70)) & (s.lon.between(126.76, 127.18))]
    return set(int(x) for x in s.stn)


def parse(text: str, keep_stn: set[int]) -> pd.DataFrame:
    rows = []
    for ln in text.splitlines():
        if not ln or ln.startswith('#'):
            continue
        t = ln.rstrip(',=').split(',')
        if len(t) < 18:
            continue
        try:
            stn = int(t[1])
        except ValueError:
            continue
        if stn not in keep_stn:
            continue
        rows.append(t[:18])
    if not rows:
        return pd.DataFrame(columns=KEEP)
    df = pd.DataFrame(rows, columns=COLS)
    df['tm'] = pd.to_datetime(df['tm'], format='%Y%m%d%H%M', errors='coerce')
    df['stn'] = df['stn'].astype(int)
    for c in ['RN15m','RN60m','RN12H','RNDAY','TA','WS10','HM']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df.loc[df[c] <= -50, c] = np.nan          # 결측(-99) 처리
    # 강우 음수는 결측
    for c in ['RN15m','RN60m','RN12H','RNDAY']:
        df.loc[df[c] < 0, c] = np.nan
    return df[KEEP]


def fetch_window(session, key, tm1, tm2, keep_stn, retries=3):
    for k in range(retries):
        try:
            r = session.get(URL, params={"tm1": tm1, "tm2": tm2, "stn": "0",
                            "disp": "1", "help": "0", "authKey": key}, timeout=300)
            if r.status_code != 200 or "유효하지" in r.text or "활용신청" in r.text:
                print(f"  오류({r.status_code}) {r.text[:80]!r}"); time.sleep(3); continue
            return parse(r.text, keep_stn)
        except Exception as e:
            print(f"  예외 재시도 {k+1}: {e}"); time.sleep(5)
    return None


RAIN_FEATURES = "dataset/processed/eda_based/rain_features_10min.parquet"

def rainy_window_starts(t0, t1):
    """공공 강우로 '비 온 3h창' 시작시각 집합을 만든다. 공공 강우 커버리지 밖 기간(이전·이후 모두)은
    필터 못 하므로 전부 받도록 처리(cover_start~cover_end 밖은 무조건 fetch)."""
    r = pd.read_parquet(RAIN_FEATURES, columns=["timestamp", "rainfall_mm", "rain_6h_sum"])
    cover_start, cover_end = r["timestamp"].min(), r["timestamp"].max()
    r = r[(r["timestamp"] >= t0) & (r["timestamp"] < t1)]
    rr = r[(r["rainfall_mm"] > 0) | (r["rain_6h_sum"] > 0)]
    starts = set(pd.to_datetime(rr["timestamp"]).dt.floor(f"{WINDOW_MIN}min"))
    return starts, cover_start, cover_end


def main():
    key = os.getenv("KMA_AUTH_KEY")
    if not key:
        raise SystemExit("KMA_AUTH_KEY 미설정: set -a; source .env; set +a")
    start = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_START
    end   = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_END
    t0 = datetime.strptime(start, "%Y%m%d%H%M")
    t1 = datetime.strptime(end,   "%Y%m%d%H%M")
    keep_stn = seoul_stations()
    rainy, cover_start, cover_end = rainy_window_starts(t0, t1)   # 강우창 타겟
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    print(f"서울 {len(keep_stn)}지점 | {start}~{end} | {WINDOW_MIN}분 창 | 강우창 {len(rainy)}개(커버 {cover_start:%Y-%m-%d}~{cover_end:%Y-%m-%d})")
    cur = t0; n_done = n_skip = n_dry = 0; tstart = time.time()
    while cur < t1:
        win_end = min(cur + timedelta(minutes=WINDOW_MIN), t1)
        tm1 = cur.strftime("%Y%m%d%H%M")
        # 창 종료는 포함이므로 1분 빼서 겹침 방지
        tm2 = (win_end - timedelta(minutes=1)).strftime("%Y%m%d%H%M")
        fp = OUT_DIR / f"aws_{tm1}.parquet"
        if fp.exists():
            n_skip += 1; cur = win_end; continue
        # 강우 커버리지 안에서 비 안 온 창은 건너뜀(마른 시간 = 강우 0, 받을 필요 없음). 커버리지 밖(이전·이후)은 전부 받음
        in_coverage = cover_start <= pd.Timestamp(cur) <= cover_end
        if in_coverage and pd.Timestamp(cur) not in rainy:
            n_dry += 1; cur = win_end; continue
        df = fetch_window(session, key, tm1, tm2, keep_stn)
        if df is None:
            print(f"  [{tm1}] 실패 — 중단(나중에 재실행하면 이어받음)"); break
        df.to_parquet(fp, index=False)
        n_done += 1
        if n_done % 20 == 0:
            el = time.time() - tstart
            print(f"  진행 {tm1} | 받음 {n_done} 스킵 {n_skip} | {el/ max(n_done,1):.1f}s/창")
        cur = win_end
    print(f"완료: 받음 {n_done} 스킵 {n_skip} 마른창생략 {n_dry} | 저장 {OUT_DIR}")
    print("→ 합치기: python -c \"import pandas as pd,glob; pd.concat([pd.read_parquet(f) for f in glob.glob('data/aws_seoul/win/*.parquet')]).to_parquet('data/aws_seoul_rain.parquet')\"")


if __name__ == "__main__":
    main()
