"""
AWS 서울 창별 1분 데이터(data/aws_seoul/win/*.parquet) → 10분 강우 피처.

출력: data/aws_seoul_rain_10min.parquet
  컬럼: stn, ts10, rn15m(15분누적), rn60m(1h누적,mm), rn12h(12h누적), is_rain60
  ※ RNDAY(일강수)는 09시 기준 리셋이라 차분 비신뢰 → 미사용. RN15m/RN60m은 리셋없는 이동누적.
"""
from __future__ import annotations
import glob
import pandas as pd
import numpy as np


def build(win_glob="data/aws_seoul/win/*.parquet",
          out="data/aws_seoul_rain_10min.parquet") -> pd.DataFrame:
    files = sorted(glob.glob(win_glob))
    if not files:
        raise SystemExit("창 파일 없음: " + win_glob)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df['tm'] = pd.to_datetime(df['tm'], errors='coerce')   # 빈 창 파일이 tm을 object로 만들 수 있음
    df = df.dropna(subset=['tm'])
    df = df.drop_duplicates(['stn', 'tm']).sort_values(['stn', 'tm'])

    # RN15m/RN60m = 리셋 없는 이동누적(15분/1h 강수). RNDAY는 09시 기준 리셋이라 차분 비신뢰 → 미사용.
    # 10분 격자에서 각 누적의 마지막 값을 취함(해당 10분 끝 시점의 최근 누적).
    df['ts10'] = df['tm'].dt.floor('10min')
    agg = df.groupby(['stn', 'ts10']).agg(
        rn15m=('RN15m', 'last'),
        rn60m=('RN60m', 'last'),
        rn12h=('RN12H', 'last'),
        n=('tm', 'size'),
    ).reset_index()
    agg['is_rain60'] = (agg['rn60m'] > 0).astype('int8')
    agg.to_parquet(out, index=False)
    return agg


if __name__ == "__main__":
    a = build()
    print("저장: data/aws_seoul_rain_10min.parquet")
    print("행:", len(a), "| 지점:", a.stn.nunique(),
          "| 기간:", a.ts10.min(), "~", a.ts10.max())
    print("rn60m>0 비율:", round((a.rn60m > 0).mean() * 100, 2), "%",
          "| 최대 RN60m(1h):", round(a.rn60m.max(), 1), "mm")
