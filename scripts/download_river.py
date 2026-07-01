"""한강홍수통제소(hrfco) 하천 수위 수집 — 관악 도림천·안양천.

무료 서비스키 필요(즉시 발급): https://www.hrfco.go.kr/ → OpenAPI 신청.
.env에 HRFCO_AUTH_KEY=... 추가 후 실행:
    set -a; source .env; set +a
    python scripts/download_river.py 202406010000 202509302350

산출: data/river_level_10min.parquet  [obscd, obsnm, ts, wl]  (관악 인근 도림천·안양천)
용도: 조원로 5-6 도로침수(강우동반 0%·여름 80%) ↔ 하천 수위 급등 동조 = 하천역류 가설 검증.
"""
from __future__ import annotations
import os, sys, time, json, urllib.request
import pandas as pd

KEY = os.environ.get("HRFCO_AUTH_KEY", "")
BASE = "http://api.hrfco.go.kr"
TARGET_KEYWORDS = ["도림", "안양천", "신림", "시흥", "구로", "대림"]  # 관악·인근 도시하천


def _get(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def station_list() -> pd.DataFrame:
    """수위관측소 제원 → 관악 인근만 필터."""
    d = _get(f"{BASE}/{KEY}/waterlevel/info.json")
    df = pd.DataFrame(d.get("content", []))
    if df.empty:
        return df
    mask = df["obsnm"].astype(str).str.contains("|".join(TARGET_KEYWORDS), na=False)
    return df.loc[mask, ["wlobscd", "obsnm", "lon", "lat"]].rename(columns={"wlobscd": "obscd"})


def fetch_wl(obscd: str, start: str, end: str) -> pd.DataFrame:
    """10분 수위 시계열."""
    url = f"{BASE}/{KEY}/waterlevel/list/10M/{obscd}/{start}/{end}.json"
    rows = _get(url).get("content", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["obscd"] = obscd
    df["ts"] = pd.to_datetime(df["ymdhm"], format="%Y%m%d%H%M", errors="coerce")
    df["wl"] = pd.to_numeric(df.get("wl"), errors="coerce")
    return df[["obscd", "ts", "wl"]]


def main():
    if not KEY:
        sys.exit("HRFCO_AUTH_KEY 없음 — .env에 추가 후 재실행 (무료 발급: hrfco.go.kr OpenAPI)")
    start, end = (sys.argv[1], sys.argv[2]) if len(sys.argv) >= 3 else ("202406010000", "202509302350")
    st = station_list()
    print(f"관악 인근 하천 수위관측소 {len(st)}개:\n{st.to_string(index=False)}")
    out = []
    for _, s in st.iterrows():
        try:
            df = fetch_wl(s.obscd, start, end)
            df["obsnm"] = s.obsnm
            out.append(df)
            print(f"  {s.obsnm}({s.obscd}): {len(df)} bins")
            time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"  {s.obsnm} 실패: {e}")
    if out:
        res = pd.concat(out, ignore_index=True)
        os.makedirs("data", exist_ok=True)
        res.to_parquet("data/river_level_10min.parquet", index=False)
        print(f"저장: data/river_level_10min.parquet ({len(res)} rows)")


if __name__ == "__main__":
    main()
