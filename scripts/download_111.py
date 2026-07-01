from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests


API_URL = (
    "https://apihub.kma.go.kr/api/typ01/"
    "cgi-bin/url/nph-aws2_min"
)


def download_all_stations_by_10_minutes(
    auth_key: str,
    start_datetime: datetime,
    end_datetime: datetime,
    output_dir: str = "data/aws_minute",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    current = start_datetime

    while current < end_datetime:
        chunk_end = min(
            current + timedelta(minutes=10),
            end_datetime,
        )

        tm1 = current.strftime("%Y%m%d%H%M")
        tm2 = chunk_end.strftime("%Y%m%d%H%M")

        save_path = output_path / f"aws_{tm1}_{tm2}.csv"

        if save_path.exists() and save_path.stat().st_size > 0:
            print(f"이미 존재하여 건너뜀: {save_path.name}")
            current = chunk_end
            continue

        params = {
            "tm1": tm1,
            "tm2": tm2,
            "stn": "425",
            "disp": "1",
            "help": "2",
            "authKey": auth_key,
        }

        print(f"조회 중: {tm1} ~ {tm2}")

        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=120,
            )

            response.raise_for_status()

            text = response.text.strip()

            if not text:
                print("  응답이 비어 있습니다.")
                current = chunk_end
                continue

            if (
                "인증키" in text
                or "AUTH_KEY" in text
                or "ERROR" in text.upper()
            ):
                print("  API 오류 응답:")
                print(text[:1000])
                raise RuntimeError("API 호출 오류")

            save_path.write_text(
                response.text,
                encoding=response.encoding or "utf-8",
            )

            print(
                f"  저장 완료: {save_path.name} "
                f"({save_path.stat().st_size / 1024:,.1f} KB)"
            )

        except requests.RequestException as error:
            print(f"  요청 실패: {error}")

        current = chunk_end

        # 서버 부하와 호출 제한 방지
        time.sleep(0.3)


if __name__ == "__main__":
    auth_key = os.getenv("KMA_AUTH_KEY")

    if not auth_key:
        raise ValueError(
            "KMA_AUTH_KEY 환경변수를 설정하세요."
        )

    download_all_stations_by_10_minutes(
        auth_key=auth_key,
        start_datetime=datetime(2022, 1, 1, 0, 0),
        end_datetime=datetime(2022, 12, 31, 23, 59),
        output_dir="data/aws_minute/2022",
    )