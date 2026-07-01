from __future__ import annotations

import os
from pathlib import Path

import requests


API_URL = (
    "https://apihub.kma.go.kr/api/typ01/"
    "cgi-bin/url/nph-aws2_min"
)


def download_aws_minute_data(
    auth_key: str,
    start_time: str,
    end_time: str,
    station: str = "0",
    save_path: str = "aws_minute_data.csv",
) -> Path:
    """
    기상청 API허브 AWS 매분자료를 내려받습니다.

    start_time, end_time 형식:
        YYYYMMDDHHMM

    station:
        "0"이면 전체 지점
        특정 지점번호를 넣으면 해당 지점만 조회
    """

    params = {
        "tm1": start_time,
        "tm2": end_time,
        "stn": station,
        "disp": "1",
        "help": "0",
        "authKey": auth_key,
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=120,
    )

    # 인증키 전체가 터미널에 노출되지 않도록 요청 주소 일부만 확인
    print("HTTP 상태 코드:", response.status_code)
    print("응답 형식:", response.headers.get("Content-Type"))

    if response.status_code != 200:
        print("오류 응답 앞부분:")
        print(response.text[:2000])
        response.raise_for_status()

    response_text = response.text.strip()

    # 서버가 HTTP 200으로 오류 메시지를 반환하는 경우 검사
    error_keywords = [
        "AUTH_KEY",
        "인증키",
        "ERROR",
        "Error",
        "INVALID",
        "Forbidden",
        "Not Found",
    ]

    if any(keyword in response_text for keyword in error_keywords):
        print("서버 응답:")
        print(response_text[:3000])
        raise RuntimeError("기상청 API가 오류 메시지를 반환했습니다.")

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)

    save_path_obj.write_text(
        response.text,
        encoding=response.encoding or "utf-8",
    )

    print("저장 완료:", save_path_obj.resolve())
    print(
        "파일 크기:",
        f"{save_path_obj.stat().st_size / 1024:,.2f} KB",
    )

    return save_path_obj


if __name__ == "__main__":
    auth_key = os.getenv("KMA_AUTH_KEY")

    if not auth_key:
        raise ValueError(
            "KMA_AUTH_KEY 환경변수에 인증키를 설정하세요."
        )

    download_aws_minute_data(
        auth_key=auth_key,
        start_time="202301010000",
        end_time="202512312359",
        station="0",
        save_path="data/aws_rain_2023_2025.csv",
    )