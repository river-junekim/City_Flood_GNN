"""matplotlib 한글 폰트 설정 (NanumGothic). 노트북/스크립트에서 호출.

사용:
    import sys; sys.path.insert(0, "scripts")
    from krfont import set_korean
    set_korean()
"""
from __future__ import annotations
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_CANDIDATES = [
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def set_korean(size: int = 11) -> str:
    """한글 폰트를 matplotlib 전역에 적용하고 폰트명을 반환. 마이너스 깨짐도 방지."""
    name = None
    for path in _CANDIDATES:
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            break
        except Exception:
            continue
    if name is None:
        name = "DejaVu Sans"  # 폴백
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False  # 유니코드 마이너스(−) → ASCII '-'로
    plt.rcParams["font.size"] = size
    return name


if __name__ == "__main__":
    import numpy as np
    nm = set_korean()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(-3, 3, 100)
    ax.plot(x, x ** 2 - 2, label="수위 변화(-2 오프셋)")
    ax.set_title("한글 폰트 테스트 — 하수관로 위험수위·만관·침수")
    ax.set_xlabel("강우 강도 (mm/h)"); ax.set_ylabel("충전율(fill_rate)")
    ax.legend()
    fig.savefig("reports/figures_gnn/_krfont_test.png", dpi=110, bbox_inches="tight")
    print("적용 폰트:", nm, "| 저장: reports/figures_gnn/_krfont_test.png")
