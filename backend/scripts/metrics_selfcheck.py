"""评测指标纯函数自检（合成数据，确定性，不碰外部服务）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.metrics import accuracy, mean, retrieval_case  # noqa: E402

PASS = 0
TOTAL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global PASS, TOTAL
    TOTAL += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {label} {detail}")
    PASS += 1 if ok else 0


def main() -> int:
    r = retrieval_case(["a", "b", "c"], {"a"}, 5)
    check("首位命中", r.hit and r.reciprocal_rank == 1.0 and r.recall == 1.0, f"→ {r}")

    r = retrieval_case(["x", "y", "b"], {"b"}, 5)
    check("第三位命中 RR=1/3", r.hit and r.reciprocal_rank == 0.3333, f"→ rr={r.reciprocal_rank}")

    r = retrieval_case(["x", "y"], {"b"}, 5)
    check("未命中记 0", not r.hit and r.reciprocal_rank == 0.0 and r.recall == 0.0, f"→ {r}")

    r = retrieval_case(["a", "b", "c", "d", "e"], {"a", "e"}, 5)
    check("召回率按相关集大小", r.recall == 1.0 and r.precision == 0.4, f"→ {r}")

    r = retrieval_case(["a"], set(), 5)
    check("无相关标注视为不可判定", not r.hit and r.recall == 0.0, f"→ {r}")

    r = retrieval_case(["a", "b", "c"], {"a"}, 2)
    check("只统计 top_k 内", r.precision == 0.5 and r.recall == 1.0, f"→ {r}")

    check("None 不计入分母", accuracy([True, False, None, True]) == 66.67,
          f"→ {accuracy([True, False, None, True])}")
    check("全 None 返回 None", accuracy([None, None]) is None)
    check("空列表返回 None", accuracy([]) is None and mean([]) is None)
    check("mean 忽略 None", mean([1, None, 3]) == 2.0, f"→ {mean([1, None, 3])}")

    print(f"\n{PASS}/{TOTAL} 通过")
    return 0 if PASS == TOTAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
