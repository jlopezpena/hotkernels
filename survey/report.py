"""Turn a survey result file into a Markdown table.

    python3 survey/report.py survey/results/apple-m1-pro.json > survey/results/apple-m1-pro.md

Rows are ranked by "large-n gap": the geometric mean of NumPy time / naive C time
over sizes of 65,536 and up, where per-call overhead no longer matters.
"""

import json
import math
import sys
from collections import defaultdict

SHOW = [16, 4096, 65536, 1048576, 16777216]
LARGE = 65536


def ratio(x: float) -> str:
    return f"{x:.0f}x" if x >= 100 else f"{x:.1f}x" if x >= 10 else f"{x:.2f}x"


def simd(loops: list[str]) -> str:
    return ", ".join("w" + l.split("width ")[1] for l in loops) if loops else "no"


def main(path: str) -> None:
    data = json.load(open(path))
    meta = data["meta"]
    by_key = defaultdict(dict)
    for r in data["rows"]:
        by_key[r["key"]][r["n"]] = r

    ranked = []
    for key, rows in by_key.items():
        large = [r["numpy_over_c"] for n, r in rows.items() if n >= LARGE]
        ranked.append((math.exp(sum(map(math.log, large)) / len(large)), key))
    ranked.sort(reverse=True)

    print(f"# Survey results: {meta['cpu']}\n")
    print(f"- NumPy {meta['numpy']}, Python {meta['python']}, {meta['os']}")
    print(f"- Naive C: {meta['clang']}, `{' '.join(meta['cflags'])}`")
    print(f"- Reference loop drift over the run: {meta['reference_drift']:.1%}"
          f"{' (noisy: above 3%)' if meta['reference_drift'] > 0.03 else ''}")
    print(f"- Git revision: `{meta['git'][:7]}`\n")
    print("Each ratio is NumPy time divided by naive C time. Above 1 means NumPy is slower than the naive loop.")
    print("`C SIMD` gives the vector width clang's loop vectoriser used for the naive kernel, or `no`. `alloc` marks statements where NumPy allocates")
    print("its result; the C side then mallocs and frees its output on every call as well.\n")
    head = ["Candidate", "NumPy statement", "C SIMD", "alloc", "large-n gap"]
    head += [f"n={n:,}" for n in SHOW] + ["C ns/elem at 1M", "NumPy ns/elem at 1M", "NumPy at n=1"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for gap, key in ranked:
        rows = by_key[key]
        any_row = next(iter(rows.values()))
        cells = [key, f"`{any_row['stmt']}`", simd(meta["c_vectorised"][any_row["cfn"]]),
                 "yes" if any_row["alloc"] else "", f"**{ratio(gap)}**"]
        cells += [ratio(rows[n]["numpy_over_c"]) if n in rows else "" for n in SHOW]
        m = rows.get(1048576)
        cells += [f"{m['c_ns'] / m['work']:.2f}", f"{m['numpy_ns'] / m['work']:.2f}"] if m else ["", ""]
        cells.append(f"{rows[1]['numpy_ns']:.0f} ns" if 1 in rows else "")
        print("| " + " | ".join(cells) + " |")

    bad = sorted({r["key"] for r in data["rows"] if not r["naive_matches_numpy"]})
    if bad:
        print("\n**Naive C disagreed with NumPy for:** " + ", ".join(bad))


if __name__ == "__main__":
    main(sys.argv[1])
