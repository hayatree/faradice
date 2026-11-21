#!/usr/bin/env python3
"""Summarize the L4 parameter study (design M2a §7).

Reads the `L4|variant|case|seed|value|std_err|exact|walks|steps|seconds`
lines printed by `crates/faradice-frw/tests/l4_study.rs` and prints, per
variant and case:

- the mean relative error over seeds and its standard error (bias),
- the relative variance per sample `(σ/C)² · samples`,
- the steps per sample and the wall time per million samples,
- the efficiency `1 / (variance · time)` relative to the `default` variant.

A last column pools the cases per variant (inverse-variance weighted bias).

Usage: summarize.py LOG [LOG ...]
"""

import math
import sys
from collections import defaultdict


def main() -> None:
    runs = defaultdict(list)
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("L4|"):
                    continue
                _, variant, case, _seed, value, se, exact, walks, steps, secs = (
                    line.strip().split("|")
                )
                runs[(variant, case)].append(
                    (float(value), float(se), float(exact), int(walks), float(steps), float(secs))
                )
    stats = {}
    for key, rs in runs.items():
        n = len(rs)
        errs = [(v / e - 1.0) for v, _, e, *_ in rs]
        ses = [s / e for _, s, e, *_ in rs]
        mean = sum(errs) / n
        se = math.sqrt(sum(s * s for s in ses)) / n
        var = sum((s * s) * w for s, (_, _, _, w, _, _) in zip(ses, rs)) / n
        steps = sum(r[4] for r in rs) / n
        t_per_m = sum(r[5] / r[3] * 1e6 for r in rs) / n
        stats[key] = (n, mean, se, var, steps, t_per_m)

    variants = sorted({k[0] for k in stats}, key=lambda v: (v != "default", v))
    cases = sorted({k[1] for k in stats})
    print("| variant | case | seeds | bias | σ_bias | z | rel. var/sample | steps/sample | s per M | efficiency |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        num = den = 0.0
        for c in cases:
            if (v, c) not in stats:
                continue
            n, mean, se, var, steps, t = stats[(v, c)]
            eff = ""
            if ("default", c) in stats:
                _, _, _, var0, _, t0 = stats[("default", c)]
                eff = f"{(var0 * t0) / (var * t):.2f}"
            print(
                f"| {v} | {c} | {n} | {100 * mean:+.4f} % | {100 * se:.4f} % | {mean / se:+.1f} "
                f"| {var:.3f} | {steps:.2f} | {t:.2f} | {eff} |"
            )
            num += mean / se**2
            den += 1.0 / se**2
        pooled, pse = num / den, math.sqrt(1.0 / den)
        print(f"| {v} | pooled | | {100 * pooled:+.4f} % | {100 * pse:.4f} % | {pooled / pse:+.1f} | | | | |")


if __name__ == "__main__":
    main()
