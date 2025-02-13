from subprocess import check_output
from pathlib import Path
from random import shuffle
import json

def run_single(cmd: Path, n: int, runtime: str) -> float:
    if runtime == "cpu" and n > 300:
        return 1000.0

    cmd = [cmd, "--size", str(n)]
    if runtime == "cpu":
        cmd.append("--cpu")
    if runtime == "cuda":
        cmd.append("--cuda")

    o = check_output(cmd, encoding="utf-8", universal_newlines=True)

    for l in o.splitlines():
        if not l.startswith("Average runtime"):
            continue

        *_, run_time = l.split()
        return float(run_time.strip())


def main():
    exe = Path().cwd() / "bin_x64" / "Release" / "vk_mini_fusion_exe_app"

    all_cmds = [
        # (n, c) for n  in range(790, 810, 1) for _ in range(1) for c in ("cuda", "shader", "cpu")
        (n, c) for n  in range(25, 951, 25) for _ in range(1) for c in ("cuda", "shader", "cpu")
    ]
    # shuffle(all_cmds)

    results = {}

    final_results = []

    for n, c in all_cmds:
        print(c, n, flush=True, end=" ")
        res = run_single(exe, n, c)
        print(res, flush=True)

        key = (n, c)

        if key not in results:
            results[key] = []
        results[key].append(res)

        final_results.append(
                {
                    "type": c,
                    "n": n,
                    "res": res
                }
        )

    with Path("vol_run_times.json").open("w") as f:
        json.dump(final_results, f, indent=2)


if __name__ == '__main__':
    main()
