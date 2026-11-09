"""C16 云端压测：并发调用平台适配服务 /process，测延迟与吞吐。

用法：
    python tools/load_test.py --url http://127.0.0.1:8020/process \
        --audio platform/samples/group_xx_voice_guard/input.wav \
        --requests 12 --concurrency 3 --eps 0.05 --fast
输出：状态码分布、延迟 p50/p90/p99、吞吐（请求/分钟）、错误明细。
注意：适配服务默认「GPU 忙时返回 429」，压测会大量命中 429，这本身是治理行为的验证。
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import statistics
import time

import requests


def one(url: str, audio_path: str, eps: float, fast: bool):
    t0 = time.perf_counter()
    try:
        with open(audio_path, "rb") as f:
            files = {"defend_audio": (audio_path.split("/")[-1], f, "audio/wav")}
            data = {"method_id": "group_xx_voice_guard",
                    "parameters": json.dumps({"eps": eps, "fast_mode": fast})}
            r = requests.post(url, files=files, data=data, timeout=600)
        dt = time.perf_counter() - t0
        return r.status_code, dt, None
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t0, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8020/process")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, args.url, args.audio, args.eps, args.fast)
                for _ in range(args.requests)]
        results = [f.result() for f in futs]
    total = time.perf_counter() - t0

    codes = {}
    lat = []
    errs = []
    for code, dt, err in results:
        codes[code] = codes.get(code, 0) + 1
        lat.append(dt * 1000)
        if err:
            errs.append(err)
    lat.sort()

    def pct(p):
        return lat[min(len(lat) - 1, int(len(lat) * p))] if lat else 0

    print(f"requests={len(results)} concurrency={args.concurrency} total={total:.1f}s "
          f"throughput={len(results) / total * 60:.1f} req/min")
    print(f"status codes: {codes}")
    print(f"latency ms: p50={pct(0.5):.0f} p90={pct(0.9):.0f} p99={pct(0.99):.0f} "
          f"max={lat[-1]:.0f}" if lat else "")
    if errs:
        print("errors:", errs[:5])


if __name__ == "__main__":
    main()
