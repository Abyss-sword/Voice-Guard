"""一键验收自检：检查权重/注册库/交付物/文档/服务/测试，打印清单。

用法：
    python tools/acceptance_check.py                  # 全量检查（服务未起时提示可选）
    python tools/acceptance_check.py --no-weights     # CI/无权重环境
    python tools/acceptance_check.py --tests          # 额外跑 39 项测试（较慢）
    python tools/acceptance_check.py --services       # 探测 7860/8020 服务

退出码：必需项缺失返回 1；可选（服务/测试/权重）失败只在清单中标注。
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REQUIRED_DOCS = [
    "README.md",
    "docs/04_报告.md",
    "docs/05_验收PPT大纲.md",
    "docs/08_验收自查.md",
    "docs/13_课程要求与交付清单.md",
    "deliverables/README.md",
    "deliverables/提交信息模板.md",
    "deliverables/Voice_Guard_课程报告.pdf",
    "deliverables/Voice_Guard_9分钟汇报.pptx",
]
REQUIRED_CODE = [
    "vguard/attack.py", "vguard/masking.py", "vguard/features.py",
    "vguard/pipeline.py", "frontend/app.py", "platform/service/adapter.py",
    "scripts/reproduce.py", "requirements.txt",
]
WEIGHT_DIRS = ["checkpoints/adain-vc-official", "checkpoints/ecapa-local"]
REGISTRIES = ["outputs/registry.npz", "outputs/registry_vc.npz"]


def ok(p: str) -> bool:
    return os.path.exists(os.path.join(ROOT, p))


def http_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-weights", action="store_true")
    ap.add_argument("--no-services", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--tests", action="store_true", help="运行 tests/ 全部测试")
    args = ap.parse_args()

    rows = []  # (类别, 项, 结论, 是否必需)

    for p in REQUIRED_DOCS:
        rows.append(("文档/交付物", p, "OK" if ok(p) else "缺失", True))
    for p in REQUIRED_CODE:
        rows.append(("代码", p, "OK" if ok(p) else "缺失", True))

    if not args.no_weights:
        for d in WEIGHT_DIRS:
            rows.append(("权重", d, "OK" if ok(d) else "缺失（可用云盘/Release 下载）", False))
        for r in REGISTRIES:
            rows.append(("注册库", r, "OK" if ok(r) else "缺失（build_registry.py 生成）", False))

    ev = glob.glob(os.path.join(ROOT, "docs", "evidence", "*", "README.md"))
    rows.append(("证据", f"docs/evidence/*/README.md × {len(ev)}", "OK" if ev else "缺失", False))

    if not args.no_services:
        rows.append(("服务", "Gradio 7860", "200" if http_ok("http://127.0.0.1:7860/") else "未启动（可选）", False))
        rows.append(("服务", "平台适配 8020", "health OK" if http_ok("http://127.0.0.1:8020/health") else "未启动（可选）", False))

    if args.tests:
        direct = subprocess.run([sys.executable, "tests/test_features.py"],
                                cwd=ROOT, capture_output=True, text=True)
        rows.append(("测试", "test_features", "PASS" if direct.returncode == 0 else "FAIL", False))
        unitt = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                                "-p", "test_*.py"], cwd=ROOT, capture_output=True, text=True)
        tail = (unitt.stderr or "").strip().splitlines()[-1:] or [""]
        rows.append(("测试", "unittest 全量", tail[0] if unitt.returncode == 0 else "FAIL", False))

    print(f"{'类别':<8}{'检查项':<46}{'结论':<26}必需")
    print("-" * 92)
    bad = 0
    for cat, item, res, need in rows:
        if need and res != "OK":
            bad += 1
        print(f"{cat:<8}{item:<46}{res:<26}{'是' if need else '否'}")
    print("-" * 92)
    print(f"必需项缺失：{bad} 个" if bad else "必需项全部通过 ✅")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
