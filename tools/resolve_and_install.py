"""curl 抓取 wheel + --no-index 本地解析安装：完全绕开 pip 网络层。

流程：
  1. 对缺失包逐个用 curl 抓 pypi JSON/华为源页面，选最新兼容 wheel 下载
  2. `pip install --dry-run --no-index <已下载 wheels>` 本地解析
  3. 解析报出的缺包加入待抓列表，循环，直到解析成功
  4. `pip install --no-index` 落地安装

用法：
    python tools/resolve_and_install.py numpy scipy librosa soundfile pyyaml tqdm \
        matplotlib openai-whisper speechbrain gradio fastapi uvicorn python-multipart \
        gdown python-docx python-pptx openpyxl pypdf
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_wheels import WHEELS, curl_download, find_wheel_url  # noqa: E402

MISSING_RE = re.compile(
    r"satisfies the requirement\s+([A-Za-z0-9_.\-]+)", re.I
)
FALLBACK_RE = re.compile(
    r"No matching distribution found for\s+([A-Za-z0-9_.\-]+)", re.I
)
CONFLICT_RE = re.compile(
    r"([A-Za-z0-9_.\-]+)\s+depends on\s+([A-Za-z0-9_.\-]+)\s*([<>=!~][^;,\s]*)", re.I
)


def parse_conflict(text):
    """从冲突信息提取 (被约束包, op, ver)。

    一个包可能被多个父包约束（如 pillow>=9 与 pillow<11），先合并所有约束，
    优先取上界 <，其次 ~=，其次 ==，再次 >=。
    """
    grouped = {}
    for m in CONFLICT_RE.finditer(text):
        pkg = m.group(2).lower().replace("_", "-")
        cons = m.group(3) or ""
        grouped.setdefault(pkg, []).append(cons)
    for pkg, conss in grouped.items():
        blob = " ".join(conss)
        mm = re.search(r"<([=]?)\s*([0-9][0-9.]*)", blob)
        if mm:
            return pkg, ("<=" if mm.group(1) == "=" else "<"), mm.group(2)
        mm = re.search(r"~=\s*([0-9]+(?:\.[0-9]+)*)", blob)
        if mm:
            parts = mm.group(1).split(".")
            upper = f"{parts[0]}.{int(parts[1]) + 1}" if len(parts) >= 2 else f"{int(parts[0]) + 1}"
            return pkg, "<", upper
        mm = re.search(r"==\s*([0-9][0-9.]*)", blob)
        if mm:
            return pkg, "==", mm.group(1)
        mm = re.search(r">=\s*([0-9][0-9.]*)", blob)
        if mm:
            return pkg, ">=", mm.group(1)
    return None


def pip_dry_run(wheels):
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "--no-index",
         "--no-build-isolation", "--no-cache-dir", *wheels],
        capture_output=True, text=True,
    )
    return r.returncode, r.stdout + r.stderr


def parse_missing(text):
    names = set()
    for m in MISSING_RE.finditer(text):
        names.add(m.group(1).lower().replace("_", "-"))
    if not names:
        for m in FALLBACK_RE.finditer(text):
            names.add(m.group(1).lower().replace("_", "-"))
    return names


def parse_spec(s: str):
    m = re.match(r"([A-Za-z0-9_.\-]+?)((==|>=|<=|<|>)([0-9][0-9.]*))?$", s.strip().lower())
    if not m:
        return s.strip().lower(), None, None
    return m.group(1).replace("_", "-"), m.group(3), m.group(4)


def main():
    args = sys.argv[1:]
    source = "huawei"
    if args and args[0].startswith("--source="):
        source = args[0].split("=", 1)[1]
        args = args[1:]
    top = args
    specs = {}
    PINS = os.path.join(os.path.dirname(WHEELS), "pins.json")
    if os.path.exists(PINS):
        import json as _json
        with open(PINS, encoding="utf-8") as f:
            specs.update({k: tuple(v) for k, v in _json.load(f).items()})
    for t in top:
        name, op, ver = parse_spec(t)
        if op:
            specs[name] = (op, ver)
    os.makedirs(WHEELS, exist_ok=True)
    wheel_files = {}

    # 先装 setuptools/wheel（sdist 如 whisper 的构建需要，py3.12 venv 默认没有）
    import importlib.util as _ilu
    if _ilu.find_spec("setuptools") is None or _ilu.find_spec("wheel") is None:
        print("--- bootstrap setuptools+wheel ---")
        for pkg in ("setuptools", "wheel"):
            url, fn = find_wheel_url(pkg, source=source)
            if not url:
                print(f"  !! cannot fetch {pkg}")
                continue
            dst = os.path.join(WHEELS, fn)
            if not os.path.exists(dst) and not curl_download(url, dst):
                print(f"  !! download failed {pkg}")
                continue
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", dst],
                           capture_output=True, text=True)
            wheel_files[pkg] = dst
            print(f"  ok {fn}")

    for rnd in range(80):
        todo = []
        if rnd == 0:
            todo = [parse_spec(t)[0] for t in top]
        else:
            rc, out = pip_dry_run(list(wheel_files.values()))
            if rc == 0:
                break
            missing = parse_missing(out)
            todo = [m for m in missing if m not in wheel_files]
            if not todo:
                conf = parse_conflict(out)
                if conf:
                    pkg, op, ver = conf
                    if pkg in specs and specs[pkg] == (op, ver):
                        print("UNRESOLVABLE (conflict pin already applied):")
                        print(out[-2500:])
                        return
                    specs[pkg] = (op, ver)
                    import json as _json
                    with open(PINS, "w", encoding="utf-8") as f:
                        _json.dump(specs, f, indent=1)
                    # 作废已抓的旧 wheel，按新约束重抓
                    old = wheel_files.pop(pkg, None)
                    if old and os.path.exists(old):
                        os.remove(old)
                        print(f"  re-pin {pkg}{op}{ver} (replaced {os.path.basename(old)})")
                    else:
                        print(f"  pin {pkg}{op}{ver}")
                    todo = [pkg]
                else:
                    print("UNRESOLVABLE (no missing names / no conflict parsed):")
                    print(out[-2500:])
                    return
        print(f"--- round {rnd}: fetching {todo} ---")
        for pkg in todo:
            if pkg in wheel_files:
                continue
            op, ver = specs.get(pkg, (None, None))
            url, fn = find_wheel_url(pkg, source=source, spec_op=op, spec_ver=ver)
            if not url:
                print(f"  !! cannot fetch {pkg}")
                continue
            dst = os.path.join(WHEELS, fn)
            if not os.path.exists(dst) or os.path.getsize(dst) == 0:
                if not curl_download(url, dst):
                    print(f"  !! download failed {pkg}")
                    continue
            wheel_files[pkg] = dst
            print(f"  ok {fn}")

    print(f"=== final resolution over {len(wheel_files)} wheels ===")
    rc, out = pip_dry_run(list(wheel_files.values()))
    if rc != 0:
        print("STILL UNRESOLVED:")
        print(out[-3000:])
        return
    print("RESOLVED. installing (--no-index --no-build-isolation --no-cache-dir, no network)...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-build-isolation",
         "--no-cache-dir", *wheel_files.values()],
        capture_output=True, text=True,
    )
    tail = (r.stdout + r.stderr).strip().splitlines()[-12:]
    print("\n".join(tail))
    print("rc:", r.returncode)


if __name__ == "__main__":
    main()
