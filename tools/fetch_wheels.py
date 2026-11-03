"""用 curl 从华为云镜像抓取 wheel 并本地安装（绕开 pip 网络层卡死）。

机制：
  1. curl 拉 https://repo.huaweicloud.com/repository/pypi/simple/<pkg>/ 页面
  2. 解析出兼容 wheel（cp312-win_amd64 优先，其次 py3-none-any）
  3. curl 断点续传下载到 refs/wheels/
  4. 全部下完后 `pip install --no-index` 本地安装（无网络请求，不会卡）

用法：
    python tools/fetch_wheels.py gdown requests tqdm filelock beautifulsoup4 ...
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHEELS = os.path.join(ROOT, "refs", "wheels")
PYPI = "https://pypi.org/simple"
HUAWEI = "https://repo.huaweicloud.com/repository/pypi"
UA = "voiceguard-repro/0.1"


def curl(args, max_time: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["curl", "-s", "-H", f"User-Agent: {UA}", "--max-time", str(max_time), *args],
        capture_output=True, text=True,
    )


def curl_download(url: str, dst: str) -> bool:
    """低速 60s 即中断并重试 10 次；失败时删除残缺文件（防止续传拼接出坏包）。"""
    r = subprocess.run([
        "curl", "-L", "-C", "-", "--retry", "10", "--retry-delay", "3",
        "--retry-all-errors", "--speed-time", "60", "--speed-limit", "1024",
        "-H", f"User-Agent: {UA}", "-o", dst, url,
    ])
    if r.returncode != 0 and os.path.exists(dst):
        try:
            os.remove(dst)
        except OSError:
            pass
    return r.returncode == 0


def _vkey(v: str):
    return [int(x) for x in re.findall(r"\d+", v)]


def _wheel_parts(fn: str, pkg_norm: str):
    """wheel 文件名 -> (version, python_tag, abi, platform_tag) 或 None。"""
    m = re.match(rf"^{re.escape(pkg_norm)}-(.+?)-([\w.]+)-([\w.]+)-([\w.]+)\.whl$", fn, re.I)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def _compat(python_tag: str, abi: str, platform: str) -> int:
    """兼容分：越大越合适；-1 表示不适配 py3.12-win。"""
    if "macosx" in platform or "manylinux" in platform or "musllinux" in platform:
        return -1
    if platform not in ("win_amd64", "any"):
        return -1
    s = 0
    ok = False
    if python_tag.startswith("cp312") or abi == "cp312":
        s += 4
        ok = True
    elif python_tag.startswith("cp") and abi == "abi3":
        s += 3  # abi3 稳定 ABI：cp310-abi3 等对 cp312 兼容
        ok = True
    elif python_tag.startswith("cp"):
        return -1  # 非 312 且非 abi3 的 ABI 轮子装不上
    if "py3" in python_tag or "py2.py3" in python_tag:
        s += 3
        ok = True
    elif not ok:
        return -1
    if "win_amd64" in platform:
        s += 2
    return s


def _is_prerelease(ver: str) -> bool:
    return bool(re.search(r"(a|b|rc|dev)\d", ver, re.I))


def pick_best(files, spec_op: str = None, spec_ver: str = None):
    """从 (filename,url,yanked) 列表里选：兼容分最高、版本最新的非 yanked 文件。

    优先 wheel；无兼容 wheel 时退回 sdist（tar.gz）。可加版本约束（==/< 等）。
    """
    wheels, sdists = [], []
    for fn, url, yanked in files:
        if yanked:
            continue
        if fn.endswith(".whl"):
            wheels.append((fn, url))
        elif fn.endswith(".tar.gz") or fn.endswith(".zip"):
            sdists.append((fn, url))

    def allowed(ver):
        if not spec_op:
            return True
        v = _vkey(ver)
        s = _vkey(spec_ver)
        return {"==": v == s, "<": v < s, "<=": v <= s,
                ">": v > s, ">=": v >= s}.get(spec_op, True)

    best = None
    for fn, url in wheels:
        pkg_norm = fn.split("-")[0]
        parts = _wheel_parts(fn, pkg_norm)
        if not parts:
            continue
        ver, py, abi, plat = parts
        if _is_prerelease(ver) or not allowed(ver):
            continue
        score = _compat(py, abi, plat)
        if score < 0:
            continue
        # 版本优先，平台分仅在版本相同时打破平局
        cand = (_vkey(ver), score, fn, url)
        if best is None or (cand[0], cand[1]) > (best[0], best[1]):
            best = cand
    if best:
        return best[3], best[2]
    if sdists:
        ok = [(fn, u) for fn, u in sdists
              if not _is_prerelease(_ver_from_name(fn)) and allowed(_ver_from_name(fn))]
        if ok:
            ok.sort(key=lambda x: _vkey(_ver_from_name(x[0])), reverse=True)
            return ok[0][1], ok[0][0]
    return None, None


def _ver_from_name(fn: str) -> str:
    m = re.search(r"-(\d[^-]*)\.(tar\.gz|zip)$", fn)
    return m.group(1) if m else "0"


def fetch_pypi_json(pkg: str):
    r = curl(["-H", "Accept: application/vnd.pypi.simple.v1+json", f"{PYPI}/{pkg}/"], max_time=40)
    if r.returncode == 0 and r.stdout.lstrip().startswith("{"):
        data = json.loads(r.stdout)
        return [(f["filename"], f["url"], f.get("yanked", False)) for f in data.get("files", [])]
    return None


def fetch_huawei_html(pkg: str):
    r = curl([f"{HUAWEI}/simple/{pkg}/"], max_time=40)
    if r.returncode != 0 or not r.stdout:
        return []
    out = []
    for h in re.findall(r'href="([^"]+\.(whl|tar\.gz|zip)[^"]*)"', r.stdout):
        href = h[0]
        u = href.split("#")[0]
        if u.startswith("../../"):
            u = HUAWEI + "/" + u[6:]
        elif u.startswith("/"):
            u = "https://repo.huaweicloud.com" + u
        out.append((os.path.basename(u), u, False))
    return out


def find_wheel_url(pkg: str, source: str = "huawei", spec_op: str = None, spec_ver: str = None):
    """选源抓 index：huawei（快，默认）/ pypi（精确 JSON）/ auto。
    任一源抓取失败或选不出兼容版本时，自动兜底另一源。"""
    tried = []
    if source in ("huawei", "auto"):
        files = fetch_huawei_html(pkg)
        tried.append("huawei")
        if files:
            url, fn = pick_best(files, spec_op, spec_ver)
            if url:
                return url, fn
    if source in ("pypi", "auto") or "huawei" in tried:
        files = fetch_pypi_json(pkg)
        if files is not None:
            url, fn = pick_best(files, spec_op, spec_ver)
            if url:
                return url, fn
    print(f"  [FAIL] {pkg}: no files (tried {tried})")
    return None, None


def main(pkgs, install: bool = True):
    os.makedirs(WHEELS, exist_ok=True)
    downloaded = []
    for pkg in pkgs:
        url, fname = find_wheel_url(pkg)
        if not url:
            continue
        dst = os.path.join(WHEELS, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  [have] {fname}")
        else:
            print(f"  [curl] {pkg} <- {fname}")
            if not curl_download(url, dst):
                print(f"  [FAIL] {pkg}: download failed")
                continue
        downloaded.append(dst)

    if downloaded and install:
        print(f"[pip --no-index] {len(downloaded)} wheels")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", *downloaded],
            capture_output=True, text=True,
        )
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        print("\n".join(tail))
        print("rc:", r.returncode)
    elif downloaded:
        print("downloaded (no install):")
        for d in downloaded:
            print("  ", os.path.basename(d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pkgs", nargs="+")
    ap.add_argument("--no-install", action="store_true")
    a = ap.parse_args()
    main(a.pkgs, install=not a.no_install)
