"""L10 启用 CI：把 deploy/ci.workflow.yml.template 推送到 .github/workflows/ci.yml。

需要 GitHub PAT 带 `workflow` scope（否则 GitHub 会拒绝，这是平台限制）。

用法：
    python tools/enable_ci.py --token ghp_xxx
    # 或设置环境变量后不传 --token：set GITHUB_TOKEN=ghp_xxx

行为：
  1. 读模板内容，GET 现有 ci.yml 拿 sha（若存在）
  2. PUT 到 .github/workflows/ci.yml
  3. 列出仓库 Actions workflows 验证
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER_REPO = "Abyss-sword/Voice-Guard"
TEMPLATE = os.path.join(ROOT, "deploy", "ci.workflow.yml.template")
TARGET = ".github/workflows/ci.yml"
API = "https://api.github.com"


def req(method: str, path: str, token: str, payload=None):
    url = f"{API}/repos/{OWNER_REPO}/{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "voiceguard-enable-ci",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            body = resp.read().decode()
            return resp.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode()[:300]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""))
    args = ap.parse_args()
    token = args.token.strip()
    if not token.startswith(("ghp_", "github_pat_", "gho_")):
        sys.exit("需要 --token（或 GITHUB_TOKEN 环境变量）：ghp_ 或 github_pat_ 开头")
    content = open(TEMPLATE, encoding="utf-8").read()
    if not content.strip():
        sys.exit(f"模板为空：{TEMPLATE}")

    status, body = req("GET", f"contents/{TARGET}", token)
    sha = body.get("sha") if status == 200 else None
    print(f"现有 {TARGET}: {'存在 sha=' + sha[:8] if sha else '不存在'}")

    payload = {"message": "ci: 添加轻量 CI（compileall + 纯数值测试 + 自检）",
               "content": base64.b64encode(content.encode()).decode()}
    if sha:
        payload["sha"] = sha
    status, body = req("PUT", f"contents/{TARGET}", token, payload)
    print("PUT status:", status)
    if status not in (200, 201):
        print(json.dumps(body, ensure_ascii=False)[:500])
        sys.exit("推送失败——确认 PAT 带 workflow scope")

    status, body = req("GET", "actions/workflows", token)
    print("workflows:", status)
    for wf in body.get("workflows", []):
        print(f"  - {wf.get('name')} ({wf.get('path')}) state={wf.get('state')}")

    print("\n✅ CI 已启用。首次触发：push 或到 Actions 页面手动 Run workflow（workflow_dispatch）")


if __name__ == "__main__":
    main()
