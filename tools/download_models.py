"""下载模型权重与语料。

用法（在项目 .venv 中）：
    python tools/download_models.py adainvc     # attack-vc 的 AdaIN-VC checkpoint（Google Drive 文件夹）
    python tools/download_models.py vctk        # CSTR VCTK 语料（约 11 GB）
    python tools/download_models.py vqmivc      # VQMIVC 预训练（W4 用，暂缓）
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")
DATA = os.path.join(ROOT, "data")

# attack-vc README 提供的 AdaIN-VC 预训练（文件夹，含 config.yaml / attr.pkl / model.ckpt）
ADAINVC_GDRIVE_FOLDER = "1vGU7DdfuebG1y8f4s4MOqEMunzsxxAHX"

# CSTR VCTK 0.92 官方下载（已校验，datashare.ed.ac.uk）
VCTK_URL = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip?v=1f71e8e688e2859091bd62b9c8bdd17a"


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def adainvc():
    out = os.path.join(CHECKPOINTS, "adain-vc")
    os.makedirs(out, exist_ok=True)
    run([sys.executable, "-m", "gdown", "--folder", ADAINVC_GDRIVE_FOLDER, "-O", out])
    print("已下载到：", out)


def vctk():
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, "VCTK-Corpus-0.92.zip")
    run(["curl", "-L", "-o", dst, VCTK_URL])
    print("已下载到：", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["adainvc", "vctk", "vqmivc"])
    a = ap.parse_args()
    {"adainvc": adainvc, "vctk": vctk, "vqmivc": lambda: print("TODO: W4 再下")}[a.what]()


if __name__ == "__main__":
    main()
