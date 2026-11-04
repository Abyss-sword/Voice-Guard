"""Read selected original VCTK22 ZIP members via HTTP ranges, preserving sorted enrollment.

Each extracted member is CRC-checked by zipfile and recorded with SHA256. This is
not a full-archive SHA256 verification. Enrollment uses the same first 30 files
as the full dataset; extra candidates retain their original sorted positions.
"""
from __future__ import annotations
import argparse
import io
import http.client
import os
from pathlib import Path
import sys
import urllib.request
import urllib.parse
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.preflight import SPEAKERS
from vguard.run_state import file_sha256, write_json

URL = "https://hf-mirror.com/datasets/arampacha/vctk_22kHz/resolve/main/raw_data.zip"


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.connection = None
        self.source = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as response:
            self.url = response.url
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("ETag")
        self.position = 0
        self.cached_start, self.cached = 0, b""
        self.transferred = 0
        parsed = urllib.parse.urlsplit(self.url)
        self.connection_class = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        self.host = parsed.netloc
        self.request_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))

    def close(self):
        if self.connection:
            self.connection.close()
        super().close()

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def prefetch(self, start, end):
        end = min(end, self.size - 1)
        count = end - start + 1
        if not 0 < count <= 64 * 1024**2:
            raise ValueError(f"Invalid or excessive range: {start}-{end}")
        last_error = None
        for attempt in range(3):
            try:
                if self.connection is None:
                    self.connection = self.connection_class(self.host, timeout=60)
                self.connection.request("GET", self.request_path, headers={"Range": f"bytes={start}-{end}"})
                with self.connection.getresponse() as response:
                    expected = f"bytes {start}-{end}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError("Server did not honor the exact requested byte range")
                    if self.etag and response.headers.get("ETag") not in (None, self.etag):
                        raise ValueError("Remote archive changed during download")
                    data = response.read(count + 1)
                    if len(data) != count:
                        raise ValueError("Incomplete range response")
                self.cached_start, self.cached = start, data
                self.transferred += len(data)
                return
            except (OSError, ValueError, http.client.HTTPException) as error:
                if self.connection:
                    self.connection.close()
                    self.connection = None
                last_error = error
        raise last_error

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b""
        end = self.position + size
        if not (self.cached_start <= self.position and end <= self.cached_start + len(self.cached)):
            self.prefetch(self.position, end - 1)
        start = self.position - self.cached_start
        self.position = end
        return self.cached[start:start + size]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=str(ROOT / "data/VCTK22"))
    ap.add_argument("--candidates", type=int, default=80, help="files per test speaker, including 30 enrollment files")
    ap.add_argument("--test-speakers", default="p225,p226,p227,p228,p229")
    args = ap.parse_args()
    if args.candidates < 31:
        ap.error("candidates must be >=31")
    output = Path(args.out)
    tests = set(args.test_speakers.split(","))
    records = []
    with HTTPRangeFile(args.url) as remote, zipfile.ZipFile(remote) as archive:
        transcripts = {Path(item.filename).name: item for item in archive.infolist()
                       if item.filename.lower().endswith(".txt")}
        by_speaker = {s: [] for s in SPEAKERS}
        for info in archive.infolist():
            if not info.filename.lower().endswith(".wav"):
                continue
            parts = Path(info.filename.replace("\\", "/")).parts
            speaker = next((p for p in parts[:-1] if p in by_speaker), None)
            if speaker:
                by_speaker[speaker].append(info)
        for speaker, available in by_speaker.items():
            chosen = sorted(available, key=lambda item: Path(item.filename).name)[:args.candidates if speaker in tests else 31]
            if len(chosen) < 31:
                raise ValueError(f"{speaker}: only {len(chosen)} files found")
            members_to_extract = chosen + [transcripts[Path(item.filename).with_suffix(".txt").name]
                                           for item in chosen
                                           if Path(item.filename).with_suffix(".txt").name in transcripts]
            # Coalesce nearby ZIP members, bounded to 32MiB, to avoid one HTTP request per WAV.
            groups = []
            for item in sorted(members_to_extract, key=lambda item: item.header_offset):
                end = min(remote.size - 1, item.header_offset + item.compress_size + len(item.filename.encode("utf-8")) + len(item.extra) + 1054)
                if groups and item.header_offset - groups[-1][1] <= 128 * 1024 and end - groups[-1][0] < 32 * 1024**2:
                    groups[-1][1] = max(groups[-1][1], end)
                    groups[-1][2].append(item)
                else:
                    groups.append([item.header_offset, end, [item]])
            for start, end, members in groups:
                pending = []
                for member in members:
                    target = output / speaker / Path(member.filename).name
                    if target.is_file() and target.stat().st_size == member.file_size and zlib.crc32(target.read_bytes()) == member.CRC:
                        records.append({"path": str(target.relative_to(output)), "zip_member": member.filename,
                                        "zip_crc32": member.CRC, "sha256": file_sha256(target), "bytes": member.file_size})
                    else:
                        pending.append((member, target))
                if not pending:
                    continue
                start = min(member.header_offset for member, _ in pending)
                end = max(min(remote.size - 1, member.header_offset + member.compress_size +
                              len(member.filename.encode("utf-8")) + len(member.extra) + 1054) for member, _ in pending)
                remote.prefetch(start, end)
                for member, target in pending:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    data = archive.read(member)  # verifies ZIP CRC before publishing the file
                    partial = target.with_suffix(target.suffix + ".part")
                    partial.write_bytes(data)
                    os.replace(partial, target)
                    records.append({"path": str(target.relative_to(output)), "zip_member": member.filename,
                                    "zip_crc32": member.CRC, "sha256": file_sha256(target), "bytes": len(data)})
            print(f"{speaker}: {len(chosen)}/{len(available)} original WAVs; transferred {remote.transferred / 1024**2:.1f} MiB", flush=True)
        manifest = {"source": args.url, "archive_bytes": remote.size, "etag": remote.etag,
                    "downloaded_bytes": remote.transferred, "verification": "ZIP member CRC32 and local SHA256; full archive digest not checked",
                    "test_speakers": sorted(tests), "candidates": args.candidates, "files": records}
    write_json(output.parent / "VCTK22_subset_manifest.json", manifest)
    wave_count = sum(record["path"].lower().endswith(".wav") for record in records)
    print(f"Saved {wave_count} WAVs and {len(records) - wave_count} transcripts to {output}")


if __name__ == "__main__":
    main()
