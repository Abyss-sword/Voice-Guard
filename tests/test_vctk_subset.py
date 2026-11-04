"""HTTP range downloads must preserve ZIP contents and reject full responses."""
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import unittest
import zipfile

from tools.download_vctk_subset import HTTPRangeFile


class RangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("p225/p225_001.wav", b"first-original-waveform" * 100)
            archive.writestr("p226/p226_002.wav", b"second-original-waveform" * 100)
        cls.payload = content.getvalue()
        payload = cls.payload
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_HEAD(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("ETag", '"stable-archive"')
                self.end_headers()
            def do_GET(self):
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                if self.path == "/ignore-range":
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                self.send_header("Content-Length", str(end - start + 1))
                self.send_header("ETag", '"stable-archive"')
                self.end_headers()
                self.wfile.write(payload[start:end + 1])
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_remote_zip_matches_original_members(self):
        with HTTPRangeFile(self.url) as remote, zipfile.ZipFile(remote) as archive:
            self.assertEqual(archive.read("p225/p225_001.wav"), b"first-original-waveform" * 100)
            self.assertEqual(archive.read("p226/p226_002.wav"), b"second-original-waveform" * 100)

    def test_prefetch_avoids_extra_download_for_contained_reads(self):
        with HTTPRangeFile(self.url) as remote:
            remote.prefetch(0, len(self.payload) - 1)
            transferred = remote.transferred
            remote.seek(20)
            self.assertEqual(remote.read(30), self.payload[20:50])
            remote.seek(-10, 2)
            self.assertEqual(remote.read(), self.payload[-10:])
            self.assertEqual(remote.transferred, transferred)

    def test_ignored_range_is_rejected(self):
        with HTTPRangeFile(self.url + "/ignore-range") as remote:
            with self.assertRaisesRegex(ValueError, "exact requested byte range"):
                remote.read(10)


if __name__ == "__main__":
    unittest.main()
