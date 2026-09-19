"""Loopback-only report server with MP4 byte ranges for reliable checkpoint seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


class ReportHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.range_remaining = None
        path = Path(self.translate_path(self.path))
        value = self.headers.get("Range")
        if path.suffix.lower() != ".mp4" or not path.is_file() or not value:
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", value)
        if not match or not any(match.groups()):
            return self.invalid_range(size)
        left, right = match.groups()
        if left:
            start = int(left); end = min(int(right), size-1) if right else size-1
        else:
            start, end = max(0, size-int(right)), size-1
        if start > end or start >= size:
            return self.invalid_range(size)
        stream = path.open("rb")
        stream.seek(start)
        self.range_remaining = end-start+1
        self.send_response(206)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self.range_remaining))
        self.end_headers()
        return stream

    def invalid_range(self, size):
        self.send_response(416)
        self.send_header("Content-Range", f"bytes */{size}")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return None

    def copyfile(self, source, outputfile):
        if self.range_remaining is None:
            return super().copyfile(source, outputfile)
        remaining = self.range_remaining
        while remaining:
            data = source.read(min(65536, remaining))
            if not data: break
            outputfile.write(data)
            remaining -= len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=9050)
    args = parser.parse_args()
    if not (args.directory / "index.html").is_file():
        parser.error("Render the warning report first")
    print(f"Report: http://127.0.0.1:{args.port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), partial(ReportHandler, directory=str(args.directory.resolve()))).serve_forever()
