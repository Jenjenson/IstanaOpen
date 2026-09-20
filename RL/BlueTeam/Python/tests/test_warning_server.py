from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import threading

import pytest
from serve_warning_report import ReportHandler


@pytest.mark.parametrize("value,status,body", [("bytes=2-5",206,b"2345"), ("bytes=7-",206,b"789"),
                                               ("bytes=-3",206,b"789"), ("bytes=100-",416,b""),
                                               ("bytes=5-2",416,b""), ("bytes=abc",416,b"")])
def test_mp4_range_seeking(tmp_path, value, status, body):
    (tmp_path / "test.mp4").write_bytes(b"0123456789")
    server = ThreadingHTTPServer(("127.0.0.1",0), partial(ReportHandler,directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever,daemon=True); thread.start()
    try:
        connection = HTTPConnection(*server.server_address)
        connection.request("GET","/test.mp4",headers={"Range":value})
        response = connection.getresponse()
        assert response.status == status and response.read() == body
        connection.close()
    finally:
        server.shutdown(); server.server_close(); thread.join()
