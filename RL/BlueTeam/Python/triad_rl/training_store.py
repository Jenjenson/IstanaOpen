"""Process-scoped ownership of a training run directory, using OS file locks."""
from __future__ import annotations

import os
from pathlib import Path


class TrainingStoreLease:
    """A crashed process releases its lease automatically; no stale PID guessing.

    Both APIs lock an independently opened handle, so separate managers in one
    process are also excluded. The harmless lock file remains after release.
    """

    def __init__(self, root):
        self.path = Path(root) / ".training.lock"
        self.stream = None

    def acquire(self):
        if self.stream is not None:
            return True
        stream = self.path.open("a+b")
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.stream = stream
        return True

    def release(self):
        if self.stream is not None:
            # Closing the owning handle releases the OS lock on both platforms.
            self.stream.close()
            self.stream = None
