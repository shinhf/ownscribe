"""Non-blocking single-key reads from the terminal, on POSIX and Windows."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

KeyPoll = Callable[[float], str | None]

CTRL_C = "\x03"
CTRL_Q = "\x11"

# msvcrt has no blocking-with-timeout read, so the wait is sliced into short sleeps.
_WINDOWS_POLL_SLICE = 0.05


@contextmanager
def key_reader() -> Iterator[KeyPoll]:
    """Yield ``poll(timeout)``, which returns one pressed key or None after ``timeout`` seconds.

    On POSIX the terminal is in cbreak mode for the duration of the block and is
    restored on exit. Windows consoles need no mode change.
    """
    if sys.platform == "win32":
        yield _poll_windows
    else:
        with _posix_cbreak() as poll:
            yield poll


def _poll_windows(timeout: float) -> str | None:
    import msvcrt

    deadline = time.monotonic() + timeout
    while True:
        if msvcrt.kbhit():
            return msvcrt.getwch()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(_WINDOWS_POLL_SLICE, remaining))


@contextmanager
def _posix_cbreak() -> Iterator[KeyPoll]:
    import select
    import termios
    import tty

    old_termios = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    # Ctrl+Q is XON flow control by default; turn that off so the key reaches poll().
    attrs = termios.tcgetattr(sys.stdin)
    attrs[0] &= ~termios.IXON
    termios.tcsetattr(sys.stdin, termios.TCSANOW, attrs)

    def poll(timeout: float) -> str | None:
        readable, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if readable else None

    try:
        yield poll
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_termios)
