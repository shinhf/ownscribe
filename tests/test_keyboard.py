"""Tests for the cross-platform key reader."""

from __future__ import annotations

import types
from unittest import mock

from ownscribe.keyboard import key_reader


def _fake_msvcrt(keys: list[str]) -> types.SimpleNamespace:
    pending = list(keys)
    return types.SimpleNamespace(kbhit=lambda: bool(pending), getwch=lambda: pending.pop(0))


class TestWindowsKeyReader:
    def test_returns_pressed_key(self):
        with (
            mock.patch("ownscribe.keyboard.sys.platform", "win32"),
            mock.patch.dict("sys.modules", {"msvcrt": _fake_msvcrt(["m"])}),
            key_reader() as poll,
        ):
            assert poll(0.5) == "m"

    def test_returns_none_after_timeout_without_key(self):
        with (
            mock.patch("ownscribe.keyboard.sys.platform", "win32"),
            mock.patch.dict("sys.modules", {"msvcrt": _fake_msvcrt([])}),
            mock.patch("ownscribe.keyboard.time.sleep") as sleep,
            key_reader() as poll,
        ):
            assert poll(0.0) is None
        sleep.assert_not_called()


class TestPosixKeyReader:
    def test_cbreak_is_restored_on_exit(self):
        fake_termios = types.SimpleNamespace(
            TCSADRAIN=1,
            tcgetattr=mock.MagicMock(return_value=["saved"]),
            tcsetattr=mock.MagicMock(),
        )
        fake_tty = types.SimpleNamespace(setcbreak=mock.MagicMock())
        fake_select = types.SimpleNamespace(select=mock.MagicMock(return_value=([1], [], [])))
        stdin = mock.MagicMock()
        stdin.read.return_value = "m"

        with (
            mock.patch("ownscribe.keyboard.sys.platform", "linux"),
            mock.patch("ownscribe.keyboard.sys.stdin", stdin),
            mock.patch.dict("sys.modules", {"termios": fake_termios, "tty": fake_tty, "select": fake_select}),
        ):
            with key_reader() as poll:
                assert poll(0.5) == "m"
            fake_termios.tcsetattr.assert_called_once_with(stdin, 1, ["saved"])
        fake_tty.setcbreak.assert_called_once()
