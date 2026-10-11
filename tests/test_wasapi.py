"""Tests for the Windows WASAPI loopback recorder. No audio hardware unless marked."""

from __future__ import annotations

import time
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import soundfile as sf

from ownscribe.audio.wasapi_recorder import (
    WasapiDeviceError,
    WasapiLoopbackRecorder,
    _find_input_device,
    _find_loopback_device,
    _TrackWriter,
    mix_tracks,
)


def _write(path: Path, data: np.ndarray, rate: int) -> Path:
    sf.write(str(path), data.astype(np.float32), rate, subtype="FLOAT")
    return path


def _read(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32")
    return data, rate


class TestMixTracks:
    def test_equal_rates_are_summed(self, tmp_path):
        a = _write(tmp_path / "a.wav", np.full(1000, 0.25), 16000)
        b = _write(tmp_path / "b.wav", np.full(1000, 0.5), 16000)

        mix_tracks([a, b], tmp_path / "out.wav")

        out, rate = _read(tmp_path / "out.wav")
        assert rate == 16000
        assert len(out) == 1000
        np.testing.assert_allclose(out, 0.75, atol=1e-6)

    def test_mismatched_rates_resample_to_highest_and_stay_aligned(self, tmp_path):
        system = _write(tmp_path / "system.wav", np.zeros(48000), 48000)
        # A step at exactly 0.5 s on the 44.1 kHz track.
        step = np.where(np.arange(44100) >= 22050, 0.5, 0.0)
        mic = _write(tmp_path / "mic.wav", step, 44100)

        mix_tracks([system, mic], tmp_path / "out.wav")

        out, rate = _read(tmp_path / "out.wav")
        assert rate == 48000
        assert len(out) == 48000
        assert out[24000 - 10] == pytest.approx(0.0, abs=1e-6)
        assert out[24000 + 10] == pytest.approx(0.5, abs=1e-6)

    def test_empty_track_is_treated_as_silence(self, tmp_path):
        empty = _write(tmp_path / "system.wav", np.zeros(0), 48000)
        mic = _write(tmp_path / "mic.wav", np.full(1600, 0.5), 16000)

        mix_tracks([empty, mic], tmp_path / "out.wav")

        out, rate = _read(tmp_path / "out.wav")
        assert rate == 48000
        assert len(out) == 4800
        np.testing.assert_allclose(out[:-3], 0.5, atol=1e-6)

    def test_shorter_track_is_zero_padded(self, tmp_path):
        long = _write(tmp_path / "a.wav", np.full(200, 0.25), 8000)
        short = _write(tmp_path / "b.wav", np.full(100, 0.25), 8000)

        mix_tracks([long, short], tmp_path / "out.wav")

        out, _ = _read(tmp_path / "out.wav")
        np.testing.assert_allclose(out[:100], 0.5, atol=1e-6)
        np.testing.assert_allclose(out[100:], 0.25, atol=1e-6)

    def test_clipping_mix_is_peak_normalised(self, tmp_path):
        a = _write(tmp_path / "a.wav", np.full(500, 0.8), 16000)
        b = _write(tmp_path / "b.wav", np.full(500, 0.8), 16000)

        mix_tracks([a, b], tmp_path / "out.wav")

        out, _ = _read(tmp_path / "out.wav")
        assert float(np.max(np.abs(out))) == pytest.approx(1.0, abs=1e-6)

    def test_quiet_mix_is_not_amplified(self, tmp_path):
        a = _write(tmp_path / "a.wav", np.full(500, 0.1), 16000)

        mix_tracks([a], tmp_path / "out.wav")

        out, _ = _read(tmp_path / "out.wav")
        np.testing.assert_allclose(out, 0.1, atol=1e-6)

    def test_stereo_is_downmixed(self, tmp_path):
        stereo = np.column_stack([np.full(100, 1.0), np.zeros(100)])
        a = _write(tmp_path / "a.wav", stereo, 16000)

        mix_tracks([a], tmp_path / "out.wav")

        out, _ = _read(tmp_path / "out.wav")
        assert out.ndim == 1
        np.testing.assert_allclose(out, 0.5, atol=1e-6)

    def test_resampling_is_continuous_across_blocks(self, tmp_path):
        # 30 s at 50 Hz upsampled to 100 Hz spans three 10 s mixing blocks.
        ramp = np.arange(1500) / 1500 * 0.5
        slow = _write(tmp_path / "slow.wav", ramp, 50)
        fast = _write(tmp_path / "fast.wav", np.zeros(3000), 100)

        mix_tracks([slow, fast], tmp_path / "out.wav")

        out, rate = _read(tmp_path / "out.wav")
        assert rate == 100
        assert len(out) == 3000
        expected = np.arange(2999) / 2 / 1500 * 0.5
        np.testing.assert_allclose(out[:2999], expected, atol=1e-6)

    def test_no_tracks_is_an_error(self, tmp_path):
        with pytest.raises(ValueError):
            mix_tracks([], tmp_path / "out.wav")


class TestTrackWriterGapPadding:
    def test_hole_in_timeline_is_filled_with_zeros(self, tmp_path):
        writer = _TrackWriter(tmp_path / "t.wav", samplerate=1000, start_time=0.0)
        writer.write(np.ones(100, dtype=np.float32), now=0.1)
        # Nothing arrived between 0.1 s and 0.9 s, as with loopback during silence.
        writer.write(np.ones(100, dtype=np.float32), now=1.0)
        writer.close()

        data, _ = _read(tmp_path / "t.wav")
        assert len(data) == 1000
        np.testing.assert_array_equal(data[:100], 1.0)
        np.testing.assert_array_equal(data[100:900], 0.0)
        np.testing.assert_array_equal(data[900:], 1.0)

    def test_jitter_below_tolerance_is_not_padded(self, tmp_path):
        writer = _TrackWriter(tmp_path / "t.wav", samplerate=1000, start_time=0.0)
        writer.write(np.ones(100, dtype=np.float32), now=0.1)
        writer.write(np.ones(100, dtype=np.float32), now=0.25)  # 50 ms late
        writer.close()

        data, _ = _read(tmp_path / "t.wav")
        assert len(data) == 200
        np.testing.assert_array_equal(data, 1.0)

    def test_pad_idle_catches_up_to_wall_clock_minus_tolerance(self, tmp_path):
        writer = _TrackWriter(tmp_path / "t.wav", samplerate=1000, start_time=0.0)
        writer.pad_idle(now=2.0)
        assert writer.frames == 1900
        # A buffer that was in flight lands right after the padding, not after extra zeros.
        writer.write(np.ones(100, dtype=np.float32), now=2.0)
        writer.close()

        data, _ = _read(tmp_path / "t.wav")
        assert len(data) == 2000
        np.testing.assert_array_equal(data[1900:], 1.0)

    def test_writes_after_close_are_ignored(self, tmp_path):
        writer = _TrackWriter(tmp_path / "t.wav", samplerate=1000, start_time=0.0)
        writer.close()
        writer.write(np.ones(10, dtype=np.float32), now=5.0)
        writer.pad_idle(now=5.0)
        assert writer.frames == 0


class TestMute:
    @staticmethod
    def _run_mic_callback(tmp_path: Path, *, muted: bool) -> np.ndarray:
        recorder = WasapiLoopbackRecorder(mic=True)
        if muted:
            recorder.toggle_mute()
        writer = _TrackWriter(tmp_path / "mic.wav", samplerate=48000, start_time=time.monotonic())
        callback = recorder._make_callback(writer, channels=2, is_mic=True)

        stereo = np.full(960 * 2, 0.5, dtype=np.float32).tobytes()
        assert callback(stereo, 960, None, 0) == (None, 0)
        writer.close()
        data, _ = _read(tmp_path / "mic.wav")
        return data

    def test_muted_mic_callback_writes_zeros(self, tmp_path):
        data = self._run_mic_callback(tmp_path, muted=True)
        assert len(data) == 960
        np.testing.assert_array_equal(data, 0.0)

    def test_unmuted_mic_callback_writes_audio(self, tmp_path):
        data = self._run_mic_callback(tmp_path, muted=False)
        np.testing.assert_allclose(data, 0.5)

    def test_mute_does_not_silence_loopback(self, tmp_path):
        recorder = WasapiLoopbackRecorder(mic=True)
        recorder.toggle_mute()
        writer = _TrackWriter(tmp_path / "sys.wav", samplerate=48000, start_time=time.monotonic())
        callback = recorder._make_callback(writer, channels=1, is_mic=False)

        callback(np.full(480, 0.5, dtype=np.float32).tobytes(), 480, None, 0)
        writer.close()

        data, _ = _read(tmp_path / "sys.wav")
        np.testing.assert_allclose(data, 0.5)

    def test_supports_mute_follows_mic(self):
        assert WasapiLoopbackRecorder(mic=True).supports_mute is True
        assert WasapiLoopbackRecorder(mic=False).supports_mute is False

    def test_toggle_without_mic_is_noop(self):
        recorder = WasapiLoopbackRecorder(mic=False)
        recorder.toggle_mute()
        assert recorder.is_muted is False


class TestSilenceTracking:
    def test_loud_buffer_resets_silence_timer(self, tmp_path):
        recorder = WasapiLoopbackRecorder(silence_timeout=10)
        writer = _TrackWriter(tmp_path / "sys.wav", samplerate=48000, start_time=time.monotonic())
        callback = recorder._make_callback(writer, channels=1, is_mic=False)

        recorder._last_loud_time = 0.0
        callback(np.full(480, 0.5, dtype=np.float32).tobytes(), 480, None, 0)
        writer.close()

        assert recorder._last_loud_time > 0.0

    def test_quiet_buffer_keeps_silence_timer(self, tmp_path):
        recorder = WasapiLoopbackRecorder(silence_timeout=10)
        writer = _TrackWriter(tmp_path / "sys.wav", samplerate=48000, start_time=time.monotonic())
        callback = recorder._make_callback(writer, channels=1, is_mic=False)

        recorder._last_loud_time = 0.0
        callback(np.full(480, 1e-6, dtype=np.float32).tobytes(), 480, None, 0)
        writer.close()

        assert recorder._last_loud_time == 0.0


class TestAvailability:
    def test_unavailable_off_windows(self):
        with mock.patch("ownscribe.audio.wasapi_recorder.sys.platform", "darwin"):
            assert WasapiLoopbackRecorder().is_available() is False

    def test_unavailable_without_pyaudiowpatch(self):
        with (
            mock.patch("ownscribe.audio.wasapi_recorder.sys.platform", "win32"),
            mock.patch.dict("sys.modules", {"pyaudiowpatch": None}),
        ):
            assert WasapiLoopbackRecorder().is_available() is False

    @pytest.mark.windows
    def test_available_on_windows(self):
        assert WasapiLoopbackRecorder().is_available() is True

    def test_not_recording_before_start(self):
        assert WasapiLoopbackRecorder().is_recording is False

    def test_stop_before_start_is_noop(self):
        WasapiLoopbackRecorder().stop()


def _fake_pa(devices: list[dict], default_input: int = 1) -> mock.MagicMock:
    pa = mock.MagicMock()
    pa.get_host_api_info_by_type.return_value = {"index": 2, "defaultInputDevice": default_input}
    pa.get_device_count.return_value = len(devices)
    pa.get_device_info_by_index.side_effect = lambda i: devices[i]
    return pa


_DEVICES = [
    {"index": 0, "name": "Speakers (Realtek)", "hostApi": 2, "maxInputChannels": 0},
    {"index": 1, "name": "Microphone (Realtek)", "hostApi": 2, "maxInputChannels": 2},
    {
        "index": 2,
        "name": "Speakers (Realtek) [Loopback]",
        "hostApi": 2,
        "maxInputChannels": 2,
        "isLoopbackDevice": True,
    },
    {"index": 3, "name": "Headset Earphone [Loopback]", "hostApi": 2, "maxInputChannels": 2, "isLoopbackDevice": True},
    {"index": 4, "name": "Headset Microphone", "hostApi": 0, "maxInputChannels": 1},
    {"index": 5, "name": "Headset Microphone", "hostApi": 2, "maxInputChannels": 1},
]


@pytest.fixture
def fake_pyaudio_module():
    with mock.patch.dict("sys.modules", {"pyaudiowpatch": types.SimpleNamespace(paWASAPI=13)}):
        yield


class TestDeviceSelection:
    def test_default_speaker_uses_default_loopback(self):
        pa = _fake_pa(_DEVICES)
        pa.get_default_wasapi_loopback.return_value = _DEVICES[2]
        assert _find_loopback_device(pa, "") is _DEVICES[2]

    def test_speaker_by_name(self, fake_pyaudio_module):
        assert _find_loopback_device(_fake_pa(_DEVICES), "headset")["index"] == 3

    def test_speaker_index_maps_to_loopback_twin(self):
        pa = _fake_pa(_DEVICES)
        pa.get_wasapi_loopback_analogue_by_dict.return_value = _DEVICES[2]
        assert _find_loopback_device(pa, "0") is _DEVICES[2]
        pa.get_wasapi_loopback_analogue_by_dict.assert_called_once_with(_DEVICES[0])

    def test_unknown_speaker_raises(self, fake_pyaudio_module):
        with pytest.raises(WasapiDeviceError, match="nope"):
            _find_loopback_device(_fake_pa(_DEVICES), "nope")

    def test_mic_by_name_prefers_wasapi_host_api(self, fake_pyaudio_module):
        assert _find_input_device(_fake_pa(_DEVICES), "headset mic")["index"] == 5

    def test_default_mic(self, fake_pyaudio_module):
        assert _find_input_device(_fake_pa(_DEVICES), "")["index"] == 1

    def test_missing_default_mic_suggests_no_mic(self, fake_pyaudio_module):
        with pytest.raises(WasapiDeviceError, match="--no-mic"):
            _find_input_device(_fake_pa(_DEVICES, default_input=-1), "")


@pytest.mark.windows
@pytest.mark.hardware
class TestRealDevices:
    def test_list_devices_shows_a_loopback_speaker(self):
        listing = WasapiLoopbackRecorder().list_devices()
        assert "[Loopback]" in listing

    def test_short_loopback_recording_spans_wall_clock_time(self, tmp_path):
        recorder = WasapiLoopbackRecorder(mic=False)
        out = tmp_path / "recording.wav"
        recorder.start(out)
        time.sleep(1.0)
        recorder.stop()

        info = sf.info(str(out))
        # Gap padding keeps the track ~1 s long even if nothing played.
        assert info.duration == pytest.approx(1.0, abs=0.3)
        assert not out.with_suffix(".system.wav").exists()
