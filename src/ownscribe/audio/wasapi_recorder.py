"""Windows recorder: WASAPI loopback of the speakers plus an optional microphone.

Uses PyAudioWPatch, a PyAudio fork whose PortAudio build exposes WASAPI loopback
devices. It is imported inside functions so this module still imports on other
platforms (for tests and for ``mix_tracks``).
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ownscribe.audio.base import _SILENCE_THRESHOLD, AudioRecorder

DeviceInfo = dict[str, Any]
StreamCallback = Callable[[bytes | None, int, Any, int], tuple[None, int]]

# PortAudio's paContinue, defined here so callbacks can be built and tested without pyaudiowpatch.
_PA_CONTINUE = 0

# Gaps shorter than this are callback jitter, not silence; padding them would add clicks.
_GAP_TOLERANCE_SECONDS = 0.1
# How often idle streams are padded up to wall-clock time and the silence timer is checked.
_MONITOR_INTERVAL_SECONDS = 0.1
# Zero padding and mixing are written in blocks so long recordings never sit in memory at once.
_BLOCK_SECONDS = 10


class WasapiDeviceError(RuntimeError):
    """A requested WASAPI speaker or microphone could not be found or opened."""


class _TrackWriter:
    """Writes one stream to a mono float WAV and keeps it aligned to wall-clock time.

    WASAPI loopback delivers no packets while nothing is playing, so writing only
    what arrives would shorten the system track and shift it against the mic.
    Frames expected since ``start_time`` are compared with frames written and
    the difference is filled with zeros.
    """

    def __init__(self, path: Path, samplerate: int, start_time: float) -> None:
        self.path = path
        self._rate = samplerate
        self._start_time = start_time
        self._frames = 0
        self._tolerance = int(samplerate * _GAP_TOLERANCE_SECONDS)
        self._lock = threading.Lock()
        self._file: sf.SoundFile | None = sf.SoundFile(
            str(path), mode="w", samplerate=samplerate, channels=1, format="WAV", subtype="FLOAT"
        )

    @property
    def frames(self) -> int:
        return self._frames

    def _expected_frames(self, now: float) -> int:
        return round((now - self._start_time) * self._rate)

    def _pad_to(self, target: int) -> None:
        if self._file is None:
            return
        block = self._rate * _BLOCK_SECONDS
        while self._frames < target:
            n = min(block, target - self._frames)
            self._file.write(np.zeros(n, dtype=np.float32))
            self._frames += n

    def write(self, samples: np.ndarray, now: float) -> None:
        """Append ``samples``, which ended at monotonic time ``now``, padding any gap before them."""
        with self._lock:
            if self._file is None:
                return
            chunk_start = self._expected_frames(now) - len(samples)
            if chunk_start - self._frames > self._tolerance:
                self._pad_to(chunk_start)
            self._file.write(samples)
            self._frames += len(samples)

    def pad_idle(self, now: float) -> None:
        """Pad an idle stream up to ``now`` minus the jitter tolerance.

        The margin leaves room for a buffer that is already in flight, so it is
        not placed after zeros that cover its own time span.
        """
        with self._lock:
            self._pad_to(self._expected_frames(now) - self._tolerance)

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


def _to_mono(in_data: bytes | None, channels: int) -> np.ndarray:
    if not in_data:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(in_data, dtype=np.float32)
    if channels > 1:
        return samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return samples.copy()


def _resampled_length(frames: int, src_rate: int, dst_rate: int) -> int:
    return round(frames * dst_rate / src_rate)


def _read_resampled(source: sf.SoundFile, dst_rate: int, start: int, stop: int) -> np.ndarray:
    """Output frames ``[start, stop)`` of ``source`` downmixed to mono and resampled to ``dst_rate``.

    Positions past the end of the source are zero.
    """
    out = np.zeros(stop - start, dtype=np.float32)
    usable_stop = min(stop, _resampled_length(source.frames, source.samplerate, dst_rate))
    if usable_stop <= start:
        return out

    positions = np.arange(start, usable_stop, dtype=np.float64) * (source.samplerate / dst_rate)
    first = int(positions[0])
    last = min(int(np.ceil(positions[-1])), source.frames - 1)
    source.seek(first)
    chunk = source.read(last - first + 1, dtype="float32", always_2d=True).mean(axis=1)
    out[: usable_stop - start] = np.interp(positions, np.arange(first, first + len(chunk)), chunk)
    return out


def _mixed_blocks(sources: Sequence[sf.SoundFile], rate: int, length: int) -> Iterator[np.ndarray]:
    block = rate * _BLOCK_SECONDS
    for start in range(0, length, block):
        stop = min(start + block, length)
        mixed = np.zeros(stop - start, dtype=np.float32)
        for src in sources:
            mixed += _read_resampled(src, rate, start, stop)
        yield mixed


def mix_tracks(paths: Sequence[Path], out_path: Path) -> None:
    """Mix audio files into one mono float WAV at the highest input sample rate.

    Each track is downmixed to mono, linearly resampled, zero-padded to the
    longest track and summed. The mix is peak-normalised only if it clips. Works
    block by block, so memory stays flat for hours-long recordings.
    """
    if not paths:
        raise ValueError("mix_tracks needs at least one input track")

    with contextlib.ExitStack() as stack:
        sources = [stack.enter_context(sf.SoundFile(str(p))) for p in paths]
        rate = max(src.samplerate for src in sources)
        length = max(_resampled_length(src.frames, src.samplerate, rate) for src in sources)

        peak = max((float(np.max(np.abs(b))) for b in _mixed_blocks(sources, rate, length)), default=0.0)
        gain = 1.0 / peak if peak > 1.0 else 1.0

        with sf.SoundFile(str(out_path), mode="w", samplerate=rate, channels=1, format="WAV", subtype="FLOAT") as out:
            for block in _mixed_blocks(sources, rate, length):
                out.write(block * gain if gain != 1.0 else block)


@contextlib.contextmanager
def _pyaudio_session() -> Iterator[Any]:
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        yield pa
    finally:
        pa.terminate()


def _wasapi_host_api(pa: Any) -> DeviceInfo:
    import pyaudiowpatch as pyaudio

    try:
        return pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise WasapiDeviceError("WASAPI is not available on this system.") from exc


def _wasapi_devices(pa: Any) -> list[DeviceInfo]:
    host_index = _wasapi_host_api(pa)["index"]
    return [
        info
        for info in (pa.get_device_info_by_index(i) for i in range(pa.get_device_count()))
        if info["hostApi"] == host_index
    ]


def _matches(info: DeviceInfo, selector: str) -> bool:
    if selector.isdigit():
        return info["index"] == int(selector)
    return selector.lower() in info["name"].lower()


def _find_loopback_device(pa: Any, device: str) -> DeviceInfo:
    """Loopback twin of the default speakers, or of the speaker named/indexed by ``device``."""
    if not device:
        try:
            return pa.get_default_wasapi_loopback()
        except (OSError, LookupError) as exc:
            raise WasapiDeviceError("No default WASAPI output device to loop back.") from exc

    if device.isdigit():
        # An index may point at the speaker itself or at its loopback twin.
        try:
            info = pa.get_device_info_by_index(int(device))
            return info if info.get("isLoopbackDevice") else pa.get_wasapi_loopback_analogue_by_dict(info)
        except (OSError, LookupError, ValueError) as exc:
            raise WasapiDeviceError(
                f"No WASAPI output device with index {device}. Run 'ownscribe devices' to list them."
            ) from exc

    for info in _wasapi_devices(pa):
        if info.get("isLoopbackDevice") and _matches(info, device):
            return info
    raise WasapiDeviceError(f"No WASAPI output device matches {device!r}. Run 'ownscribe devices' to list them.")


def _find_input_device(pa: Any, mic_device: str) -> DeviceInfo:
    if not mic_device:
        default_index = _wasapi_host_api(pa)["defaultInputDevice"]
        if default_index < 0:
            raise WasapiDeviceError("No default microphone found. Use --no-mic to record speakers only.")
        return pa.get_device_info_by_index(default_index)

    for info in _wasapi_devices(pa):
        if info["maxInputChannels"] > 0 and not info.get("isLoopbackDevice") and _matches(info, mic_device):
            return info
    raise WasapiDeviceError(f"No WASAPI microphone matches {mic_device!r}. Run 'ownscribe devices' to list them.")


class WasapiLoopbackRecorder(AudioRecorder):
    """Records what plays on the speakers (WASAPI loopback) and, optionally, the microphone.

    Each stream goes to its own temp WAV next to the output; ``stop()`` mixes them
    into the requested path and removes the temp files.
    """

    def __init__(
        self,
        mic: bool = False,
        mic_device: str = "",
        device: str = "",
        silence_timeout: int = 0,
    ) -> None:
        self._mic = mic
        self._mic_device = mic_device
        self._device = device
        self._silence_timeout = silence_timeout
        self._pa: Any = None
        self._streams: list[Any] = []
        self._writers: list[_TrackWriter] = []
        self._output_path: Path | None = None
        self._muted = False
        self._last_loud_time = 0.0
        self._timed_out = False
        self._stop_monitor = threading.Event()
        self._monitor: threading.Thread | None = None

    def is_available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            import pyaudiowpatch  # noqa: F401
        except ImportError:
            return False
        return True

    @property
    def supports_mute(self) -> bool:
        return self._mic

    @property
    def is_muted(self) -> bool:
        return self._muted

    def toggle_mute(self) -> None:
        if self._mic:
            self._muted = not self._muted

    @property
    def silence_timed_out(self) -> bool:
        return self._timed_out

    @property
    def is_recording(self) -> bool:
        return bool(self._streams) and not self._timed_out and all(s.is_active() for s in self._streams)

    def _make_callback(self, writer: _TrackWriter, channels: int, *, is_mic: bool) -> StreamCallback:
        def callback(in_data: bytes | None, frame_count: int, time_info: Any, status: int) -> tuple[None, int]:
            now = time.monotonic()
            samples = _to_mono(in_data, channels)
            if is_mic and self._muted:
                samples = np.zeros_like(samples)
            writer.write(samples, now)
            if samples.size and float(np.max(np.abs(samples))) > _SILENCE_THRESHOLD:
                self._last_loud_time = now
            return None, _PA_CONTINUE

        return callback

    def _open_stream(self, info: DeviceInfo, track_path: Path, start_time: float, *, is_mic: bool) -> None:
        import pyaudiowpatch as pyaudio

        # WASAPI shared mode only accepts the device's own mix format, so take its rate and channel count.
        rate = int(info["defaultSampleRate"])
        channels = int(info["maxInputChannels"])
        writer = _TrackWriter(track_path, rate, start_time)
        self._writers.append(writer)
        stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=info["index"],
            stream_callback=self._make_callback(writer, channels, is_mic=is_mic),
        )
        self._streams.append(stream)

    def _monitor_loop(self) -> None:
        while not self._stop_monitor.wait(_MONITOR_INTERVAL_SECONDS):
            now = time.monotonic()
            for writer in self._writers:
                writer.pad_idle(now)
            if self._silence_timeout > 0 and now - self._last_loud_time > self._silence_timeout:
                self._timed_out = True

    def start(self, output_path: Path) -> None:
        import pyaudiowpatch as pyaudio

        self._output_path = output_path
        self._timed_out = False
        self._muted = False
        self._pa = pyaudio.PyAudio()
        try:
            speaker = _find_loopback_device(self._pa, self._device)
            mic = _find_input_device(self._pa, self._mic_device) if self._mic else None
            start_time = time.monotonic()
            self._last_loud_time = start_time
            self._open_stream(speaker, output_path.with_suffix(".system.wav"), start_time, is_mic=False)
            if mic is not None:
                self._open_stream(mic, output_path.with_suffix(".mic.wav"), start_time, is_mic=True)
        except BaseException:
            self._close_streams()
            for writer in self._take_writers():
                writer.path.unlink(missing_ok=True)
            raise

        self._stop_monitor.clear()
        self._monitor = threading.Thread(target=self._monitor_loop, name="wasapi-monitor", daemon=True)
        self._monitor.start()

    def _close_streams(self) -> None:
        for stream in self._streams:
            with contextlib.suppress(OSError):
                stream.stop_stream()
            stream.close()
        self._streams = []
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _take_writers(self) -> list[_TrackWriter]:
        """Close every track file and hand the writers over to the caller."""
        writers, self._writers = self._writers, []
        for writer in writers:
            writer.close()
        return writers

    def stop(self) -> None:
        if self._monitor is not None:
            self._stop_monitor.set()
            self._monitor.join()
            self._monitor = None

        stop_time = time.monotonic()
        self._close_streams()
        # Bring every track up to the stop time so trailing silence is kept.
        for writer in self._writers:
            writer.pad_idle(stop_time)
        writers = self._take_writers()
        if not writers or self._output_path is None:
            return

        # On failure the per-stream tracks stay on disk so the recording can still be recovered.
        mix_tracks([w.path for w in writers], self._output_path)
        for writer in writers:
            writer.path.unlink(missing_ok=True)

    def list_devices(self) -> str:
        with _pyaudio_session() as pa:
            devices = _wasapi_devices(pa)
            try:
                default_loopback = pa.get_default_wasapi_loopback()["index"]
            except (OSError, LookupError):
                default_loopback = None
            default_input = _wasapi_host_api(pa)["defaultInputDevice"]

        def line(info: DeviceInfo, default: int | None) -> str:
            marker = "*" if info["index"] == default else " "
            return f" {marker} [{info['index']}] {info['name']}"

        speakers = [line(d, default_loopback) for d in devices if d.get("isLoopbackDevice")]
        mics = [line(d, default_input) for d in devices if d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice")]
        return "\n".join(
            [
                "Speakers (loopback; select with --device):",
                *(speakers or ["   (none)"]),
                "",
                "Microphones (select with --mic-device):",
                *(mics or ["   (none)"]),
                "",
                "* = default",
            ]
        )
