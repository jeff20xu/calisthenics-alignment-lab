"""Per-connection browser-video processing; no Streamlit calls or recordings."""
import json
import math
import threading
import time
from urllib.parse import urlsplit

import av
from streamlit_webrtc import VideoProcessorBase

from analyzer import FrameTracker


def ice_configuration(raw=None):
    """Parse ICE_SERVERS without including credentials in validation errors."""
    if raw is None:
        return {'iceServers': [{'urls': ['stun:stun.l.google.com:19302']}]}
    try:
        servers = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError) as exc:
        raise ValueError('ICE_SERVERS must be a JSON array of ICE server objects.') from exc
    if not isinstance(servers, list) or not servers:
        raise ValueError('ICE_SERVERS must be a nonempty JSON array of ICE server objects.')
    clean = []
    for index, server in enumerate(servers, 1):
        error = f'ICE_SERVERS entry {index} needs valid stun:/stuns:/turn:/turns: URLs and string credentials.'
        if not isinstance(server, dict) or set(server) - {'urls', 'username', 'credential', 'credentialType'}:
            raise ValueError(error)
        urls = server.get('urls')
        urls = [urls] if isinstance(urls, str) else urls
        if not isinstance(urls, list) or not urls:
            raise ValueError(error)
        for url in urls:
            if not isinstance(url, str) or any(char.isspace() for char in url):
                raise ValueError(error)
            scheme, separator, address = url.partition(':')
            if not separator or scheme not in ('stun', 'stuns', 'turn', 'turns'):
                raise ValueError(error)
            try:
                parsed = urlsplit('//' + address)
                port = parsed.port
                valid = (parsed.hostname and not parsed.username and not parsed.password
                         and not parsed.path and not parsed.fragment
                         and parsed.query in ('', 'transport=udp', 'transport=tcp')
                         and (port is None or port > 0))
            except ValueError:
                valid = False
            if not valid:
                raise ValueError(error)
        if any(not isinstance(server[key], str) for key in ('username', 'credential') if key in server):
            raise ValueError(error)
        if server.get('credentialType', 'password') != 'password':
            raise ValueError('ICE_SERVERS supports password credentials, not OAuth credentials.')
        clean.append({key: value for key, value in {**server, 'urls': urls}.items() if key != 'credentialType'})
    return {'iceServers': clean}


class LiveProcessor(VideoProcessorBase):
    """One tracker per peer; serialize inference, resets, and detector release."""

    def __init__(self, config):
        self._process_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._config = dict(config)
        self._tracker = None
        self._closed = False
        self._origin = None
        self._last_time = -0.001
        self._frames = 0
        self._latest = {'row': None, 'summary': None, 'error': None, 'closed': False}

    def _publish(self, **values):
        with self._snapshot_lock:
            self._latest.update(values)

    def snapshot(self):
        with self._snapshot_lock:
            return {key: dict(value) if isinstance(value, dict) else value
                    for key, value in self._latest.items()}

    def _release(self):
        tracker, self._tracker = self._tracker, None
        if tracker is not None:
            tracker.close()

    def configure(self, config):
        with self._process_lock:
            if self._config == config or self._closed:
                return
            try:
                self._release()
            except Exception as exc:
                self._closed = True
                self._publish(error=f'Detector cleanup failed: {exc}', closed=True)
                return
            self._config = dict(config)
            self._origin = None
            self._last_time = -0.001
            self._frames = 0
            self._publish(row=None, summary=None, error=None)

    def close(self):
        with self._process_lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._release()
            except Exception as exc:
                self._publish(error=f'Detector cleanup failed: {exc}')
            finally:
                self._publish(closed=True)

    def on_ended(self):
        self.close()

    async def recv_queued(self, frames):
        # Drop stale arrivals instead of doing progressively larger batches.
        return [self.recv(frames[-1])] if frames else []

    def recv(self, frame):
        with self._process_lock:
            if self._closed:
                return frame
            try:
                if self._tracker is None:
                    self._tracker = FrameTracker(**self._config)
                clock = float(frame.time) if frame.time is not None else time.monotonic()
                if not math.isfinite(clock):
                    raise ValueError('Camera supplied an invalid frame timestamp.')
                if self._origin is None:
                    self._origin = clock
                timestamp = max(clock - self._origin, self._last_time + 0.001)
                image, row = self._tracker.process(frame.to_ndarray(format='bgr24'), timestamp)
                self._last_time = timestamp
                self._frames += 1
                fps = (self._frames - 1) / timestamp if timestamp > 0 else 0.0
                summary = self._tracker.summary(duration_s=timestamp, sampling_fps=fps)
                output = av.VideoFrame.from_ndarray(image, format='bgr24')
                output.pts, output.time_base = frame.pts, frame.time_base
                self._publish(row=row, summary=summary)
                return output
            except Exception as exc:
                self._closed = True
                error = f'Live analysis failed: {exc}'
                try:
                    self._release()
                except Exception as cleanup_exc:
                    error += f' (Detector cleanup failed: {cleanup_exc})'
                self._publish(error=error, closed=True)
                return frame
