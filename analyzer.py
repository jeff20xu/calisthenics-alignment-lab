"""Shared frame tracking for live cameras and annotated video/CSV exports."""
from pathlib import Path
from contextlib import ExitStack
import math
import os
import tempfile
import threading
import urllib.request
import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
import pandas as pd
from biomechanics import SIDES, METRICS, RepCounter, measurements, in_posture

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
_MODEL_LOCK = threading.Lock()


def ensure_model():
    path = Path(__file__).parent / 'models' / 'pose_landmarker_full.task'
    with _MODEL_LOCK:
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.download')
            os.close(fd)
            try:
                with urllib.request.urlopen(MODEL_URL, timeout=90) as response, open(temporary, 'wb') as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
    return path


class FrameTracker:
    """One serialized stream; close when finished, including after failures.

    Times are elapsed seconds and must increase strictly. Holds measure elapsed
    time between measured frames, never an assumed frame duration. Missing
    measurements or gaps over 250 ms break hold and rep continuity. Dimension
    changes also restart detector tracking and torso calibration, retaining
    cumulative counts. Frames are resized to at most 960 px, with even dimensions.
    """

    def __init__(self, skill='Handstand', side='Left', variant='Full', confidence=.6,
                 detector_inverted=False, mass_kg=0.0, torso_m=0.0):
        if skill not in ('Handstand', 'Handstand push-up', 'Planche') or side not in SIDES:
            raise ValueError('Choose a supported skill and visible body side.')
        self.skill, self.side, self.variant = skill, side, variant
        self.confidence, self.detector_inverted = confidence, detector_inverted
        self.mass_kg, self.torso_m = mass_kg, torso_m
        self._dimensions = None
        self._last_time = None
        self._last_timestamp_ms = -1
        self._previous = None
        self._last_valid = None
        self._last_measured = None
        self._reference_samples = []
        self._torso_reference = None
        self._counter = RepCounter()
        self._current_hold = self._longest = 0.0
        self._analyzed_frames = self._measured_frames = 0
        self._options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=.5,
            min_pose_presence_confidence=.5, min_tracking_confidence=.5)
        self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(self._options)

    def close(self):
        """Release the detector once; summary remains available afterwards."""
        detector, self._detector = self._detector, None
        if detector is not None:
            detector.close()

    def _reset_continuity(self, t):
        self._previous = self._last_valid = self._last_measured = None
        self._current_hold = 0.0
        self._counter.update(t, None)

    def process(self, frame_bgr, time_seconds):
        if self._detector is None:
            raise RuntimeError('This frame tracker is closed.')
        try:
            t = float(time_seconds)
            if not math.isfinite(t) or t < 0 or (self._last_time is not None and t <= self._last_time):
                raise ValueError('Frame times must be finite, nonnegative, and strictly increasing.')
            return self._process(frame_bgr, t)
        except Exception:
            self.close()
            raise

    def _process(self, frame, t):
        dimensions = frame.shape[:2]
        if self._dimensions is not None and dimensions != self._dimensions:
            self._reset_continuity(t)
            self._reference_samples.clear()
            self._torso_reference = None
            self.close()
            self._detector = mp.tasks.vision.PoseLandmarker.create_from_options(self._options)
        elif self._last_time is not None and t-self._last_time > .25:
            self._reset_continuity(t)
        self._dimensions = dimensions
        self._last_time = t
        height, width = dimensions
        scale = min(1.0, 960/max(height, width))
        width, height = max(2, int(width*scale)//2*2), max(2, int(height*scale)//2*2)
        frame = cv2.resize(frame, (width, height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector_rgb = np.ascontiguousarray(rgb[::-1, ::-1]) if self.detector_inverted else rgb
        timestamp_ms = max(self._last_timestamp_ms+1, round(t*1000))
        self._last_timestamp_ms = timestamp_ms
        result = self._detector.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=detector_rgb), timestamp_ms)
        row = {'time_s': t, 'status': 'Insufficient visibility', 'reps': self._counter.count,
               **{name: math.nan for name in METRICS}}
        points = None
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            selected = {name: landmarks[i] for name, i in SIDES[self.side].items()}
            if all(min(l.visibility, l.presence) >= self.confidence and
                   0 <= l.x <= 1 and 0 <= l.y <= 1 for l in selected.values()):
                points = {n: np.array([(1-l.x if self.detector_inverted else l.x)*width,
                                      (1-l.y if self.detector_inverted else l.y)*height])
                          for n, l in selected.items()}
                if self._previous is not None:
                    alpha = 1-math.exp(-(t-self._last_valid)/.08)
                    points = {n: alpha*p+(1-alpha)*self._previous[n] for n, p in points.items()}
                self._previous, self._last_valid = points, t
                torso = float(np.linalg.norm(points['hip']-points['shoulder']))
                if torso >= 5:
                    if self._torso_reference is None:
                        self._reference_samples.append(torso)
                        if len(self._reference_samples) == 10:
                            self._torso_reference = float(np.median(self._reference_samples))
                            self._reference_samples.clear()
                    if self._torso_reference is None:
                        row['status'] = 'Calibrating torso scale'
                    else:
                        values = measurements(points, self._torso_reference,
                            self.mass_kg if self.skill != 'Handstand push-up' else 0, self.torso_m)
                        if values is not None:
                            row['status'] = 'Measured' if in_posture(points, self.skill) else 'Outside selected posture'
                            if row['status'] == 'Measured':
                                row.update(values)
        measured = row['status'] == 'Measured'
        if measured:
            if self._last_measured is not None:
                self._current_hold += t-self._last_measured
            self._last_measured = t
            self._longest = max(self._longest, self._current_hold)
            self._measured_frames += 1
        else:
            self._reset_continuity(t)
        self._counter.update(t, row['elbow_deg'] if measured and self.skill == 'Handstand push-up' else None)
        row['reps'] = self._counter.count
        self._analyzed_frames += 1
        if points is not None:
            for a, b in (('wrist', 'elbow'), ('elbow', 'shoulder'), ('shoulder', 'hip'), ('hip', 'knee'), ('knee', 'ankle')):
                cv2.line(frame, tuple(points[a].astype(int)), tuple(points[b].astype(int)), (70, 220, 120), 2)
            for name, p in points.items():
                xy = tuple(p.astype(int))
                cv2.circle(frame, xy, 4, (0, 220, 255), -1)
                cv2.putText(frame, name, (xy[0]+5, xy[1]-5), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
            x = int(points['wrist'][0])
            cv2.line(frame, (x, 0), (x, height-1), (180, 160, 80), 1)
        text = f"{self.skill} | {row['status']} | {t:.2f}s"
        cv2.rectangle(frame, (0, 0), (width, 60), (24, 24, 24), -1)
        cv2.putText(frame, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1, cv2.LINE_AA)
        if measured:
            detail = f"Elbow {row['elbow_deg']:.0f} deg | Stack {row['stack_offset']:.2f} torso | Reps {self._counter.count}"
            cv2.putText(frame, detail, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, .45, (100, 230, 170), 1, cv2.LINE_AA)
        return frame, row

    def summary(self, duration_s, sampling_fps):
        return {'skill': self.skill, 'side': self.side, 'variant': self.variant,
                'duration_s': duration_s, 'analyzed_frames': self._analyzed_frames,
                'measured_fraction': self._measured_frames/self._analyzed_frames if self._analyzed_frames else 0.0,
                'full_reps': self._counter.count, 'longest_tracked_posture_s': self._longest,
                'torso_reference_px': self._torso_reference, 'sampling_fps': sampling_fps,
                'model': MODEL_URL, 'detector_inverted': self.detector_inverted,
                'confidence_threshold': self.confidence, 'mass_kg': self.mass_kg, 'torso_m': self.torso_m}


def analyze(source, destination, skill='Handstand', side='Left', variant='Full', confidence=.6,
            detector_inverted=False, mass_kg=0.0, torso_m=0.0, progress=None):
    if skill not in ('Handstand', 'Handstand push-up', 'Planche') or side not in SIDES:
        raise ValueError('Choose a supported skill and visible body side.')
    with ExitStack() as resources:
        cap = cv2.VideoCapture(str(source))
        resources.callback(cap.release)
        if not cap.isOpened():
            raise ValueError('Cannot open this video. Export it as an MP4 with a standard video codec.')
        writer = None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise ValueError('Video frame rate or duration is unavailable.')
        if total/fps > 180:
            raise ValueError('Use a clip of 3 minutes or less.')
        stride = max(1, math.ceil(fps/15))
        out_fps = fps/stride
        tracker = FrameTracker(skill=skill, side=side, variant=variant, confidence=confidence,
                               detector_inverted=detector_inverted, mass_kg=mass_kg, torso_m=torso_m)
        resources.callback(tracker.close)
        rows = []
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = index
            index += 1
            if frame_index % stride:
                continue
            frame, row = tracker.process(frame, frame_index/fps)
            height, width = frame.shape[:2]
            if writer is None:
                output_size = (width, height)
                writer = imageio_ffmpeg.write_frames(str(destination), (width, height), fps=out_fps,
                    codec='libx264', pix_fmt_in='rgb24', pix_fmt_out='yuv420p', macro_block_size=1,
                    output_params=['-movflags', '+faststart'])
                resources.callback(writer.close)
                writer.send(None)
            elif (width, height) != output_size:
                frame = cv2.resize(frame, output_size)
            rows.append(row)
            writer.send(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if progress:
                progress(min(index/total, 1.0))
        if not rows:
            raise ValueError('The video contained no decodable frames.')
        df = pd.DataFrame(rows)
        return df, tracker.summary(min(index, total)/fps, out_fps)
