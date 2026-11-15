import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from analyzer import FrameTracker, analyze
from biomechanics import SIDES


def detected_pose(bent=False):
    coordinates = {'wrist': (100, 600), 'elbow': (200 if bent else 100, 500),
                   'shoulder': (100, 400), 'hip': (100, 250),
                   'knee': (100, 150), 'ankle': (100, 50)}
    landmarks = [None] * 33
    for name, index in SIDES['Left'].items():
        x, y = coordinates[name]
        landmarks[index] = SimpleNamespace(x=x/320, y=y/640, visibility=1.0, presence=1.0)
    return SimpleNamespace(pose_landmarks=[landmarks])


class FrameTrackerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((640, 320, 3), dtype=np.uint8)
        self.result = detected_pose()
        self.detectors = []
        model = patch('analyzer.ensure_model', return_value='unused.task')
        model.start()
        self.addCleanup(model.stop)
        factory = patch('analyzer.mp.tasks.vision.PoseLandmarker.create_from_options',
                        side_effect=self.make_detector)
        factory.start()
        self.addCleanup(factory.stop)

    def make_detector(self, options):
        detector = Mock()
        detector.detect_for_video.side_effect = lambda image, timestamp: self.result
        self.detectors.append(detector)
        return detector

    def tracker(self, **config):
        tracker = FrameTracker(**config)
        self.addCleanup(tracker.close)
        return tracker

    def calibrate(self, tracker):
        for index in range(10):
            _, row = tracker.process(self.frame, index/10)
        self.assertEqual(row['status'], 'Measured')
        self.assertEqual(tracker.summary(.9, 10)['longest_tracked_posture_s'], 0)

    def test_hold_uses_elapsed_time_and_breaks_on_missing_frames_and_gaps(self):
        tracker = self.tracker()
        self.calibrate(tracker)
        tracker.process(self.frame, .97)
        tracker.process(self.frame, 1.13)
        self.assertAlmostEqual(tracker.summary(1.13, 30)['longest_tracked_posture_s'], .23)
        self.result = SimpleNamespace(pose_landmarks=[])
        _, missing = tracker.process(self.frame, 1.17)
        self.assertEqual(missing['status'], 'Insufficient visibility')
        self.result = detected_pose()
        tracker.process(self.frame, 1.2)
        tracker.process(self.frame, 1.3)
        tracker.process(self.frame, 2.0)
        tracker.process(self.frame, 2.05)
        summary = tracker.summary(2.05, 30)
        self.assertAlmostEqual(summary['longest_tracked_posture_s'], .23)
        self.assertEqual(summary['analyzed_frames'], 17)
        self.assertAlmostEqual(summary['measured_fraction'], 7/17)

    def test_submillisecond_times_remain_strictly_increasing_for_detector(self):
        tracker = self.tracker()
        for t in (0, .00001, .00002, .0001, .001):
            tracker.process(self.frame, t)
        timestamps = [call.args[1] for call in self.detectors[-1].detect_for_video.call_args_list]
        self.assertEqual(timestamps, [0, 1, 2, 3, 4])

    def test_continuous_sustained_cycle_counts_one_rep(self):
        tracker = self.tracker(skill='Handstand push-up')
        self.calibrate(tracker)
        tracker.process(self.frame, 1.0)
        tracker.process(self.frame, 1.1)
        self.result = detected_pose(bent=True)
        for index in range(12, 22):
            tracker.process(self.frame, index/10)
        self.result = detected_pose()
        for index in range(22, 32):
            _, row = tracker.process(self.frame, index/10)
        self.assertEqual(row['reps'], 1)
        self.assertEqual(tracker.summary(3.1, 10)['full_reps'], 1)

    def test_missing_data_gap_or_dimension_change_cannot_complete_rep(self):
        for reset in ('missing', 'gap', 'dimensions'):
            with self.subTest(reset=reset):
                self.result = detected_pose()
                tracker = self.tracker(skill='Handstand push-up')
                self.calibrate(tracker)
                tracker.process(self.frame, 1.0)
                tracker.process(self.frame, 1.1)
                self.result = detected_pose(bent=True)
                for index in range(12, 22):
                    tracker.process(self.frame, index/10)
                frame, start = self.frame, 2.3
                if reset == 'missing':
                    self.result = SimpleNamespace(pose_landmarks=[])
                    tracker.process(frame, 2.2)
                elif reset == 'gap':
                    start = 2.8
                else:
                    frame = np.zeros((1280, 640, 3), dtype=np.uint8)
                self.result = detected_pose()
                for index in range(15):
                    _, row = tracker.process(frame, start+index/10)
                self.assertEqual(row['status'], 'Measured')
                self.assertEqual(row['reps'], 0)
                tracker.close()

    def test_dimension_change_recalibrates_and_closes_previous_detector(self):
        tracker = self.tracker()
        self.calibrate(tracker)
        tracker.process(self.frame, 1.0)
        original = self.detectors[-1]
        larger = np.zeros((1280, 640, 3), dtype=np.uint8)
        for index in range(11, 21):
            _, row = tracker.process(larger, index/10)
            self.assertEqual(row['status'], 'Measured' if index == 20 else 'Calibrating torso scale')
        original.close.assert_called_once()
        summary = tracker.summary(2.0, 10)
        self.assertAlmostEqual(summary['torso_reference_px'], 225)
        self.assertAlmostEqual(summary['longest_tracked_posture_s'], .1)
        tracker.close()
        tracker.close()
        self.detectors[-1].close.assert_called_once()

    def test_inference_failure_closes_detector_once(self):
        tracker = self.tracker()
        detector = self.detectors[-1]
        detector.detect_for_video.side_effect = RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            tracker.process(self.frame, 0)
        tracker.close()
        detector.close.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            tracker.process(self.frame, 1)

    def test_invalid_times_are_rejected_and_close_detector(self):
        for invalid in (0, -.1, math.nan, math.inf):
            with self.subTest(time=invalid):
                tracker = self.tracker()
                tracker.process(self.frame, 0)
                with self.assertRaises(ValueError):
                    tracker.process(self.frame, invalid)
                self.detectors[-1].close.assert_called_once()

    def test_upload_failure_releases_capture_writer_and_detector(self):
        capture, writer = Mock(), Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = {cv2.CAP_PROP_FPS: 10, cv2.CAP_PROP_FRAME_COUNT: 20}.__getitem__
        capture.read.return_value = (True, self.frame)
        writer.send.side_effect = [None, RuntimeError('encoder failed')]
        with patch('analyzer.cv2.VideoCapture', return_value=capture), \
             patch('analyzer.imageio_ffmpeg.write_frames', return_value=writer):
            with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
                analyze('unused.mp4', 'unused-output.mp4')
        capture.release.assert_called_once()
        writer.close.assert_called_once()
        self.detectors[-1].close.assert_called_once()

    def test_unopened_video_releases_capture(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch('analyzer.cv2.VideoCapture', return_value=capture):
            with self.assertRaisesRegex(ValueError, 'Cannot open'):
                analyze('unused.mp4', 'unused-output.mp4')
        capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
