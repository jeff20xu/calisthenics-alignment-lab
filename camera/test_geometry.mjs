import test from 'node:test';
import assert from 'node:assert/strict';
import { PoseMetrics } from './geometry.mjs';

const SIDES = {
  Left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27 },
  Right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28 },
};
const KEYS = [
  'elbow_deg', 'shoulder_deg', 'hip_deg', 'knee_deg', 'stack_offset',
  'body_line_offset', 'torso_horizontal_deg', 'leg_lever', 'shoulder_height',
  'partial_gravity_moment_nm',
];
const pose = (changes = {}) => ({
  wrist: [100, 600], elbow: [100, 500], shoulder: [100, 400],
  hip: [100, 250], knee: [100, 150], ankle: [100, 50], ...changes,
});
const planche = () => pose({
  wrist: [80, 250], elbow: [90, 175], shoulder: [100, 100],
  hip: [250, 100], knee: [350, 100], ankle: [450, 100],
});
function landmarks(points = pose(), width = 320, height = 640, side = 'Left', inverted = false) {
  const result = Array(33).fill(null);
  for (const [name, index] of Object.entries(SIDES[side])) {
    const [x, y] = points[name];
    // Browser Tasks JS commonly returns visibility but no presence property.
    result[index] = {
      x: inverted ? 1 - x / width : x / width,
      y: inverted ? 1 - y / height : y / height,
      visibility: 1,
    };
  }
  return result;
}
function close(actual, expected, tolerance = 1e-9) {
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `Expected ${actual} to be within ${tolerance} of ${expected}`);
}
function gaps(row) {
  for (const key of KEYS) assert.equal(row[key], null, key);
}
function calibrate(tracker, input = landmarks(), width = 320, height = 640, start = 0) {
  let result;
  for (let index = 0; index < 10; index += 1) {
    result = tracker.update(input, width, height, start + index / 10);
    if (index < 9) {
      assert.equal(result.row.status, 'Calibrating torso scale');
      gaps(result.row);
      assert.equal(result.summary.torso_reference_px, null);
    }
  }
  return result;
}

// Solve the smoothing equation for the input elbow, so transition tests exercise
// known *filtered* angles through the public API rather than private counters.
function repFixture(initialAngle = 180) {
  const tracker = new PoseMetrics({ skill: 'Handstand push-up' });
  const elbowFor = degrees => [100 + 100 / Math.tan(degrees * Math.PI / 360), 500];
  let previous = calibrate(tracker, landmarks(pose({ elbow: elbowFor(initialAngle) })));
  let time = 0.9;
  return {
    tracker,
    at(t, degrees) {
      const desired = elbowFor(degrees);
      const alpha = 1 - Math.exp(-(t - time) / 0.08);
      const elbow = desired.map((value, index) =>
        (value - (1 - alpha) * previous.points.elbow[index]) / alpha);
      previous = tracker.update(landmarks(pose({ elbow })), 320, 640, t);
      time = t;
      assert.equal(previous.row.status, 'Measured');
      close(previous.row.elbow_deg, degrees, 1e-5);
      return previous;
    },
  };
}

test('ten samples calibrate; output matches the stacked Python geometry and elapsed summary', () => {
  const tracker = new PoseMetrics();
  const result = calibrate(tracker);
  assert.equal(result.row.status, 'Measured');
  assert.deepEqual(result.points, pose());
  for (const key of ['elbow_deg', 'shoulder_deg', 'hip_deg', 'knee_deg']) close(result.row[key], 180);
  for (const key of ['stack_offset', 'body_line_offset', 'leg_lever']) close(result.row[key], 0);
  close(result.row.torso_horizontal_deg, 90);
  close(result.row.shoulder_height, 200 / 150);
  assert.equal(result.row.partial_gravity_moment_nm, null);
  assert.equal(result.summary.analyzed_frames, 10);
  assert.equal(result.summary.measured_fraction, 0.1);
  assert.equal(result.summary.torso_reference_px, 150);
  assert.equal(result.summary.longest_tracked_posture_s, 0);
  assert.equal(result.summary.duration_s, 0.9);
});

test('angles and ratios use display pixels, not distorted normalized coordinates', () => {
  const pixels = pose({ elbow: [200, 500], hip: [130, 250] });
  const portrait = calibrate(new PoseMetrics(), landmarks(pixels));
  const landscape = calibrate(new PoseMetrics(), landmarks(pixels, 1280, 720), 1280, 720);
  close(portrait.row.elbow_deg, 90);
  for (const key of KEYS.filter(key => key !== 'partial_gravity_moment_nm')) {
    close(portrait.row[key], landscape.row[key]);
  }
  const transformed = Object.fromEntries(Object.entries(pixels).map(([name, xy]) => [name, xy.map(x => x * 2 + 50)]));
  const larger = calibrate(new PoseMetrics(), landmarks(transformed, 800, 1400), 800, 1400);
  for (const key of KEYS.filter(key => key !== 'partial_gravity_moment_nm')) close(portrait.row[key], larger.row[key]);
});

test('right-side selection and 180-degree detector mapping preserve original frame geometry', () => {
  const pixels = pose({ hip: [130, 250], elbow: [180, 500] });
  const normal = calibrate(new PoseMetrics(), landmarks(pixels));
  const rotated = calibrate(new PoseMetrics({ side: 'Right', detector_inverted: true }),
    landmarks(pixels, 320, 640, 'Right', true));
  for (const name of Object.keys(pixels)) {
    rotated.points[name].forEach((value, index) => close(value, pixels[name][index]));
  }
  for (const key of KEYS.filter(key => key !== 'partial_gravity_moment_nm')) close(rotated.row[key], normal.row[key]);
  const wrongSide = new PoseMetrics({ side: 'Right' }).update(landmarks(), 320, 640, 0);
  assert.equal(wrongSide.points, null);
  gaps(wrongSide.row);
});

test('planche geometry and optional illustrative moment match Python mass fractions', () => {
  const result = calibrate(new PoseMetrics({ skill: 'Planche', mass_kg: 70, torso_m: 0.5 }),
    landmarks(planche(), 640, 320), 640, 320);
  assert.equal(result.row.status, 'Measured');
  close(result.row.torso_horizontal_deg, 0);
  close(result.row.hip_deg, 180);
  close(result.row.partial_gravity_moment_nm, 70 * 9.81 * (0.5 * 75 + 0.2 * 200 + 0.09 * 300) * 0.5 / 150);
  for (const config of [{ mass_kg: 70 }, { torso_m: 0.5 }, { skill: 'Handstand push-up', mass_kg: 70, torso_m: 0.5 }]) {
    assert.equal(calibrate(new PoseMetrics(config)).row.partial_gravity_moment_nm, null);
  }
});

test('outside-posture and degenerate geometry retain all metric gaps', () => {
  const outside = calibrate(new PoseMetrics(), landmarks(planche(), 640, 320), 640, 320);
  assert.equal(outside.row.status, 'Outside selected posture');
  assert.notEqual(outside.points, null);
  gaps(outside.row);
  for (const pixels of [pose({ elbow: [100, 600] }), pose({ ankle: [100, 400] })]) {
    const result = calibrate(new PoseMetrics(), landmarks(pixels));
    assert.equal(result.row.status, 'Insufficient visibility');
    gaps(result.row);
  }
  const shortTorso = new PoseMetrics().update(landmarks(pose({ hip: [100, 396] })), 320, 640, 0);
  assert.equal(shortTorso.row.status, 'Insufficient visibility');
  assert.equal(shortTorso.summary.torso_reference_px, null);
});

test('posture boundaries are strict for inverted vertical ordering and planche inclination', () => {
  const verticalBoundary = calibrate(new PoseMetrics(), landmarks(pose({ ankle: [200, 250] })));
  assert.equal(verticalBoundary.row.status, 'Outside selected posture');
  const wristBoundary = calibrate(new PoseMetrics({ skill: 'Planche' }),
    landmarks({ ...planche(), wrist: [80, 100] }, 640, 320), 640, 320);
  assert.equal(wristBoundary.row.status, 'Outside selected posture');
  for (const [ratio, expected] of [[0.6499, 'Measured'], [0.6501, 'Outside selected posture']]) {
    const pixels = { ...planche(), hip: [100 + 150 * Math.sqrt(1 - ratio ** 2), 100 + 150 * ratio] };
    const result = calibrate(new PoseMetrics({ skill: 'Planche' }), landmarks(pixels, 640, 320), 640, 320);
    assert.equal(result.row.status, expected);
    if (expected !== 'Measured') gaps(result.row);
  }
});

test('visibility is required, presence is optional but enforced when supplied, thresholds inclusive', () => {
  for (const presence of [undefined, 0.6]) {
    const input = landmarks();
    for (const p of input.filter(Boolean)) {
      p.visibility = 0.6;
      if (presence !== undefined) p.presence = presence;
    }
    assert.equal(calibrate(new PoseMetrics(), input).row.status, 'Measured');
  }
  for (const change of [
    { visibility: 0.5999 }, { visibility: undefined }, { visibility: NaN }, { visibility: Infinity },
    { visibility: 1.01 }, { presence: 0.5999 }, { presence: null }, { presence: NaN },
    { x: -0.01 }, { x: 1.01 }, { x: Infinity }, { y: NaN },
  ]) {
    const tracker = new PoseMetrics();
    calibrate(tracker);
    tracker.update(landmarks(), 320, 640, 1);
    const input = landmarks();
    Object.assign(input[13], change);
    const missing = tracker.update(input, 320, 640, 1.1);
    assert.equal(missing.row.status, 'Insufficient visibility', JSON.stringify(change));
    assert.equal(missing.points, null);
    gaps(missing.row);
    const recovered = tracker.update(landmarks(pose({ elbow: [200, 500] })), 320, 640, 1.2);
    close(recovered.row.elbow_deg, 90); // no smoothing across confidence loss
    close(recovered.summary.longest_tracked_posture_s, 0.1);
  }
});

test('fixed calibration is the median of ten samples and does not follow later pose foreshortening', () => {
  const tracker = new PoseMetrics();
  let result;
  for (let index = 0; index < 10; index += 1) {
    result = tracker.update(landmarks(pose({ hip: [100, 400 - (110 + 10 * index)] })), 320, 640, index / 10);
  }
  close(result.summary.torso_reference_px, 155);
  result = tracker.update(landmarks(pose({ hip: [100, 300] })), 320, 640, 1.3);
  close(result.summary.torso_reference_px, 155);
  close(result.row.shoulder_height, 200 / 155);
  const fresh = new PoseMetrics().update(landmarks(), 320, 640, 0);
  assert.equal(fresh.summary.torso_reference_px, null);
  assert.equal(fresh.summary.analyzed_frames, 1);
  assert.equal(fresh.summary.full_reps, 0);
});

test('smoothing uses elapsed-time exponential weighting and output mutation cannot corrupt state', () => {
  const tracker = new PoseMetrics();
  const first = calibrate(tracker);
  first.points.elbow[0] = 9999;
  first.summary.torso_reference_px = 9999;
  const result = tracker.update(landmarks(pose({ elbow: [200, 500] })), 320, 640, 0.98);
  close(result.points.elbow[0], 100 + 100 * (1 - Math.exp(-1)));
  close(result.summary.torso_reference_px, 150);
  const afterGap = tracker.update(landmarks(pose({ elbow: [200, 500] })), 320, 640, 1.3);
  close(afterGap.points.elbow[0], 200);
});

test('hold uses elapsed time, accepts a 250ms boundary, and breaks on longer or missing gaps', () => {
  const tracker = new PoseMetrics();
  calibrate(tracker);
  tracker.update(landmarks(), 320, 640, 1);
  let result = tracker.update(landmarks(), 320, 640, 1.25);
  close(result.summary.longest_tracked_posture_s, 0.35);
  result = tracker.update(landmarks(), 320, 640, 1.500001);
  close(result.summary.longest_tracked_posture_s, 0.35);
  tracker.update(landmarks(), 320, 640, 1.6);
  result = tracker.update(null, 320, 640, 1.7);
  gaps(result.row);
  tracker.update(landmarks(), 320, 640, 1.8);
  result = tracker.update(landmarks(), 320, 640, 2);
  close(result.summary.longest_tracked_posture_s, 0.35);
  assert.equal(result.summary.analyzed_frames, 17);
  close(result.summary.measured_fraction, 7 / 17);
});

test('outside posture breaks hold continuity and smoothing, without discarding calibration', () => {
  const tracker = new PoseMetrics();
  calibrate(tracker);
  tracker.update(landmarks(), 320, 640, 1);
  const outside = tracker.update(landmarks(pose({ ankle: [100, 350] })), 320, 640, 1.2);
  assert.equal(outside.row.status, 'Outside selected posture');
  gaps(outside.row);
  const recovered = tracker.update(landmarks(), 320, 640, 1.3);
  close(recovered.points.ankle[1], 50);
  close(recovered.summary.longest_tracked_posture_s, 0.1);
  close(recovered.summary.torso_reference_px, 150);
});

test('HSPU needs sustained lockout, bend and return; short excursions cannot count', () => {
  const fixture = repFixture();
  fixture.at(1.1, 180);
  fixture.at(1.3, 90);
  assert.equal(fixture.at(1.419, 90).row.reps, 0);
  fixture.at(1.421, 90);
  fixture.at(1.6, 180);
  assert.equal(fixture.at(1.719, 180).row.reps, 0);
  const complete = fixture.at(1.721, 180);
  assert.equal(complete.row.reps, 1);
  assert.equal(complete.summary.full_reps, 1);
  fixture.at(1.9, 90);
  fixture.at(2, 180);
  assert.equal(fixture.at(2.2, 180).row.reps, 1);
});

test('HSPU threshold neighborhoods are 160 degree top and 100 degree bottom', () => {
  const fixture = repFixture(159.99);
  fixture.at(1.1, 159.99);
  fixture.at(1.3, 90);
  fixture.at(1.5, 90);
  fixture.at(1.7, 160.01);
  assert.equal(fixture.at(1.9, 160.01).row.reps, 0); // first sustained top only arms
  fixture.at(2.1, 100.01);
  fixture.at(2.3, 100.01);
  fixture.at(2.5, 160.01);
  assert.equal(fixture.at(2.7, 160.01).row.reps, 0);
  fixture.at(2.9, 99.99);
  fixture.at(3.1, 99.99);
  fixture.at(3.3, 159.99);
  assert.equal(fixture.at(3.5, 159.99).row.reps, 0);
  fixture.at(3.7, 160.01);
  assert.equal(fixture.at(3.9, 160.01).row.reps, 1);
});

test('confidence loss, null pose, gaps and dimension changes cannot finish interrupted reps', () => {
  for (const reset of ['confidence', 'missing', 'gap', 'dimensions']) {
    const fixture = repFixture();
    fixture.at(1.1, 180);
    fixture.at(1.3, 90);
    fixture.at(1.5, 90);
    let start = 1.7, width = 320, height = 640;
    if (reset === 'confidence') {
      const input = landmarks();
      input[13].visibility = 0.1;
      fixture.tracker.update(input, width, height, 1.6);
    } else if (reset === 'missing') {
      fixture.tracker.update(null, width, height, 1.6);
    } else if (reset === 'gap') {
      start = 1.8;
    } else {
      width = 640;
      height = 1280;
    }
    let result;
    for (let index = 0; index < 15; index += 1) {
      // Same normalized input intentionally doubles pixel geometry on resize.
      result = fixture.tracker.update(landmarks(), width, height, start + index / 10);
    }
    assert.equal(result.row.status, 'Measured');
    assert.equal(result.row.reps, 0, reset);
  }
});

test('dimension changes recalibrate scale while preserving finished reps and longest hold', () => {
  const fixture = repFixture();
  fixture.at(1.1, 180);
  fixture.at(1.3, 90);
  fixture.at(1.5, 90);
  fixture.at(1.7, 180);
  const completed = fixture.at(1.9, 180);
  assert.equal(completed.row.reps, 1);
  const resized = calibrate(fixture.tracker, landmarks(), 640, 1280, 2);
  close(resized.summary.torso_reference_px, 300);
  close(resized.summary.longest_tracked_posture_s, completed.summary.longest_tracked_posture_s);
  assert.equal(resized.summary.full_reps, 1);
  assert.equal(resized.summary.analyzed_frames, 25);
});

test('invalid config, confidence intervals, times and dimensions are rejected without consuming a frame', () => {
  for (const confidence of [-0.01, 1.01, NaN, Infinity, '0.6', null]) {
    assert.throws(() => new PoseMetrics({ confidence }), /Confidence/);
  }
  for (const confidence of [0, 1]) assert.equal(calibrate(new PoseMetrics({ confidence })).row.status, 'Measured');
  for (const config of [{ skill: 'Squat' }, { side: 'left' }, { side: '__proto__' }, { mass_kg: Infinity }, { torso_m: -1 }]) {
    assert.throws(() => new PoseMetrics(config), RangeError);
  }
  const tracker = new PoseMetrics();
  tracker.update(null, 320, 640, 0);
  for (const time of [0, -1, NaN, Infinity, '1']) assert.throws(() => tracker.update(null, 320, 640, time), /Frame times/);
  for (const [width, height] of [[0, 640], [320, 1], [320.5, 640], [Infinity, 640], [320, NaN]]) {
    assert.throws(() => tracker.update(null, width, height, 0.1), /Frame dimensions/);
  }
  const next = tracker.update(null, 320, 640, 0.00001);
  assert.equal(next.summary.analyzed_frames, 2);
  assert.equal(next.summary.duration_s, 0.00001);
  gaps(next.row);
});
