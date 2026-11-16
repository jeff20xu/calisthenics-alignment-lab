// Pixel-space equivalents of biomechanics.py and analyzer.FrameTracker.
const SIDES = {
  Left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27 },
  Right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28 },
};
const NAMES = Object.keys(SIDES.Left);
const METRICS = [
  'elbow_deg', 'shoulder_deg', 'hip_deg', 'knee_deg', 'stack_offset',
  'body_line_offset', 'torso_horizontal_deg', 'leg_lever', 'shoulder_height',
  'partial_gravity_moment_nm',
];
const GAP_SECONDS = 0.25;
const SUSTAIN_SECONDS = 0.12;
const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function angle(a, b, c) {
  const ux = a[0] - b[0], uy = a[1] - b[1];
  const vx = c[0] - b[0], vy = c[1] - b[1];
  const norm = Math.hypot(ux, uy) * Math.hypot(vx, vy);
  if (norm < 1e-8) return null;
  return Math.acos(Math.max(-1, Math.min(1, (ux * vx + uy * vy) / norm))) * 180 / Math.PI;
}

function measurements(p, reference, mass, torsoMeters) {
  const { shoulder: s, elbow: e, wrist: w, hip: h, knee: k, ankle: a } = p;
  if (reference < 1 || [[s, e], [e, w], [s, h], [h, k], [k, a]].some(([u, v]) => distance(u, v) < 1)) {
    return null;
  }
  const lineLength = distance(a, s);
  if (lineLength < 1) return null;
  const cross = (a[0] - s[0]) * (h[1] - s[1]) - (a[1] - s[1]) * (h[0] - s[0]);
  const result = {
    elbow_deg: angle(s, e, w),
    shoulder_deg: angle(e, s, h),
    hip_deg: angle(s, h, k),
    knee_deg: angle(h, k, a),
    stack_offset: Math.abs(s[0] - w[0]) / reference,
    body_line_offset: Math.abs(cross) / lineLength / reference,
    torso_horizontal_deg: Math.atan2(Math.abs(s[1] - h[1]), Math.abs(s[0] - h[0])) * 180 / Math.PI,
    leg_lever: Math.abs(a[0] - h[0]) / reference,
    shoulder_height: (w[1] - s[1]) / reference,
    partial_gravity_moment_nm: null,
  };
  if (mass > 0 && torsoMeters > 0) {
    // Same illustrative mass fractions and projected segment midpoints as Python;
    // this is not net joint torque and deliberately excludes the arms and head.
    const weightedArm = 0.50 * ((s[0] + h[0]) / 2 - s[0])
      + 0.20 * ((h[0] + k[0]) / 2 - s[0])
      + 0.09 * ((k[0] + a[0]) / 2 - s[0]);
    const moment = Math.abs(mass * 9.81 * weightedArm * torsoMeters / reference);
    result.partial_gravity_moment_nm = Number.isFinite(moment) ? moment : null;
  }
  return METRICS.slice(0, -1).every(key => Number.isFinite(result[key])) ? result : null;
}

function inPosture(p, skill) {
  const { shoulder: s, wrist: w, hip: h, ankle: a } = p;
  if (skill === 'Planche') {
    return s[1] < w[1] && Math.abs(s[1] - h[1]) < 0.65 * Math.max(distance(h, s), 1);
  }
  return a[1] < h[1] && h[1] < s[1] && s[1] < w[1];
}

/**
 * Bounded, browser-independent tracker. Landmarks are the single-pose array from
 * MediaPipe Tasks JS: {x, y, visibility, presence?}. Visibility is required;
 * presence is checked when supplied (Tasks JS may omit it). Coordinates must
 * be finite and normalized to [0, 1]. Inverted input is mapped back to display
 * pixels before smoothing and measurement. Dimensions describe that display.
 *
 * update() requires finite, nonnegative, strictly increasing elapsed seconds.
 * Invalid arguments throw without consuming a frame. Missing pose data is null,
 * not an error. Metric gaps are null, never stale values or NaN. Ten valid torso
 * samples establish a fixed median scale. Dimension changes recalibrate and
 * break continuity, retaining cumulative counts. A fresh instance resets all.
 */
export class PoseMetrics {
  #config;
  #width = null;
  #height = null;
  #lastTime = null;
  #previous = null;
  #lastValid = null;
  #lastMeasured = null;
  #referenceSamples = [];
  #torsoReference = null;
  #phase = 'Find lockout';
  #candidate = '';
  #since = 0;
  #reps = 0;
  #currentHold = 0;
  #longest = 0;
  #analyzedFrames = 0;
  #measuredFrames = 0;

  constructor({ skill = 'Handstand', side = 'Left', variant = 'Full', confidence = 0.6,
    detector_inverted = false, mass_kg = 0, torso_m = 0 } = {}) {
    if (!['Handstand', 'Handstand push-up', 'Planche'].includes(skill)
      || !Object.hasOwn(SIDES, side)) {
      throw new RangeError('Choose a supported skill and visible body side.');
    }
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
      throw new RangeError('Confidence must be finite and between 0 and 1.');
    }
    if (!Number.isFinite(mass_kg) || mass_kg < 0 || !Number.isFinite(torso_m) || torso_m < 0) {
      throw new RangeError('Mass and torso length must be finite and nonnegative.');
    }
    this.#config = { skill, side, variant, confidence, detector_inverted: Boolean(detector_inverted), mass_kg, torso_m };
  }

  #resetContinuity() {
    this.#previous = this.#lastValid = this.#lastMeasured = null;
    this.#currentHold = 0;
    this.#phase = 'Find lockout';
    this.#candidate = '';
  }

  #updateReps(t, elbow) {
    const target = elbow >= 160 ? 'top' : elbow <= 100 ? 'bottom' : '';
    if (target !== this.#candidate) {
      this.#candidate = target;
      this.#since = t;
    }
    if (!target || t - this.#since < SUSTAIN_SECONDS) return;
    if (target === 'top') {
      if (this.#phase === 'Bent') this.#reps += 1;
      this.#phase = 'Ready';
    } else if (this.#phase === 'Ready') {
      this.#phase = 'Bent';
    }
  }

  update(landmarks, width, height, timeSeconds) {
    const t = timeSeconds;
    if (!Number.isFinite(t) || t < 0 || (this.#lastTime !== null && t <= this.#lastTime)) {
      throw new RangeError('Frame times must be finite, nonnegative, and strictly increasing.');
    }
    if (!Number.isSafeInteger(width) || width < 2 || !Number.isSafeInteger(height) || height < 2) {
      throw new RangeError('Frame dimensions must be integer pixel sizes of at least 2.');
    }
    if (this.#width !== null && (width !== this.#width || height !== this.#height)) {
      this.#resetContinuity();
      this.#referenceSamples.length = 0;
      this.#torsoReference = null;
    } else if (this.#lastTime !== null && t - this.#lastTime > GAP_SECONDS) {
      this.#resetContinuity();
    }
    this.#width = width;
    this.#height = height;
    this.#lastTime = t;
    const row = { time_s: t, status: 'Insufficient visibility', reps: this.#reps };
    for (const key of METRICS) row[key] = null;
    let points = null;
    const config = this.#config;
    const selected = SIDES[config.side];
    const validConfidence = value => Number.isFinite(value) && value >= config.confidence && value <= 1;
    if (Array.isArray(landmarks) && NAMES.every(name => {
      const p = landmarks[selected[name]];
      return p && validConfidence(p.visibility)
        && (p.presence === undefined || validConfidence(p.presence))
        && Number.isFinite(p.x) && p.x >= 0 && p.x <= 1
        && Number.isFinite(p.y) && p.y >= 0 && p.y <= 1;
    })) {
      points = {};
      const alpha = this.#previous === null ? 1 : 1 - Math.exp(-(t - this.#lastValid) / 0.08);
      for (const name of NAMES) {
        const p = landmarks[selected[name]];
        const x = (config.detector_inverted ? 1 - p.x : p.x) * width;
        const y = (config.detector_inverted ? 1 - p.y : p.y) * height;
        points[name] = this.#previous === null ? [x, y] : [
          alpha * x + (1 - alpha) * this.#previous[name][0],
          alpha * y + (1 - alpha) * this.#previous[name][1],
        ];
      }
      // Retain only one frame, isolated from the mutable output handed to callers.
      this.#previous = Object.fromEntries(NAMES.map(name => [name, [...points[name]]]));
      this.#lastValid = t;
      const torso = distance(points.hip, points.shoulder);
      if (torso >= 5) {
        if (this.#torsoReference === null) {
          this.#referenceSamples.push(torso);
          if (this.#referenceSamples.length === 10) {
            this.#referenceSamples.sort((a, b) => a - b);
            this.#torsoReference = (this.#referenceSamples[4] + this.#referenceSamples[5]) / 2;
            this.#referenceSamples.length = 0;
          }
        }
        if (this.#torsoReference === null) {
          row.status = 'Calibrating torso scale';
        } else {
          const values = measurements(points, this.#torsoReference,
            config.skill === 'Handstand push-up' ? 0 : config.mass_kg, config.torso_m);
          if (values !== null) {
            row.status = inPosture(points, config.skill) ? 'Measured' : 'Outside selected posture';
            if (row.status === 'Measured') Object.assign(row, values);
          }
        }
      }
    }
    if (row.status === 'Measured') {
      if (this.#lastMeasured !== null) this.#currentHold += t - this.#lastMeasured;
      this.#lastMeasured = t;
      this.#longest = Math.max(this.#longest, this.#currentHold);
      this.#measuredFrames += 1;
      if (config.skill === 'Handstand push-up') this.#updateReps(t, row.elbow_deg);
    } else {
      this.#resetContinuity();
    }
    row.reps = this.#reps;
    this.#analyzedFrames += 1;
    return {
      row,
      points,
      summary: {
        skill: config.skill, side: config.side, variant: config.variant,
        duration_s: t, analyzed_frames: this.#analyzedFrames,
        measured_fraction: this.#measuredFrames / this.#analyzedFrames,
        full_reps: this.#reps, longest_tracked_posture_s: this.#longest,
        torso_reference_px: this.#torsoReference,
        detector_inverted: config.detector_inverted, confidence_threshold: config.confidence,
        mass_kg: config.mass_kg, torso_m: config.torso_m,
      },
    };
  }
}
