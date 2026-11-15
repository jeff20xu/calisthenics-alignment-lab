"""Transparent 2-D geometry; no medical or net joint-torque inference."""
from dataclasses import dataclass
import math
import numpy as np

SIDES = {
    'Left': {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27},
    'Right': {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28},
}
METRICS = {
    'elbow_deg': 'Elbow angle (degrees)',
    'shoulder_deg': 'Shoulder opening (degrees)',
    'hip_deg': 'Hip angle (degrees)',
    'knee_deg': 'Knee angle (degrees)',
    'stack_offset': 'Wrist–shoulder horizontal offset / torso length',
    'body_line_offset': 'Hip deviation from shoulder–ankle line / torso length',
    'torso_horizontal_deg': 'Torso inclination from horizontal (degrees)',
    'leg_lever': 'Hip–ankle horizontal distance / torso length',
    'shoulder_height': 'Shoulder height above wrist / torso length',
    'partial_gravity_moment_nm': 'Illustrative trunk + leg gravitational moment (N m)',
}


def angle(a, b, c):
    u, v = a - b, c - b
    norm = np.linalg.norm(u) * np.linalg.norm(v)
    if norm < 1e-8:
        return math.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / norm, -1, 1))))


def measurements(p, torso_reference, mass_kg=0.0, torso_m=0.0):
    s, e, w, h, k, a = (p[n] for n in ('shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle'))
    if torso_reference < 1 or min(np.linalg.norm(x-y) for x,y in ((s,e),(e,w),(s,h),(h,k),(k,a))) < 1:
        return None
    line = a - s
    line_length = np.linalg.norm(line)
    if line_length < 1:
        return None
    cross = line[0] * (h-s)[1] - line[1] * (h-s)[0]
    result = {
        'elbow_deg': angle(s,e,w), 'shoulder_deg': angle(e,s,h),
        'hip_deg': angle(s,h,k), 'knee_deg': angle(h,k,a),
        'stack_offset': abs(float(s[0]-w[0])) / torso_reference,
        'body_line_offset': abs(float(cross)) / line_length / torso_reference,
        'torso_horizontal_deg': float(np.degrees(np.arctan2(abs(s[1]-h[1]), abs(s[0]-h[0])))),
        'leg_lever': abs(float(a[0]-h[0])) / torso_reference,
        'shoulder_height': float(w[1]-s[1]) / torso_reference,
        'partial_gravity_moment_nm': math.nan,
    }
    if mass_kg > 0 and torso_m > 0:
        # Deliberately simple model: segment midpoints, paired legs projected
        # onto the visible side, illustrative fractions of total body mass.
        weighted_arm = sum(f * float(((u+v)/2)[0]-s[0]) for f,u,v in ((.50,s,h),(.20,h,k),(.09,k,a)))
        result['partial_gravity_moment_nm'] = abs(mass_kg * 9.81 * weighted_arm * torso_m / torso_reference)
    return result


def in_posture(p, skill):
    s, w, h, a = (p[n] for n in ('shoulder','wrist','hip','ankle'))
    torso = max(float(np.linalg.norm(h-s)), 1)
    if skill == 'Planche':
        return s[1] < w[1] and abs(s[1]-h[1]) < .65 * torso
    return a[1] < h[1] < s[1] < w[1]


@dataclass
class RepCounter:
    """Lockout -> sustained bend -> sustained lockout. Missing pose resets cycle."""
    count: int = 0
    phase: str = 'Find lockout'
    candidate: str = ''
    since: float = 0.0
    last_time: float | None = None

    def update(self, t, elbow=None):
        if elbow is None or not math.isfinite(elbow):
            self.phase, self.candidate, self.last_time = 'Find lockout', '', None
            return self.count
        if self.last_time is not None and t-self.last_time > .25:
            self.phase, self.candidate = 'Find lockout', ''
        self.last_time = t
        target = 'top' if elbow >= 160 else 'bottom' if elbow <= 100 else ''
        if target != self.candidate:
            self.candidate, self.since = target, t
        if not target or t-self.since < .12:
            return self.count
        if target == 'top':
            if self.phase == 'Bent':
                self.count += 1
            self.phase = 'Ready'
        elif self.phase == 'Ready':
            self.phase = 'Bent'
        return self.count


def observations(df, skill, variant):
    usable = df[df['status'] == 'Measured']
    if usable.empty:
        return ['No reliable frames in the selected posture. Check camera angle, visible side, lighting, and full-body visibility.']
    notes = []
    median = usable.median(numeric_only=True)
    notes.append(f"Median elbow angle: {median['elbow_deg']:.1f}°. 180° represents straight-arm geometry, not a prescribed target for every frame.")
    if skill == 'Planche':
        notes.append(f"Median torso inclination: {median['torso_horizontal_deg']:.1f}° from horizontal; shoulder–wrist offset: {median['stack_offset']:.2f} torso lengths.")
        notes.append(f"Selected variation: {variant}. Tuck shapes are not penalized for bent hips or knees; side-view video cannot measure straddle width.")
    else:
        notes.append(f"Median shoulder opening: {median['shoulder_deg']:.1f}°; hip deviation from the shoulder–ankle line: {median['body_line_offset']:.2f} torso lengths.")
        notes.append(f"Median wrist–shoulder horizontal offset: {median['stack_offset']:.2f} torso lengths. Review frames with larger offsets rather than assuming they are errors.")
    if skill == 'Handstand push-up':
        notes.append('Rep counts require a sustained ≥160° elbow position, ≤100° bend, and return to ≥160°, with an inverted posture. Partial reps and tracking gaps do not count as full reps.')
    else:
        notes.append('Posture duration indicates tracked posture, not a verified unsupported hold. Walls, assistance, and floor contact are not detected.')
    return notes
