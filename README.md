# Calisthenics Alignment Lab

Analyze handstands, handstand push-ups, and planches from **uploaded videos or a live browser camera**. Inspect joint stacking, body alignment, movement consistency, and geometric leverage with MediaPipe pose estimation.

Repository: https://github.com/jeff20xu/calisthenics-alignment-lab

## Features

- Handstand wrist–shoulder stacking, shoulder opening, elbow extension, and body-line measurements.
- Handstand push-up elbow/shoulder trajectories and sustained full-repetition counting.
- Planche shoulder lean, torso inclination, hip/knee angles, and leverage measurements, with a selected variation label.
- Live annotated video, latest measurements, tracked-posture timing, and camera Start/Stop controls.
- Uploaded-video charts and annotated MP4, CSV, and JSON downloads.
- Explicit camera consent; audio disabled; no live recording.

**Not a validated coaching or medical system.** The app measures projected 2-D geometry. It does not determine optimal technique, injury risk, or actual joint torque. Inspect the skeleton overlay before interpreting measurements.

## Run locally

Python 3.13 is the verified development runtime. From the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --server.port 8511
```

After installation, Windows users can double-click `Launch.bat`. Open http://127.0.0.1:8511. On Linux/macOS, use `python3 -m venv .venv`, activate `.venv`, install the same requirements, then run `streamlit run app.py`.

The official full pose model downloads into `models/` on first analysis and is reused. Model weights, private videos, virtual environments, and credentials are excluded from version control.

## Live camera

1. Film one athlete from the side using a fixed, level camera; keep wrists through ankles visible.
2. Select the skill, camera-facing anatomical side, and planche variation if appropriate.
3. Choose **Live camera** and read the server-processing notice.
4. Give consent, press **Start**, and allow browser camera access. Audio is never requested.
5. Inspect the returned joint overlay and latest measurements.
6. Press **Stop** when finished. Restarting begins a new analysis. Changing analysis settings resets calibration and counts. Switching to uploads revokes consent and stops the live connection.

A camera requires HTTPS or localhost. The default public STUN server is sufficient for some networks, not all. A hosted deployment or restrictive network may require TURN; see deployment below. The app processes only the latest queued frame, rather than accumulating an ever-growing inference backlog. It does not record the live stream.

## Video upload

Choose **Video upload** and upload an MP4/MOV/AVI/WebM clip of at most 3 minutes and 200 MB. Select **Analyze video**, review the overlay and charts, and download results. MP4 exports have no audio.

Both modes use the same `FrameTracker` and measurement formulas. Uploaded videos are sampled at no more than 15 fps and resized to at most 960 pixels in either dimension. Timing assumes constant-frame-rate input; convert variable-frame-rate clips before analysis when precise timing matters.

## Privacy and public hosting

On localhost, your own computer processes the frames. On a hosted app, **uploaded videos and live frames are sent to the app server**. Inference does not call a separate remote AI service. Upload temporary files are deleted after analysis; preview/export results remain in the session's server memory until released. Live processing retains the latest measurements rather than a recording. Only use footage and camera views you have permission to share with the host.

### Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with GitHub.
2. Create an app from `jeff20xu/calisthenics-alignment-lab`, branch `main`, entrypoint `app.py`.
3. Choose Python **3.13** under Advanced settings.
4. Deploy and share the resulting HTTPS `streamlit.app` URL. Configure public viewer access if prompted.

`requirements.txt`, `packages.txt`, and `.streamlit/config.toml` are included. GitHub itself hosts the source code, not this Python application's running server. Free hosting resources are shared and limited; multiple simultaneous pose streams may be slow.

### Optional TURN configuration

Set `ICE_SERVERS` to a JSON array in an environment variable or Streamlit secrets. Include your provider's STUN/TURN URLs and limited-use TURN credentials. Never put API keys or a real `secrets.toml` in GitHub. `.streamlit/secrets.toml` and `.env` are ignored.

Example **format only**, using reserved example domains, not a working relay:

```toml
ICE_SERVERS = '[{"urls":["stun:stun.l.google.com:19302"]},{"urls":["turns:relay.example.org:443?transport=tcp"],"username":"YOUR_RELAY_USERNAME","credential":"YOUR_RELAY_PASSWORD"}]'
```

TURN connection credentials must be delivered to the browser for WebRTC to connect. Use restricted, short-lived relay credentials, not account API keys. No TURN service or credentials are bundled, and worldwide live connectivity cannot be guaranteed by STUN alone.

## Measurement definitions

- Joint angles: elbow, shoulder, hip, and knee in image coordinates.
- Stack offset: absolute shoulder–wrist horizontal distance divided by calibrated torso length.
- Body-line deviation: hip distance from the shoulder–ankle line, normalized by torso length.
- Torso inclination: angle relative to image horizontal.
- Leg leverage: hip–ankle horizontal distance divided by torso length.
- Shoulder height: wrist-to-shoulder vertical difference normalized by torso length.
- Posture time: elapsed intervals between consecutive reliable frames matching the selected coarse posture. First measured frame adds zero time. Missing measurements or gaps over 250 ms break continuity.
- Full HSPU rep: sustained elbow angle >=160°, then <=100°, then >=160°, with an inverted posture. Each endpoint persists for at least 0.12 seconds. Partial reps and cycles interrupted by tracking loss do not count.

The first ten reliable sampled frames establish median torso scale. Keep the torso side-on and unforeshortened. Dimension changes restart calibration. A short time-based smoother resets after lost tracking. Low-confidence/out-of-frame joints create missing measurements, not fabricated positions.

The posture check is a coarse geometry filter, not a trained skill classifier. Rotated standing poses can satisfy its geometry. Assistance, walls, ground contact, and straddle width are not detected. Tuck variations are labeled rather than penalized against a full-planche template.

## Optional illustrative gravitational-moment model

This is **not net shoulder torque, muscle force, or a training prescription**. It is disabled for dynamic push-ups.

The model estimates the magnitude of the summed gravitational moment of the trunk and paired thigh/shank segments about the visible shoulder. Assumptions:

- Trunk = 50%, paired thighs = 20%, paired shanks = 9% of entered body mass.
- Segment mass is concentrated at its midpoint.
- The visible leg represents both legs in projection.
- Entered shoulder-to-hip length supplies the image scale.
- Arms, head, feet, contact forces, and acceleration are excluded.

Formula: `abs(mass * 9.81 * sum(segment_fraction * signed_horizontal_midpoint_offset) * metres_per_pixel)`.

These educational assumptions are not individualized anthropometric measurements. Lower estimated demand is not inherently better: increasing leverage is part of planche progression.

## Verification and limitations

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

15 geometry/tracker regression tests passed locally, covering angle geometry, scale invariance, the explicit gravity formula, repetition transitions, tracking-loss resets, actual elapsed hold time, submillisecond timestamp ordering, dimension changes, and resource cleanup.

Browser smoke verification used an isolated fake camera fed a rotated public pose image, not a person's physical webcam. It exercised real browser WebRTC, MediaPipe inference, returned annotated video, live measurements, audio-disabled tracks, Stop/Start, and the live-to-upload transition. Uploaded-video analysis remained functional after live shutdown. These fixtures demonstrate pipeline operation, not calisthenics accuracy.

No genuine handstand/push-up/planche dataset has been evaluated. Before reporting accuracy or making training recommendations, collect consenting side-view clips, manually annotate joints/repetitions, and evaluate angle error, tracking coverage, timing, and failure cases on held-out footage.

## Project structure

- `app.py`: Streamlit modes, consent, charts, and downloads.
- `analyzer.py`: shared `FrameTracker`, model acquisition, and uploaded-video export.
- `live.py`: per-connection WebRTC processing, thread-safe measurements, and ICE configuration.
- `biomechanics.py`: joint geometry, posture filtering, and repetition state machine.
- `test_biomechanics.py`, `test_tracker.py`: deterministic behavior regressions.
- `packages.txt`, `.streamlit/config.toml`: public hosting configuration.

## References

- [MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python)
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc)
- [Streamlit deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Official pose model](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task)
- [Smoke-test image](https://storage.googleapis.com/mediapipe-assets/pose.jpg)
