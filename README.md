# CRUX — local climbing video analysis PoC

CRUX analyzes one climber in a prerecorded bouldering clip. It exports an
annotated video, a JSON report and frame-level CSV data. The browser UI is in
Hebrew and all processing stays on the computer.

Implemented features:

- MediaPipe skeleton and a fading, smoothed hip-midpoint trajectory.
- Gap-aware Savitzky–Golay speed and jerk in image-diagonal units.
- Static, moving and unknown time. Untracked time is never counted as static.
- Explainable coaching cues: pauses, straight-arm use while static, movement
  rhythm, pelvis-path directness and leg/arm joint-motion signals during ascent.
- A contact timeline preserves every detected hand and foot color, including
  colors that differ from the estimated route.
- Stationary-wrist/ankle contact candidates with HSV masking and circular-hue
  k-means (`k=2`), one vote per distinct image position.
- Conservative route result: at least three distinct holds and 65% consistent
  weighted evidence are required; otherwise the result is `Unknown`.
- Optional climber crop, async progress/cancel, recent local analyses, an
  interactive graph, MP4/CSV/JSON downloads and explicit quality warnings.

## Run the application (PowerShell, Python 3.10+)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe setup_model.py
.venv\Scripts\python.exe setup_samples.py  # optional public test clips
.venv\Scripts\python.exe run_app.py
```

Open <http://127.0.0.1:8765>. The server deliberately binds only to the local
loopback interface. It accepts MP4, MOV, M4V or WebM clips up to 500 MB, five
minutes and 36,000 frames. A single CPU worker prevents concurrent MediaPipe
jobs from exhausting memory. Completed jobs are kept under `data/jobs/`.

## Desktop shortcut

After completing setup, create or repair the Windows desktop shortcut with:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_desktop_shortcut.ps1
```

`CRUX - ניתוח טיפוס` starts the server without a terminal window and opens the
application in the default browser. If CRUX is already running it simply opens
a new browser tab. Previous analyses remain visible because job metadata and
artifacts are stored under `data/jobs/`. Startup diagnostics are appended to
`data/logs/server.log`.

For command-line processing:

```powershell
.venv\Scripts\python.exe -m climb_app input/climb.mp4 --output output/my-climb
```

Optional arguments include normalized `--crop X0 Y0 X1 Y1`,
`--static-speed 0.012`, `--smoothing 0.55`, and `--model path.task`.

## What the numbers mean

The hip midpoint is a 2D pelvis proxy, not a measured whole-body center of
mass. Position is converted into fractions of the complete image diagonal;
speed is diagonals/second and jerk is diagonals/second³. These values allow
within-video comparison but are not physical SI measurements. Jerk is displayed
as a signal metric and is not converted into an invented "fluidity score".

The leg/arm percentage is a 2D kinematic association. During upward pelvis
phases it compares knee extension with elbow flexion. It is useful as a cue for
comparing similar attempts, but it is not a measurement of force, load, muscle
work or energy. Coaching items are only emitted when the underlying tracking
has enough coverage; they should be treated as video feedback, not diagnosis.

The application smooths every continuous pose segment separately and excludes
half a smoothing window at segment edges. Missing poses, improbable jumps,
camera motion, perspective and zoom can materially change the result. Use a
fixed camera, include the climber's full movement region, and compare attempts
recorded from the same viewpoint.

Route color is a heuristic. A stable wrist or ankle only suggests contact; the
40×40 ROI can contain skin, clothing, shoes, wall and several holds. Skin-like,
neutral and wall-like pixels are removed before clustering. White, black and
gray routes intentionally remain unsupported because a small local ROI cannot
separate them reliably from walls, shoes and chalk.

## Test and sample clips

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

The repository can expose three downloaded sample clips through the UI:
`body-trajectory-input.mp4`, `warp-dynamic-input.mp4` and
`warp-fixed-input.mp4`. They come from
[tommyjtl/climbing-analysis-toolbox](https://github.com/tommyjtl/climbing-analysis-toolbox)
under the MIT license. The sample binaries are ignored by Git.

`annotated.mp4` always uses the required OpenCV `mp4v` codec. `preview.mp4` is
a separate H.264 copy for browser playback. Both are silent because OpenCV does
not carry the source audio stream.

The earlier standalone Milestone 1 program remains available as
`climbing_m1.py`.

References: [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python),
[SciPy Savitzky–Golay filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html),
and [OpenCV HSV thresholding](https://docs.opencv.org/5.0/tutorials/imgproc/threshold_inRange/threshold_inRange.html).
