# Simulation: vessel phantoms and a virtual probe

This standalone geometric simulator places a reconstructed vessel inside a rectangular phantom and generates binary segmentation masks from virtual probe poses. It provides a PyVista/VTK desktop interface and a headless Python API. Run the commands below from the project root.

Runtime resources are bundled in `src/view_transition_rl/simulation/assets`: STL meshes, 190 original poses, 190 original masks, metadata, and reconstruction reports. No symbolic links, original GPIS project, ROS, robot, or segmentation network are required.

## Installation and GUI

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[gui]'
view-transition-rl sim gui
```

With dependencies already installed, source execution works with `PYTHONPATH=src python3 -m view_transition_rl sim gui`. To reuse system VTK/PyVista installations, an environment may be created with `python3 -m venv --system-site-packages .venv`. Core dependencies are NumPy, VTK, and Pillow; the desktop interface additionally uses PyVista, without requiring Qt or GPU compute libraries. The shared folder contains no preconfigured environment.

The window shows the 3D phantom, vessel, probe, and slice on the left, the current mask at upper right, and pose sliders and reset controls below. GUI lengths are millimetres and angles are degrees; Python motion increments use radians.

| Control | Behavior |
| --- | --- |
| Mouse drag / wheel | Rotate / zoom the 3D view. |
| Surface X / Y sliders | Set the probe center in fixed phantom surface coordinates. |
| Roll / Pitch / Yaw sliders | Set absolute Euler angles, `R = Rz(yaw) Ry(pitch) Rx(roll)`. |
| Left / right arrows | Move −/+0.5 mm along the surface projection of the probe's local x axis. |
| Down / up arrows | Move −/+0.5 mm along the surface projection of the probe's local y axis. |
| A / D, W / S, Q / E | Rotate −/+1° around local x, y, and z respectively. |
| RESET TO ZERO / R | Zero X, Y, Roll, Pitch, and Yaw; return the center to the phantom surface origin. |
| P | Export PNG mask and JSON pose under `outputs/simulation/captures`. |

Click the 3D view before using the keyboard. Letter shortcuts accept either case. VTK's built-in character shortcuts are disabled: Q/E only rotate, W/S do not change wireframe display, and R only resets pose. Close the window to exit. Invalid moves display a rejection reason and restore controls to the actual pose. Slider requests are coalesced to the latest request and processed approximately every 33 ms.

## Geometry and contact model

- Reference phantom dimensions are inferred as **70×100×40 mm**. The top is `Z=0`, the origin is its center, and the right-handed +Z axis points inward. Width and length use the combined scan-footprint and vessel bounds, with 10 mm margins rounded upward to 10 mm. The 40 mm depth is an assumption, not a measurement.
- The top normal uses the mean probe +z direction across 190 frames; X uses the mean +x direction, orthogonalized. Maximum scan-center plane residual is **0.04334 mm**, RMS **0.00787 mm**. Probe orientations define the normal because the nearly straight scan path makes plane fitting from positions ambiguous.
- A shared rigid transform preserves vessel size and relative position. Depth is approximately **7.93–17.93 mm**. The original STL is not resmoothed or rescaled; the world transform is in `assets/calibration.json`.
- The probe origin is its bottom center: local x follows the 40 mm long side, y the 10 mm short side, and z points downward. `assets/probe_model_50mm.stl` supplies the visible shell. Its source is assumed to use millimetres, its contact footprint is scaled to **40×10 mm**, and it is cropped to 50 mm above the base and capped. The upper shell can be wider; constraints still use the defined footprint. Transforms and hashes are in `assets/probe_model.json`, and the source mesh is retained. This shell affects appearance only, not masks, collision constraints, or acoustics.
- Zero, tilted, and rotated probe screenshots are under `reports/simulation/`, named `probe_model_zero_pose.png`, `probe_model_tilted.png`, and `probe_model_yaw.png`.
- The center stays on the surface. Local x/y tilt may put part of the base inside the phantom as a simplified soft-contact model. Pressure, deformation, and coupling loss are not modeled; the whole base need not lie flat.
- The probe z axis must stay within 60° of the inward normal, and the surface projections of all four footprint corners must remain inside the phantom. Rejected moves leave the actual pose and mask unchanged.
- The imaging plane is local `y=0`, with **40×40 mm** field of view and **480×480** pixels by default. Columns follow +x, rows +z, and the top-edge midpoint is the probe origin. Pixel centers are `x=(c+0.5−W/2)×40/W`, `z=(r+0.5)×40/H`.
- Each slice is computed directly from the closed STL and rasterized, supporting holes and multiple regions. Output is uint8: vessel 255, background 0. No intersection yields zeros, and pixels outside the phantom are zeroed. This is ideal segmentation geometry, without B-mode intensity or finite slice thickness.
- The reconstruction is closed at the scan ends; unobserved vessel segments are not extrapolated. Original masks are for replay comparison and evaluation, not a lookup table for new poses.

## Python API

After installation, import directly; otherwise set `PYTHONPATH=src`.

```python
import numpy as np
from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.geometry import pose_from_euler

env = PhantomEnv()
result = env.set_pose(pose_from_euler(0, 0, roll=np.deg2rad(15)))
assert result.accepted, result.reason

# [dx_mm, dy_mm, rx_rad, ry_rad, rz_rad]
result = env.step([1.0, 0.0, 0.0, np.deg2rad(2), 0.0])
if not result.accepted:
    print(result.reason)

mask = env.get_mask()  # independent uint8 array, shape (480, 480)
state = env.get_state()  # actual pose, dimensions, contact/constraint state
env.export("outputs/simulation/example")  # example.png and example.json
env.reset()
```

`set_pose(T)` accepts a right-handed 4×4 rigid `phantom_T_probe` matrix with translation in **mm**. The bottom-center z coordinate must be zero. Invalid matrices, excessive tilt, or boundary violations return `StepResult(accepted=False, reason=..., state=...)`; actions are not clipped automatically.

`step(action)` first projects the **pre-action** local x/y directions onto the surface, normalizes them separately, and translates; it then applies `R_new = R_old @ Rx(rx) @ Ry(ry) @ Rz(rz)`. Projected directions need not be orthogonal. The complete action is accepted only if its final pose is legal; continuous collision paths are not simulated.

`get_state()` returns JSON-serializable data including `phantom_T_probe_mm` and `world_T_probe_mm`. Original CSV translations are in metres and have already been converted to millimetres during import/loading. Original poses are already `world_T_image`, so probe calibration offsets are not reapplied.

`replay_frame(i)` projects an original frame center onto the contact plane while retaining its rotation. `reference_pose(i)` retains the acquisition pose for independent evaluation. `env.slicer.render(pose, env.config.phantom_size_mm)` renders that pose without changing environment state.

This simulator is not itself a Gym environment and has no reward, termination, or policy. Instances are not guaranteed thread-safe; create a separate instance per worker for parallel sampling.

## Headless commands and configuration

```bash
PYTHONPATH=src python3 -m view_transition_rl sim info
PYTHONPATH=src python3 -m view_transition_rl sim sample --frame 95 --output outputs/simulation/sample
PYTHONPATH=src python3 -m view_transition_rl sim sample --pose-json outputs/simulation/example.json --output outputs/simulation/resampled
PYTHONPATH=src python3 -m view_transition_rl sim evaluate --output outputs/simulation/evaluation
PYTHONPATH=src python3 -m view_transition_rl sim gui --screenshot outputs/simulation/preview.png
PYTHONPATH=src python3 -m view_transition_rl sim --config configs/simulation/default.json gui
```

Sampling, the geometric Python API, and reference evaluation do not create OpenGL windows. Offscreen screenshots still require a working VTK OpenGL/EGL backend.

Place `--config` before the simulator subcommand. It can override phantom/probe dimensions, image shape, field of view, or maximum tilt. Configuration does not move or scale the vessel; the phantom must contain the vessel and initial probe. Python also accepts `PhantomConfig`. Reference evaluation rejects mismatched image shape or field of view to avoid invalid comparisons.

## Verification and distribution

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PHANTOM_TEST_GUI=1 PYTHONPATH=src python3 -m unittest discover -s tests/simulation -p test_gui_keyboard.py -v
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
python3 scripts/verification/verify_install.py dist/view_transition_rl-0.1.0-py3-none-any.whl --gui
```

Installation verification installs the wheel in a temporary directory and runs from a temporary working directory. Python audit hooks prohibit access to the source tree and original GPIS directory while checking the API, 190-frame evaluation, exports, and optional GUI/offscreen screenshots. It does not modify the original acquisition directory.

`reports/simulation/reference_evaluation` contains the saved 190-frame comparison at original poses: mean **Dice=0.9567**, **IoU=0.9177**. JSON/CSV reports contain per-frame metrics and uncached slicing times. Reconstruction and smoothing change boundaries, so pixel identity is not expected. Comparison colors are red for reference-only pixels, green for simulated-only pixels, and yellow for overlap.

`assets/provenance.json` records original sources and SHA-256 hashes; original directory strings are provenance only. `scripts/simulation/import_scan.py` is a one-time packaging tool, unnecessary for ordinary use, and refuses to overwrite an existing asset directory. Runtime code has no dependency on the original GPIS code.

## Phantom selection

`PhantomEnv()` defaults to `reference`. Select the second model with `PhantomEnv(phantom="vessel_no_hole")` or `python -m view_transition_rl sim --phantom vessel_no_hole gui`.

The GUI button cycles PHANTOM 1 (Reference), PHANTOM 2 (Vessel no hole), PHANTOM 3 (Curved arc), then PHANTOM 1. Switching clears pending motion and uses the selected model's initial pose. The added models start at the identity matrix. R and RESET TO ZERO zero the current model's probe pose.

The second vessel resource is copied unchanged from root-level `vessel_no_hole.STL`, interpreted in millimetres without scaling. Original X maps to phantom Y, original Z to phantom X, and original Y to phantom Z; a +8° rotation about phantom X tilts it downward (+Z) along +Y. It is then recentered within the surface. Minimum cover is 8 mm, vessel depth approximately 8–36.36 mm, and phantom size **70×190×40 mm**. Calibration JSON records the transform, source hash, and assumptions.

`--config` overrides the selected model's defaults. Python supports `PhantomEnv(phantom="vessel_no_hole", config_overrides={"image_shape": [240, 240]})`, or a complete `PhantomConfig`, but not both. GUI switching preserves explicit overrides; every field of a complete configuration counts as explicit. If dimensions cannot accommodate the new model, switching fails and retains the current environment. `load_config(path, phantom="vessel_no_hole")` resolves configuration using that model's defaults.

State/export JSON includes `phantom_id`. `sample --pose-json` rejects mismatched model identifiers. Older state files without an identifier and pure matrices use the currently selected model. Additional phantoms have no reference acquisition frames, so `reference_pose`, `replay_frame`, `sample --frame`, and `evaluate` report unsupported operations. The reference phantom retains all 190 reference frames.

## Phantom 3: tilted circular-arc vessel

Select `--phantom curved_arc` or `PhantomEnv(phantom="curved_arc")`. `scripts/simulation/generate_curved_arc.py` generates the mesh in millimetres, with identical world and phantom coordinates. Phantom dimensions are **100×100×40 mm**, using the same 40×10 mm probe, 40×40 mm field of view, and 480×480 image. Generation uses 401 axial samples, 96 circumferential samples, and planar end caps, checking degenerate faces, orientation, and closure.

Vessel radius is 2 mm, with centerline:

```text
C(t) = (20*(1-cos(t))/sqrt(2), 20*sin(t), 20+20*(1-cos(t))/sqrt(2))
Full vessel: t in [-1, 1], arc length 40 mm
Target segment: t in [-0.75, 0.75], arc length 30 mm
Centerline plane: z - x = 20 mm
```

The identity initial pose places the transverse-section center at image lateral position 0 mm and depth 20 mm. With X=Y=Roll=Pitch=0, every Yaw plane passes through the vertical axis. Target endpoints are `(a, ±b, 20+a)`, where `a≈3.79449 mm`, `b≈13.63278 mm`. For unit horizontal normal `(nx,ny,0)`, the maximum endpoint-to-plane distance is `a*abs(nx)+b*abs(ny)`. Its minimum is `min(a,b)=3.79449 mm`, larger than the 2 mm lumen radius. Every Yaw therefore misses the lumen cross-section at at least one target endpoint. All zero-Roll/Pitch Yaw poses meet contact constraints, and the target is far from the caps: the limitation is geometric.

A successful Roll/Yaw-only comparison keeps `X=Y=Pitch=0`, with `Yaw=90°` and `Roll≈4.96°` as calculated in calibration JSON. The plane need not contain the centerline but intersects the lumen cross-section everywhere along the target, without probe translation.

**This counterexample concerns the complete 30 mm segment, not the existence of local longitudinal views.** Both kinds of slice may be off-center. Phantom 3 demonstrates full target intersection with Roll+Yaw where Yaw alone misses at least one target endpoint section.

```bash
# Regenerate assets (writes into package assets by default)
python scripts/simulation/generate_curved_arc.py
# Export initial, best Yaw-only, and successful longitudinal examples
python examples/simulation/curved_arc_counterexample.py
# Replay the successful demonstration pose
python -m view_transition_rl sim --phantom curved_arc sample \
  --pose-json outputs/simulation/curved_arc_demo/longitudinal.json \
  --output outputs/simulation/curved_arc_replay
```

The demonstration scans the full Yaw range at 0.25° spacing and exports `yaw_scan.csv`, `summary.json`, and `comparison.png`. Coverage uses analytical lumen cross-section intersections, not mask aspect ratio; continuous length is estimated at 0.1 mm sampling. The endpoint-distance bound establishes failure across all angles independently of the discrete scan. Calibration JSON includes generation parameters, hashes, and `example_poses.initial / longitudinal` 4×4 poses. Reference replay and evaluation are unsupported because no acquisition data exist for this model.
