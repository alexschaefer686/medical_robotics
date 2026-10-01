"""Export Phantom 3's transverse, yaw-only and complete longitudinal views."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from view_transition_rl.simulation import PhantomEnv
from view_transition_rl.simulation.curved_arc import (
    TARGET_RANGE_RAD, VESSEL_RADIUS_MM, centerline, roll_yaw_pose, section_intersections,
)
from view_transition_rl.simulation.geometry import pose_from_euler


def central_length(intersects, step_mm):
    middle = len(intersects) // 2
    if not intersects[middle]:
        return 0.0
    left = right = middle
    while left > 0 and intersects[left - 1]:
        left -= 1
    while right + 1 < len(intersects) and intersects[right + 1]:
        right += 1
    return (right - left) * step_mm


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = PhantomEnv(phantom='curved_arc')
    env.export(output_dir / 'initial')
    t = np.linspace(*TARGET_RANGE_RAD, 301)
    target = centerline(t)
    rows, best = [], None
    for yaw in np.arange(-180, 180, .25):
        pose = pose_from_euler(0, 0, yaw=np.deg2rad(yaw))
        result = env.set_pose(pose)
        if not result.accepted:
            raise AssertionError(f"Boundary rejection would confound the example: {result.reason}")
        intersects = section_intersections(pose, t)
        length = central_length(intersects, .1)
        pixels = int(np.count_nonzero(env.get_mask()))
        rows.append({'yaw_deg': float(yaw), 'target_fraction_intersected': float(intersects.mean()),
                     'continuous_target_length_mm': length, 'foreground_pixels': pixels,
                     'complete_target': bool(intersects.all())})
        # Select the longest continuous arc containing the initial section;
        # pixel area is only a tie-breaker, never the longitudinal criterion.
        score = (length, pixels)
        if best is None or score > best[0]:
            best = (score, float(yaw))
    env.set_pose(pose_from_euler(0, 0, yaw=np.deg2rad(best[1])))
    env.export(output_dir / 'yaw_best')
    longitudinal = np.asarray(env.calibration['example_poses']['longitudinal'])
    result = env.set_pose(longitudinal)
    assert result.accepted, result.reason
    env.export(output_dir / 'longitudinal')
    assert section_intersections(longitudinal, t).all()
    projected = (target - longitudinal[:3, 3]) @ longitudinal[:3, :3]
    assert np.all(np.abs(projected[:, 0]) + VESSEL_RADIUS_MM < 20)
    assert np.all((projected[:, 2] - VESSEL_RADIUS_MM > 0)
                  & (projected[:, 2] + VESSEL_RADIUS_MM < 40))
    assert not any(row['complete_target'] for row in rows)
    with (output_dir / 'yaw_scan.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    endpoints = centerline(np.asarray(TARGET_RANGE_RAD))
    bound = min(abs(endpoints[1, 0]), abs(endpoints[1, 1]))
    summary = {
        'phantom_id': 'curved_arc', 'target_arc_length_mm': 30,
        'yaw_step_deg': .25, 'yaw_poses_checked': len(rows),
        'all_yaw_poses_legal': True, 'yaw_only_complete_target': False,
        'yaw_best_deg': best[1], 'yaw_best_continuous_target_length_mm': best[0][0],
        'analytic_minimum_max_endpoint_plane_distance_mm': float(bound),
        'vessel_radius_mm': VESSEL_RADIUS_MM,
        'longitudinal_complete_target': True,
        'longitudinal_pose': {'x_mm': 0, 'y_mm': 0, 'roll_deg': float(np.rad2deg(roll_yaw_pose())),
                              'pitch_deg': 0, 'yaw_deg': 90},
        'interpretation': 'Yaw-only cannot show the entire central 30 mm arc; Roll plus Yaw does so '
                          'without changing X or Y. Local longitudinal views remain possible.',
        'coverage_metric': 'Intersection with each analytic lumen cross-section, not mask aspect ratio. '
                           'Yaw failure for all angles also follows from the endpoint distance bound exceeding the radius.',
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    preview = Image.new('RGB', (1440, 530), '#182637')
    draw = ImageDraw.Draw(preview)
    for i, (name, label) in enumerate([
        ('initial', 'Initial: centered transverse section'),
        ('yaw_best', f'Yaw only: {best[1]:g} deg'),
        ('longitudinal', f'Roll={np.rad2deg(roll_yaw_pose()):.2f}, Yaw=90; X=Y=0'),
    ]):
        draw.text((i * 480 + 12, 18), label, fill='white')
        with Image.open(output_dir / f'{name}.png') as image:
            preview.paste(image.convert('RGB'), (i * 480, 50))
    preview.save(output_dir / 'comparison.png')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/simulation/curved_arc_demo'))
    print(json.dumps(run(parser.parse_args().output_dir), indent=2))
