#!/usr/bin/env python3
# Compare ob_in_cam pose trajectories from two BundleSDF runs on the same
# video (e.g. EfficientLoFTR vs classic LoFTR backend) as a relative
# consistency check.
#
# The milk-carton demo has no ground-truth pose, so this script cannot
# compute absolute tracking error (ADD/ADD-S). It only reports how much the
# two runs' per-frame poses disagree with each other. For GT-based ADD/ADD-S,
# use benchmark_ho3d.py against the HO3D dataset (which ships GT poses)
# instead.
#
# Usage:
#   python3 scripts/compare_poses.py \
#       --out_folder_a /workspace/data/out_milk_eloftr \
#       --out_folder_b /workspace/data/out_milk_loftr \
#       --csv_out /workspace/data/bench_results/eloftr_vs_loftr_pose_diff.csv

import argparse
import csv
import glob
import os

import numpy as np


def load_poses(out_folder):
  pose_dir = os.path.join(out_folder, 'ob_in_cam')
  files = sorted(glob.glob(os.path.join(pose_dir, '*.txt')))
  poses = {}
  for f in files:
    id_str = os.path.basename(f).replace('.txt', '')
    poses[id_str] = np.loadtxt(f).reshape(4, 4)
  return poses


def rot_angle_deg(R_rel):
  cos_theta = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
  return float(np.degrees(np.arccos(cos_theta)))


def stats(x):
  x = np.asarray(x)
  return dict(mean=float(np.mean(x)), median=float(np.median(x)),
              p90=float(np.percentile(x, 90)), max=float(np.max(x)))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--out_folder_a', type=str, required=True)
  parser.add_argument('--out_folder_b', type=str, required=True)
  parser.add_argument('--csv_out', type=str, required=True)
  args = parser.parse_args()

  poses_a = load_poses(args.out_folder_a)
  poses_b = load_poses(args.out_folder_b)

  common_ids = sorted(set(poses_a.keys()) & set(poses_b.keys()))
  if len(common_ids) == 0:
    raise RuntimeError("No common frame ids between the two out_folders; did both runs finish?")

  missing_in_a = sorted(set(poses_b.keys()) - set(poses_a.keys()))
  missing_in_b = sorted(set(poses_a.keys()) - set(poses_b.keys()))
  if missing_in_a or missing_in_b:
    print(f"[compare_poses] WARNING: {len(missing_in_a)} frames only in B, "
          f"{len(missing_in_b)} frames only in A (skipped from comparison)")

  rows = []
  for id_str in common_ids:
    Ta = poses_a[id_str]
    Tb = poses_b[id_str]
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Rb, tb = Tb[:3, :3], Tb[:3, 3]
    rot_deg = rot_angle_deg(Ra.T @ Rb)
    trans_cm = float(np.linalg.norm(ta - tb) * 100.0)  # poses are in meters
    rows.append((id_str, rot_deg, trans_cm))

  os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
  with open(args.csv_out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id_str', 'rot_diff_deg', 'trans_diff_cm'])
    w.writerows(rows)

  rot_stats = stats([r[1] for r in rows])
  trans_stats = stats([r[2] for r in rows])

  print(f"[compare_poses] compared {len(rows)} common frames")
  print(f"[compare_poses]   A={args.out_folder_a}")
  print(f"[compare_poses]   B={args.out_folder_b}")
  print(f"[compare_poses] rotation diff [deg]  : mean={rot_stats['mean']:.3f} median={rot_stats['median']:.3f} "
        f"p90={rot_stats['p90']:.3f} max={rot_stats['max']:.3f}")
  print(f"[compare_poses] translation diff [cm]: mean={trans_stats['mean']:.3f} median={trans_stats['median']:.3f} "
        f"p90={trans_stats['p90']:.3f} max={trans_stats['max']:.3f}")
  print(f"[compare_poses] per-frame CSV written to: {args.csv_out}")
  print("[compare_poses] NOTE: this is an eloftr-vs-loftr relative consistency metric, not GT ADD/ADD-S. "
        "For absolute accuracy, run benchmark_ho3d.py on the HO3D dataset (has GT poses).")


if __name__ == '__main__':
  main()
