#!/usr/bin/env python3
# Guard against "fake" speedups: verify consecutive ob_in_cam poses are not
# just the previous frame's pose repeated (which would mean tracking stopped
# computing but the frame counter kept advancing).
import glob, os, sys
import numpy as np

folder = sys.argv[1]
files = sorted(glob.glob(os.path.join(folder, 'ob_in_cam', '*.txt')))
poses = [np.loadtxt(f).reshape(4, 4) for f in files]
n = len(poses)
identical = 0
trans_deltas = []
rot_deltas = []
for a, b in zip(poses, poses[1:]):
    d = np.abs(a - b).max()
    if d < 1e-9:
        identical += 1
    dt = np.linalg.norm(a[:3, 3] - b[:3, 3]) * 100.0  # cm
    Rr = a[:3, :3].T @ b[:3, :3]
    c = np.clip((np.trace(Rr) - 1) / 2, -1, 1)
    rot_deltas.append(np.degrees(np.arccos(c)))
    trans_deltas.append(dt)
trans_deltas = np.array(trans_deltas); rot_deltas = np.array(rot_deltas)
print(f"poses={n}  consecutive-pairs={n-1}")
print(f"bit-identical consecutive poses: {identical}")
print(f"frame-to-frame trans delta cm: median={np.median(trans_deltas):.4f} "
      f"max={trans_deltas.max():.4f}  zero(<1e-6cm)={int((trans_deltas<1e-6).sum())}")
print(f"frame-to-frame rot   delta deg: median={np.median(rot_deltas):.4f} "
      f"max={rot_deltas.max():.4f}  zero(<1e-6deg)={int((rot_deltas<1e-6).sum())}")
