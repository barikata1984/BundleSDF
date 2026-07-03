#!/usr/bin/env python3
# Frame-level timing benchmark for BundleSDF online tracking on the milk-carton
# demo video. Does NOT modify run_custom.py; instead it re-implements only the
# online per-frame tracking loop found in run_custom.py::run_one_video(), with
# a wall-clock timer wrapped around each tracker.run() call.
#
# The config block below is copied verbatim from run_custom.py::run_one_video()
# and must be kept in sync manually if that function's defaults change.
#
# The offline global-refine / textured-mesh step
# (run_custom.py::run_one_video_global_nerf, invoked via
# `run_custom.py --mode global_refine`) is intentionally NOT run here since it
# is not part of the online tracking speed being measured.
#
# Usage:
#   BUNDLESDF_MATCHER=eloftr python3 scripts/bench_milk.py \
#       --video_dir /workspace/data/2022-11-18-15-10-24_milk \
#       --out_folder /workspace/data/out_milk_eloftr \
#       --csv_out /workspace/data/bench_results/eloftr_frame_times.csv

import argparse
import csv
import os
import sys
import time

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.dirname(code_dir)
sys.path.append(repo_dir)

from bundlesdf import *  # noqa: F401,F403 -- brings in BundleSdf, set_seed, yaml, np, cv2, logging, YcbineoatReader, ...


def run_one_video_timed(video_dir, out_folder, use_segmenter, debug_level, stride, warmup_frames, csv_out,
                         max_frames=None):
  if use_segmenter:
    # use_segmenter=1 has a known mask-resize mismatch bug (see notes/ISSUES.md);
    # this benchmark always relies on the dataset-provided masks instead.
    raise NotImplementedError("bench_milk.py only supports use_segmenter=0 (dataset masks)")

  set_seed(0)

  os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

  # --- config, mirrored from run_custom.py::run_one_video() ---
  cfg_bundletrack = yaml.load(open(f"{repo_dir}/BundleTrack/config_ho3d.yml", 'r'))
  cfg_bundletrack['SPDLOG'] = int(debug_level)
  cfg_bundletrack['depth_processing']["percentile"] = 95
  cfg_bundletrack['erode_mask'] = 3
  cfg_bundletrack['debug_dir'] = out_folder + '/'
  cfg_bundletrack['bundle']['max_BA_frames'] = 10
  cfg_bundletrack['bundle']['max_optimized_feature_loss'] = 0.03
  cfg_bundletrack['feature_corres']['max_dist_neighbor'] = 0.02
  cfg_bundletrack['feature_corres']['max_normal_neighbor'] = 30
  cfg_bundletrack['feature_corres']['max_dist_no_neighbor'] = 0.01
  cfg_bundletrack['feature_corres']['max_normal_no_neighbor'] = 20
  cfg_bundletrack['feature_corres']['map_points'] = True
  cfg_bundletrack['feature_corres']['resize'] = 400
  cfg_bundletrack['feature_corres']['rematch_after_nerf'] = True
  cfg_bundletrack['keyframe']['min_rot'] = 5
  cfg_bundletrack['ransac']['inlier_dist'] = 0.01
  cfg_bundletrack['ransac']['inlier_normal_angle'] = 20
  cfg_bundletrack['ransac']['max_trans_neighbor'] = 0.02
  cfg_bundletrack['ransac']['max_rot_deg_neighbor'] = 30
  cfg_bundletrack['ransac']['max_trans_no_neighbor'] = 0.01
  cfg_bundletrack['ransac']['max_rot_no_neighbor'] = 10
  cfg_bundletrack['p2p']['max_dist'] = 0.02
  cfg_bundletrack['p2p']['max_normal_angle'] = 45
  cfg_track_dir = f'{out_folder}/config_bundletrack.yml'
  yaml.dump(cfg_bundletrack, open(cfg_track_dir, 'w'))

  cfg_nerf = yaml.load(open(f"{repo_dir}/config.yml", 'r'))
  cfg_nerf['continual'] = True
  cfg_nerf['trunc_start'] = 0.01
  cfg_nerf['trunc'] = 0.01
  cfg_nerf['mesh_resolution'] = 0.005
  cfg_nerf['down_scale_ratio'] = 1
  cfg_nerf['fs_sdf'] = 0.1
  cfg_nerf['far'] = cfg_bundletrack['depth_processing']["zfar"]
  cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
  cfg_nerf['notes'] = ''
  cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
  cfg_nerf['save_dir'] = cfg_nerf['datadir']
  cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
  yaml.dump(cfg_nerf, open(cfg_nerf_dir, 'w'))
  # --- end mirrored config ---

  tracker = BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=False)

  reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)

  rows = []
  n_frames = len(reader.color_files)
  if max_frames is not None:
    n_frames = min(n_frames, max_frames)
  for i in range(0, n_frames, stride):
    color_file = reader.color_files[i]
    color = cv2.imread(color_file)
    depth = reader.get_depth(i)
    H, W = depth.shape[:2]
    color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
    depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)

    mask = reader.get_mask(i)
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    if cfg_bundletrack['erode_mask'] > 0:
      kernel = np.ones((cfg_bundletrack['erode_mask'], cfg_bundletrack['erode_mask']), np.uint8)
      mask = cv2.erode(mask.astype(np.uint8), kernel)

    id_str = reader.id_strs[i]
    pose_in_model = np.eye(4)
    K = reader.K.copy()

    t0 = time.perf_counter()
    tracker.run(color, depth, K, id_str, mask=mask, occ_mask=None, pose_in_model=pose_in_model)
    t1 = time.perf_counter()
    rows.append((i, id_str, (t1 - t0) * 1000.0))

  tracker.on_finish()

  os.makedirs(os.path.dirname(csv_out), exist_ok=True)
  with open(csv_out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame_idx', 'id_str', 'elapsed_ms'])
    w.writerows(rows)

  # --- summary stats, warmup frames excluded ---
  timed = [r[2] for r in rows[warmup_frames:]]
  if len(timed) == 0:
    print(f"[bench_milk] WARNING: no frames left after excluding {warmup_frames} warmup frames "
          f"out of {len(rows)} total; skipping summary stats")
    return

  timed_sorted = sorted(timed)

  def _pct(p):
    idx = min(len(timed_sorted) - 1, int(round(p * (len(timed_sorted) - 1))))
    return timed_sorted[idx]

  mean_ms = sum(timed) / len(timed)
  median_ms = _pct(0.5)
  p90_ms = _pct(0.9)
  fps = 1000.0 / mean_ms if mean_ms > 0 else float('nan')

  print(f"[bench_milk] frames_total={len(rows)} frames_timed={len(timed)} warmup_excluded={warmup_frames}")
  print(f"[bench_milk] track() elapsed_ms: mean={mean_ms:.2f} median={median_ms:.2f} p90={p90_ms:.2f}")
  print(f"[bench_milk] throughput={fps:.3f} frames/s (1000/mean_ms, excludes warmup)")
  print(f"[bench_milk] per-frame CSV written to: {csv_out}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--video_dir', type=str, required=True)
  parser.add_argument('--out_folder', type=str, required=True)
  parser.add_argument('--use_segmenter', type=int, default=0)
  parser.add_argument('--debug_level', type=int, default=1)
  parser.add_argument('--stride', type=int, default=1, help='interval of frames to run; 1 means using every frame')
  parser.add_argument('--warmup_frames', type=int, default=5,
                       help='number of leading frames excluded from timing stats (CUDA/cuDNN/JIT warmup)')
  parser.add_argument('--csv_out', type=str, required=True, help='path to write per-frame timing CSV')
  parser.add_argument('--max_frames', type=int, default=None,
                       help='limit number of frames processed (for smoke tests); default: all frames')
  args = parser.parse_args()

  run_one_video_timed(
      video_dir=args.video_dir,
      out_folder=args.out_folder,
      use_segmenter=args.use_segmenter,
      debug_level=args.debug_level,
      stride=args.stride,
      warmup_frames=args.warmup_frames,
      csv_out=args.csv_out,
      max_frames=args.max_frames,
  )
