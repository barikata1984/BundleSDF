#!/usr/bin/env python3
# Stacked-area timeline of perf_logger.py spans: per-step (frame or NeRF round)
# breakdown of where the time went, steps left-to-right.
#
# Usage:
#   python3 scripts/plot_perf_timeline.py --main data/out_milk_eloftr_async/perf_main.csv \
#       --nerf data/out_milk_eloftr_async/perf_nerf.csv --out_dir data/bench_results

import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# find_corres_ref/find_corres_local are parents of get_pairs/loftr_predict/raw_to_corres/ransac
# (see perf_logger.py's `sub` list); stacking both parent and children double-counts time,
# so the main breakdown uses the expanded children instead of the two parent spans.
MAIN_STACK_MS = [
    'depth_denoise_ms', 'make_frame_ms', 'invalidate_mask_ms',
    'get_pairs_ms', 'loftr_predict_ms', 'raw_to_corres_ms', 'ransac_ms',
    'ref_research_ms', 'procrustes_ms', 'select_kf_ms', 'get_match_pairs_ms',
    'optimize_gpu_ms', 'check_add_kf_ms', 'nerf_send_ms', 'nerf_wait_ms',
    'pose_writeback_ms', 'rematch_ms', 'save_result_ms', 'residual_ms',
]

NERF_STACK_MS = [
    'kf_receive_ms', 'scene_bounds_ms', 'preprocess_ms', 'runner_build_ms',
    'train_total_ms', 'extract_mesh_ms', 'pose_writeback_ms', 'residual_ms',
]


def load_csv(path):
  with open(path) as f:
    return list(csv.DictReader(f))


def plot_stack(rows, stack_cols, unit_divisor, ylabel, title, out_path, exclude=(), ymax=None):
  """Render one stacked-area chart: x=step index, y=stack_cols (ms columns / unit_divisor).

  `exclude` drops columns (e.g. nerf_wait) so the remaining series aren't dwarfed by a
  dominant spike; `ymax` clips the axis for the same reason without dropping data.
  """
  present = [c for c in stack_cols if c in rows[0] and c not in exclude]
  x = list(range(len(rows)))
  series = [[float(r.get(c, 0.0) or 0.0) / unit_divisor for r in rows] for c in present]

  fig, ax = plt.subplots(figsize=(14, 6))
  ax.stackplot(x, series, labels=[c.removesuffix('_ms') for c in present])
  ax.set_xlabel('step')
  ax.set_ylabel(ylabel)
  ax.set_title(title)
  if ymax is not None:
    ax.set_ylim(0, ymax)
  ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
  fig.tight_layout()
  fig.savefig(out_path, dpi=120)
  plt.close(fig)
  print(f"wrote {out_path}  ({len(x)} steps, {len(present)} stacked series)")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--main', help='path to perf_main.csv (per-frame tracking timeline)')
  ap.add_argument('--nerf', help='path to perf_nerf.csv (per-round NeRF timeline)')
  ap.add_argument('--out_dir', required=True)
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)

  if args.main:
    rows = load_csv(args.main)
    if rows:
      plot_stack(rows, MAIN_STACK_MS, 1.0, 'ms', 'Tracking per-frame breakdown (all spans)',
                 os.path.join(args.out_dir, 'timeline_main_full.png'))
      # nerf_wait dwarfs everything else on a shared axis; drop it to see the steady-state
      # per-frame cost (matching/BA/save) and how it drifts over the run.
      plot_stack(rows, MAIN_STACK_MS, 1.0, 'ms', 'Tracking per-frame breakdown (excl. nerf_wait)',
                 os.path.join(args.out_dir, 'timeline_main_steady.png'), exclude=['nerf_wait_ms'])
    else:
      print(f"{args.main}: no rows, skipping")

  if args.nerf:
    rows = load_csv(args.nerf)
    if rows:
      plot_stack(rows, NERF_STACK_MS, 1000.0, 's', 'NeRF per-round breakdown',
                 os.path.join(args.out_dir, 'timeline_nerf.png'))
    else:
      print(f"{args.nerf}: no rows, skipping")


if __name__ == '__main__':
  main()
