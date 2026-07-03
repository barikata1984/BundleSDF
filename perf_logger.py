# Per-stage timing profiler. Active only when BUNDLESDF_PROFILE=='1'; otherwise
# every method is a cheap no-op (single attribute check). Each process holds its
# own singleton instance and writes its own CSV file.
import os
import csv
import time
from contextlib import contextmanager, nullcontext

# Column schema per profiler. `sub` spans are nested breakdowns (multi-call):
# they are excluded from the residual sum and get an extra _n count column.
_SCHEMAS = {
  'main': dict(
    filename='perf_main.csv', id_cols=('frame_id',),
    spans=['depth_denoise', 'make_frame', 'invalidate_mask', 'find_corres_ref',
           'ref_research', 'procrustes', 'select_kf', 'get_match_pairs',
           'find_corres_local', 'optimize_gpu', 'check_add_kf', 'nerf_send',
           'nerf_wait', 'pose_writeback', 'rematch', 'save_result',
           'get_pairs', 'loftr_predict', 'raw_to_corres', 'ransac'],
    sub=['get_pairs', 'loftr_predict', 'raw_to_corres', 'ransac']),
  'nerf': dict(
    filename='perf_nerf.csv', id_cols=('round',),
    spans=['kf_receive', 'scene_bounds', 'preprocess', 'runner_build',
           'train_total', 'extract_mesh', 'pose_writeback'], sub=[]),
  'nerf_train': dict(
    filename='perf_nerf_train.csv', id_cols=('round', 'iter_range'),
    spans=['batch_get', 'render', 'loss', 'backward', 'opt_step'], sub=[]),
}

_NULL = nullcontext()

_torch = None
_cuda_ok = None


def _vram_mib():
  """Return this process's (allocated, reserved) GPU memory in MiB.

  allocated is what PyTorch currently uses (good for leak detection); reserved
  is the caching allocator's total (closer to nvidia-smi). Returns (nan, nan)
  when torch/CUDA is unavailable so the profiler never crashes off-GPU.
  """
  global _torch, _cuda_ok
  if _cuda_ok is None:
    try:
      import torch
      _torch = torch
      _cuda_ok = torch.cuda.is_available()
    except Exception:
      _cuda_ok = False
  if not _cuda_ok:
    return float('nan'), float('nan')
  scale = 1.0 / (1024 * 1024)
  return _torch.cuda.memory_allocated() * scale, _torch.cuda.memory_reserved() * scale


class SpanProfiler:
  def __init__(self, out_dir, schema):
    self.enabled = os.environ.get('BUNDLESDF_PROFILE') == '1'
    if not self.enabled:
      return
    os.makedirs(out_dir, exist_ok=True)
    self._names = schema['spans']
    self._sub = set(schema['sub'])
    self._ms = {}
    self._n = {}
    self._t0 = time.perf_counter()
    self._fh = open(os.path.join(out_dir, schema['filename']), 'w', newline='')
    self._w = csv.writer(self._fh)
    header = list(schema['id_cols']) + [f'{n}_ms' for n in self._names] \
      + [f'{n}_n' for n in self._names if n in self._sub] + ['total_ms', 'residual_ms'] \
      + ['vram_alloc_mib', 'vram_reserved_mib']
    self._w.writerow(header)
    self._fh.flush()

  def span(self, name):
    if not self.enabled:
      return _NULL
    return self._span(name)

  @contextmanager
  def _span(self, name):
    t0 = time.perf_counter()
    try:
      yield
    finally:
      dt = (time.perf_counter() - t0) * 1000.0
      self._ms[name] = self._ms.get(name, 0.0) + dt
      self._n[name] = self._n.get(name, 0) + 1

  def start(self):
    if not self.enabled:
      return
    self._ms.clear()
    self._n.clear()
    self._t0 = time.perf_counter()

  def flush(self, *ids):
    if not self.enabled:
      return
    now = time.perf_counter()
    total = (now - self._t0) * 1000.0
    top_sum = sum(self._ms.get(n, 0.0) for n in self._names if n not in self._sub)
    row = list(ids)
    row += [round(self._ms.get(n, 0.0), 3) for n in self._names]
    row += [self._n.get(n, 0) for n in self._names if n in self._sub]
    alloc, reserved = _vram_mib()
    row += [round(total, 3), round(total - top_sum, 3), round(alloc, 1), round(reserved, 1)]
    self._w.writerow(row)
    self._fh.flush()
    self._ms.clear()
    self._n.clear()
    self._t0 = now


_profilers = {}


def get_profiler(name, out_dir):
  prof = _profilers.get(name)
  if prof is None:
    prof = SpanProfiler(out_dir, _SCHEMAS[name])
    _profilers[name] = prof
  return prof
