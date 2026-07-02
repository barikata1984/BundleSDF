# EfficientLoFTR (vendored)

Vendored inference code for [EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR)
(zju3dv), used as the default feature matcher backend in `loftr_wrapper.py`.

- Upstream: https://github.com/zju3dv/EfficientLoFTR
- Commit: `ffd4a4644064354468eb1f0c7a3e732233cb732f`
- License: Apache-2.0 (see `LICENSE`)

## What is included

Only the self-contained inference package `src/loftr/` is vendored, plus a minimal
`src/utils/misc.py` providing `detect_NaN` (upstream imports it from a
`pytorch_lightning`-dependent module; the stub avoids that dependency).
Training code (`src/loftr/utils/supervision.py`), configs, and the training/eval
entry points are intentionally not vendored.

Runtime dependencies: `torch`, `einops`, `kornia`, `loguru`, `yacs`, `numpy`
(all already provided by the main container).

## Weights

The full-model checkpoint is distributed only via Google Drive:
https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q

Download `eloftr_outdoor.ckpt` and place it at:

```
BundleTrack/EfficientLoFTR/weights/eloftr_outdoor.ckpt
```

Example (from the repository root, requires `gdown`): download the whole
distribution folder, then keep only `eloftr_outdoor.ckpt`:

```bash
gdown --folder https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q \
  -O BundleTrack/EfficientLoFTR/weights
```

Alternatively, download `eloftr_outdoor.ckpt` manually from the folder and place it
at the path above. The checkpoint is neither committed nor downloaded automatically.

## Backend selection

`loftr_wrapper.py` selects the backend via the `LoftrRunner(backend=...)` argument,
defaulting to the `BUNDLESDF_MATCHER` environment variable (`eloftr` by default;
set to `loftr` to use the original LoFTR under `BundleTrack/LoFTR/`).
