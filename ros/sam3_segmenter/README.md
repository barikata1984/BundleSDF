# sam3_segmenter

SAM 3 text-prompted streaming segmentation as a ROS (ROS One) node.
It subscribes to an RGB stream, tracks a single object described by a text
prompt, and publishes a binary mask for the downstream BundleSDF tracker.

This node runs in the main BundleSDF container (`docker/ros-one.dockerfile`,
Python 3.10). SAM 3 via the transformers integration works on Python >= 3.10
(the "Python 3.12+" note in the upstream sam3 README is contradicted by its
own pyproject and by working 3.10 deployments); only the transformers package
gates the version. It stays a separate node/process from the tracker, joined
by the mask topic below.

## Topic contract

| Direction | ROS name (node-relative) | Default remap | Type | Notes |
|-----------|--------------------------|---------------|------|-------|
| sub | `~image_in`  | `/camera/color/image_raw` | `sensor_msgs/Image` | encoding `rgb8` or `bgr8` |
| pub | `~mask_out`  | `/sam3/mask`              | `sensor_msgs/Image` | encoding `mono8`, single channel |

Contract details (frozen with the interface hub):

- **Mask values**: foreground = 255, background = 0. `step = width`.
- **Header**: the mask inherits `header.stamp` and `header.frame_id` from the
  input RGB message verbatim. This is what lets the downstream node time-sync
  RGB / depth / mask.
- **Single object**: on multiple detections the node keeps one mask — the
  highest-score object, ties broken by mask area.
- **No detection**: an all-zero mask is still published (header inherited), so
  the downstream stream is never gapped. Consumers threshold with `> 0`.
- **RGB decode**: no `cv_bridge`; `np.frombuffer` with the message `step` and
  `encoding` handled directly.
- **Backpressure**: `queue_size=1` with a large `buff_size`, so when inference
  cannot keep up only the latest frame is processed.

## Parameters

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `~text_prompt` | string | (required) | SAM 3 concept prompt, e.g. `"a red mug"`. |
| `~model_id` | string | `facebook/sam3` | Hugging Face model id. |

Initialization is text-prompt only by contract. bbox/point/mask init is not
defined: the concept streaming API (`Sam3VideoModel` / `Sam3VideoProcessor`)
only exposes `add_text_prompt`. If instance-precise init is ever needed, swap
the concept model for `Sam3TrackerVideo` (SAM 2 lineage, Promptable Visual
Segmentation), whose `add_inputs_to_inference_session(input_boxes=..., input_points=...)` accepts boxes/points.

## Weights

`facebook/sam3` is a **gated** Hugging Face model (~3.45 GB, `sam3.pt`). Before
first launch:

1. Request access on the model page and wait for approval.
2. Authenticate: `hf auth login` (or set `HF_TOKEN`).
3. The weights are downloaded on first run into `HF_HOME` (`/hf_cache` in the
   container). Mount a host directory there so the download is not baked into
   the image and persists across runs.

## Streaming caveat

Streaming inference disables the SAM 3 hotstart heuristics that prune unmatched
and duplicate objects. Compared to pre-loaded (whole-video) inference this can
yield more false positives and duplicate tracks. The single-object selection
above mitigates this for BundleSDF's single-target use, but expect the mask to
occasionally jump between candidates in cluttered scenes.

## Run

```bash
roslaunch sam3_segmenter sam3_segmenter.launch \
  text_prompt:="a red mug" \
  image_in:=/camera/color/image_raw \
  mask_out:=/sam3/mask
```

## Verification status

- `python3 -m py_compile` passes for the node.
- Running as a live `rosnode` requires the container (torch / transformers /
  rospy). Build the image per `docker/ros-one.dockerfile` and download the
  weights per the section above.
