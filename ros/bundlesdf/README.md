# bundlesdf_node

Online BundleSDF 6-DoF tracking as a ROS (ROS One) node. Subscribes
time-synchronized rgb/depth/mask, feeds each frame to `BundleSdf.run()`
(`bundlesdf.py`, unmodified), and publishes the tracked object's pose.

Runs in the main BundleSDF container (`docker/ros-one.dockerfile`). Requires
`build.sh` to have been run first (`mycuda` + `BundleTrack` C++/CUDA build).

## Topic contract

Input topic names come from the shared `camera_input.yaml` (see
[Camera input config](#camera-input-config)), not from per-node remaps.

| Direction | Topic | Config key / default | Type | Notes |
|-----------|-------|----------------------|------|-------|
| sub | rgb | `rgb_in` = `/d455_1/color/image_rect` | `sensor_msgs/Image` | `rgb8` or `bgr8` |
| sub | depth | `depth_in` = `/d455_1/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | `16UC1` (mm) or `32FC1` (m) |
| sub | camera_info | `camera_info_in` = `/d455_1/color/camera_info_rect` | `sensor_msgs/CameraInfo` | `K` used as the 3x3 intrinsics |
| sub | mask | `mask_topic` = `/sam3/mask` | `sensor_msgs/Image` | `mono8`, per `sam3_segmenter`'s mask contract |
| pub | `~object_pose` | `/bundlesdf_node/object_pose` | `geometry_msgs/PoseStamped` | object pose in the input frame_id |
| pub | TF | `<input frame_id>` -> `tracked_object` | `tf2` | same pose as `~object_pose` |

Notes:

- rgb/depth/mask are joined with `message_filters.ApproximateTimeSynchronizer`
  (`slop=0.05`s). The mask contract (`sam3_segmenter`) inherits the input RGB's
  header verbatim, so this only works if the mask stamp equals the RGB stamp
  it was computed from.
- `camera_info_in` is cached on receipt; frames are dropped (with a
  throttled warning) until the first `CameraInfo` arrives.
- No `cv_bridge`: decoding uses `np.frombuffer` directly, consistent with the
  rest of this fork.
- `tracker.run()` is synchronous per frame (it blocks until BundleTrack +
  the background NeRF-refinement handshake finish). Throughput is bounded by
  per-frame processing time; see `notes/PERF_plan.md` for benchmarked numbers.
- The tracked object's mask (`mask_topic`) must cover the target object from the
  first synced frame onward — this is what seeds the object's initial
  coordinate frame. There is no separate bbox/point re-init in this node.

## Camera input config

The four input topic names live in a shared YAML
(`config/camera_input.yaml`), loaded into the `/camera_input` namespace by the
launch files (`<rosparam ... ns="camera_input" />`). Both this node and
`sam3_segmenter` read from it, so the RGB stream is defined in one place.

```yaml
rgb_in: /d455_1/color/image_rect
depth_in: /d455_1/aligned_depth_to_color/image_raw
camera_info_in: /d455_1/color/camera_info_rect
mask_topic: /sam3/mask
```

Point the launch `camera_config` arg at a different file to switch cameras.

## Parameters

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `camera_config` | string | `$(find bundlesdf)/config/camera_input.yaml` | YAML of input topic names, loaded under `/camera_input`. |
| `~target_frame` | string | `tracked_object` | TF child frame id for the published pose. |
| `~out_folder` | string | `/tmp/bundlesdf_online` | Working dir for tracker debug output (wiped on startup). Poses are read back from `<out_folder>/ob_in_cam/<id_str>.txt`. |
| `~debug_level` | int | `1` | Forwarded to `BundleTrack`'s `SPDLOG` (higher = more logging/artifacts). |
| `~use_gui` | bool | `false` | Show `BundleSdf`'s dearpygui viewer. Needs an X display; the first frame blocks until the GUI process is up. |

## Run

Standalone (mask must already be published by something else on the
`mask_topic` from `camera_input.yaml`):

```bash
roslaunch bundlesdf bundlesdf_node.launch use_gui:=true
```

Combined with `sam3_segmenter` (both nodes run in this same container --
`docker/ros-one.dockerfile` bundles SAM 3's deps alongside the tracker):

```bash
roslaunch bundlesdf bundlesdf.launch \
  use_segmenter:=true \
  target_object:="a red mug"
```

Override the input topics by pointing `camera_config` at another YAML:

```bash
roslaunch bundlesdf bundlesdf.launch \
  camera_config:=/path/to/my_camera.yaml
```

`use_segmenter:=false` skips including `sam3_segmenter.launch`, for when a
mask is already being published on `mask_topic` (default `/sam3/mask`) by
some other means.

## Out of scope (this node)

- Mesh / surface-point publishing (only pose + TF are published).
- Robot state (TF from a robot, force/torque, etc.) is not consumed here;
  there is no `with_robot` toggle.
- Re-initialization after tracking loss (frame stays `FAIL`'d and the node
  keeps running; there is no automatic re-seed from a fresh mask).

## Verification status

- `python3 -m py_compile` passes for the node.
- Running as a live `rosnode` requires the container (`build.sh` completed,
  `torch`/`rospy`/`message_filters` available) and a real or bagged rgb/depth/
  mask stream; not yet run end-to-end.
