#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BundleSDF ROS Node

Subscribes to RealSense camera topics (RGB, Depth, CameraInfo) and
publishes tracked object pose using BundleSDF.
"""

import os
import sys
import threading
import time
import numpy as np
import cv2
import yaml
import csv
from datetime import datetime

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
import tf2_ros
from geometry_msgs.msg import TransformStamped
import struct

# Add BundleSDF to Python path
BUNDLESDF_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, BUNDLESDF_DIR)

from bundlesdf import BundleSdf, set_seed
from segmentation_utils import Segmenter


class FeaturePointLogger:
    """Logs feature point data for stability investigation."""

    def __init__(self, output_dir, log_every_n_frames=1):
        self.log_dir = os.path.join(output_dir, "feature_point_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_every_n = log_every_n_frames
        self.csv_path = os.path.join(self.log_dir, "shape_summary.csv")
        self.log_path = os.path.join(self.log_dir, "shape_summary.log")
        self._init_csv()

    def _init_csv(self):
        """Initialize CSV file with header."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "frame_id",
                    "type",
                    "num_points",
                    "x_min",
                    "x_max",
                    "y_min",
                    "y_max",
                    "z_min",
                    "z_max",
                    "x_mean",
                    "y_mean",
                    "z_mean",
                    "x_std",
                    "y_std",
                    "z_std",
                    "skip_reason",
                ]
            )

    def compute_shape_stats(self, points_3d):
        """Compute shape statistics for a point cloud."""
        if points_3d is None or len(points_3d) == 0:
            return None
        stats = {
            "num_points": len(points_3d),
            "xyz_min": points_3d.min(axis=0),
            "xyz_max": points_3d.max(axis=0),
            "xyz_mean": points_3d.mean(axis=0),
            "xyz_std": points_3d.std(axis=0),
        }
        return stats

    def log_feature_points(
        self, frame_id, timestamp, ransac_points, surface_points, ransac_skip_reason=""
    ):
        """Log both feature point types."""
        # Save raw data as .npy
        if ransac_points is not None and len(ransac_points) > 0:
            np.save(
                os.path.join(self.log_dir, f"ransac_{frame_id}.npy"), ransac_points
            )
        if surface_points is not None and len(surface_points) > 0:
            np.save(
                os.path.join(self.log_dir, f"surface_{frame_id}.npy"), surface_points
            )

        # Compute and log statistics
        ransac_stats = self.compute_shape_stats(ransac_points)
        surface_stats = self.compute_shape_stats(surface_points)

        self._write_stats_to_csv(
            frame_id, timestamp, "ransac", ransac_stats, ransac_skip_reason
        )
        self._write_stats_to_csv(frame_id, timestamp, "surface", surface_stats, "")
        self._write_stats_to_log(
            frame_id, timestamp, ransac_stats, surface_stats, ransac_skip_reason
        )

    def _write_stats_to_csv(self, frame_id, timestamp, point_type, stats, skip_reason):
        """Write statistics to CSV file."""
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if stats is None:
                # Write zeros and NaN for skipped/failed extraction
                writer.writerow(
                    [
                        timestamp,
                        frame_id,
                        point_type,
                        0,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        skip_reason,
                    ]
                )
            else:
                writer.writerow(
                    [
                        timestamp,
                        frame_id,
                        point_type,
                        stats["num_points"],
                        stats["xyz_min"][0],
                        stats["xyz_max"][0],
                        stats["xyz_min"][1],
                        stats["xyz_max"][1],
                        stats["xyz_min"][2],
                        stats["xyz_max"][2],
                        stats["xyz_mean"][0],
                        stats["xyz_mean"][1],
                        stats["xyz_mean"][2],
                        stats["xyz_std"][0],
                        stats["xyz_std"][1],
                        stats["xyz_std"][2],
                        skip_reason,
                    ]
                )
            f.flush()  # Flush buffer to ensure immediate write

    def _write_stats_to_log(
        self, frame_id, timestamp, ransac_stats, surface_stats, ransac_skip_reason
    ):
        """Write statistics to human-readable log file."""
        with open(self.log_path, "a") as f:
            f.write(f"\n=== Frame {frame_id} @ {timestamp:.6f} ===\n")

            # RANSAC inliers
            if ransac_stats:
                f.write(f"RANSAC Inliers: {ransac_stats['num_points']} points\n")
                f.write(
                    f"  Bounds: X[{ransac_stats['xyz_min'][0]:.4f}, {ransac_stats['xyz_max'][0]:.4f}]\n"
                )
                f.write(
                    f"          Y[{ransac_stats['xyz_min'][1]:.4f}, {ransac_stats['xyz_max'][1]:.4f}]\n"
                )
                f.write(
                    f"          Z[{ransac_stats['xyz_min'][2]:.4f}, {ransac_stats['xyz_max'][2]:.4f}]\n"
                )
                f.write(
                    f"  Centroid: ({ransac_stats['xyz_mean'][0]:.4f}, {ransac_stats['xyz_mean'][1]:.4f}, {ransac_stats['xyz_mean'][2]:.4f})\n"
                )
            else:
                f.write(f"RANSAC Inliers: N/A (reason: {ransac_skip_reason})\n")

            # Surface points
            if surface_stats:
                f.write(f"Surface Points: {surface_stats['num_points']} points\n")
                f.write(
                    f"  Bounds: X[{surface_stats['xyz_min'][0]:.4f}, {surface_stats['xyz_max'][0]:.4f}]\n"
                )
                f.write(
                    f"          Y[{surface_stats['xyz_min'][1]:.4f}, {surface_stats['xyz_max'][1]:.4f}]\n"
                )
                f.write(
                    f"          Z[{surface_stats['xyz_min'][2]:.4f}, {surface_stats['xyz_max'][2]:.4f}]\n"
                )
                f.write(
                    f"  Centroid: ({surface_stats['xyz_mean'][0]:.4f}, {surface_stats['xyz_mean'][1]:.4f}, {surface_stats['xyz_mean'][2]:.4f})\n"
                )
            else:
                f.write("Surface Points: N/A (reason: extraction_error)\n")
            f.flush()  # Flush buffer to ensure immediate write


class BundleSdfNode:
    """ROS Node wrapper for BundleSDF object pose tracking.
    
    Improvements for long-running stability:
    - GUI process health monitoring
    - Periodic memory cleanup
    - Graceful degradation when GUI crashes
    """

    def __init__(self):
        rospy.init_node("bundlesdf_node", anonymous=False)

        # Parameters
        self.output_dir = rospy.get_param("~output_dir", "/tmp/bundlesdf_output")
        self.use_segmenter = rospy.get_param("~use_segmenter", False)
        self.use_gui = rospy.get_param("~use_gui", False)
        self.shorter_side = rospy.get_param("~shorter_side", 480)
        self.frame_stride = rospy.get_param("~frame_stride", 1)
        self.debug_level = rospy.get_param("~debug_level", 2)
        # Toggle waiting for /robot_at_home topic (default: True)
        self.with_robot = rospy.get_param("~with_robot", True)
        
        # GUI health monitoring parameters
        self.gui_heartbeat_timeout = rospy.get_param("~gui_heartbeat_timeout", 10.0)  # seconds
        self.gui_check_interval = rospy.get_param("~gui_check_interval", 5.0)  # seconds

        # Target object for segmentation (optional: if set, skips interactive prompt)
        self.target_object = rospy.get_param("~target_object", "")

        # Frame ID for published pose
        self.camera_frame_id = rospy.get_param(
            "~camera_frame_id", "camera_color_optical_frame"
        )
        self.object_frame_id = rospy.get_param("~object_frame_id", "tracked_object")

        # Topic names (default to RealSense topics)
        self.color_topic = rospy.get_param("~color_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param(
            "~depth_topic", "/camera/aligned_depth_to_color/image_raw"
        )
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )

        # CV Bridge
        self.bridge = CvBridge()

        # Internal state
        self.tracker = None
        self.K = None  # Camera intrinsics
        self.frame_count = 0
        self.first_mask = None
        self.first_frame_processed = False
        self.lock = threading.Lock()
        self.gui_dead_logged = False  # Track if GUI death has been logged
        self.shutdown_requested = False  # Flag for graceful shutdown

        # Initialize segmenter if enabled
        if self.use_segmenter:
            rospy.loginfo("Initializing Segmenter (SAM3 + Cutie)...")
            self.segmenter = Segmenter()
            rospy.loginfo("Segmenter initialized")
        else:
            self.segmenter = None

        # Feature point logging for stability investigation
        self.enable_feature_logging = rospy.get_param("~enable_feature_logging", False)
        self.feature_log_stride = rospy.get_param("~feature_log_stride", 1)
        if self.enable_feature_logging:
            self.feature_logger = FeaturePointLogger(
                self.output_dir, self.feature_log_stride
            )
            rospy.loginfo(
                f"Feature point logging enabled. Output: {self.output_dir}/feature_point_logs/"
            )
            # Note: Enter key monitoring will start after first frame is processed
            # to avoid conflicts with SAM3 mask approval input
            self.enter_monitor_thread = None
        else:
            self.feature_logger = None

        # Surface point publishing configuration
        self.surface_point_sample_count = rospy.get_param(
            "~surface_point_sample_count", 128
        )
        rospy.loginfo(
            f"Surface point sampling: {self.surface_point_sample_count} points per frame"
        )

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publisher
        self.pose_pub = rospy.Publisher("~object_pose", PoseStamped, queue_size=10)
        self.surface_points_pub = rospy.Publisher(
            "~surface_points", PointCloud2, queue_size=10
        )
        self.surface_points_normalized_pub = rospy.Publisher(
            "~surface_points_normalized", PointCloud2, queue_size=10
        )
        self.object_targeted_pub = rospy.Publisher(
            "/object_targeted", Bool, queue_size=1, latch=True
        )

        # Setup config and tracker
        self._setup_tracker()
        
        # Start GUI health monitor timer if GUI is enabled
        if self.use_gui:
            self.gui_monitor_timer = rospy.Timer(
                rospy.Duration(self.gui_check_interval),
                self._check_gui_health
            )

        # Subscribers with message_filters for synchronized callback
        # Use queue_size=1 to prevent message backlog (only process latest frame)
        self.color_sub = message_filters.Subscriber(self.color_topic, Image, queue_size=1)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image, queue_size=1)
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1
        )

        # Approximate time synchronizer for RGB-D
        # Use queue_size=1 to prevent frame backlog - only process latest synchronized pair
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=1, slop=0.1
        )
        self.ts.registerCallback(self._rgbd_callback)

        rospy.loginfo(f"BundleSDF node initialized. Subscribing to:")
        rospy.loginfo(f"  Color: {self.color_topic}")
        rospy.loginfo(f"  Depth: {self.depth_topic}")
        rospy.loginfo(f"  CameraInfo: {self.camera_info_topic}")

    def _setup_tracker(self):
        """Initialize BundleSDF tracker with config."""
        set_seed(0)

        os.makedirs(self.output_dir, exist_ok=True)

        # Load and modify BundleTrack config
        cfg_bundletrack_path = os.path.join(
            BUNDLESDF_DIR, "third_party", "BundleTrack", "config_ho3d.yml"
        )
        with open(cfg_bundletrack_path, "r") as f:
            cfg_bundletrack = yaml.safe_load(f)

        # SPDLOG: 0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical (shows this level and below)
        # To suppress warnings, set to 4 or higher
        cfg_bundletrack["SPDLOG"] = int(self.debug_level)
        cfg_bundletrack["depth_processing"]["percentile"] = 95
        cfg_bundletrack["erode_mask"] = 3
        cfg_bundletrack["debug_dir"] = self.output_dir + "/"
        
        # Bundle Adjustment settings - increased for better long-term stability
        # Increase max_BA_frames to maintain consistency over longer sequences
        cfg_bundletrack["bundle"]["max_BA_frames"] = 20  # Was 10, increased for drift reduction
        cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
        
        # Feature correspondence settings - tightened for better accuracy
        cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
        cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
        cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
        cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
        cfg_bundletrack["feature_corres"]["map_points"] = True
        cfg_bundletrack["feature_corres"]["resize"] = 400
        cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
        
        # Keyframe selection - tighter threshold to create more keyframes for stability
        cfg_bundletrack["keyframe"]["min_rot"] = 3  # Was 5, reduced for more keyframes
        
        # RANSAC settings - tighter for better pose estimation
        cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
        cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
        cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
        cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
        cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
        cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
        cfg_bundletrack["p2p"]["max_dist"] = 0.02
        cfg_bundletrack["p2p"]["max_normal_angle"] = 45

        cfg_track_dir = os.path.join(self.output_dir, "config_bundletrack.yml")
        with open(cfg_track_dir, "w") as f:
            yaml.dump(cfg_bundletrack, f)

        # Load and modify NeRF config
        cfg_nerf_path = os.path.join(BUNDLESDF_DIR, "config.yml")
        with open(cfg_nerf_path, "r") as f:
            cfg_nerf = yaml.safe_load(f)

        cfg_nerf["continual"] = True
        cfg_nerf["trunc_start"] = 0.01
        cfg_nerf["trunc"] = 0.01
        cfg_nerf["mesh_resolution"] = 0.003  # Finer mesh (was 0.005)
        cfg_nerf["down_scale_ratio"] = 1
        cfg_nerf["fs_sdf"] = 0.1
        cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
        
        # Mesh extraction parameters to prevent disappearing
        cfg_nerf["mesh_isolevel"] = 0.0  # More permissive surface threshold (default: 0)
        cfg_nerf["mesh_min_vertices"] = 100  # Keep smaller mesh fragments
        cfg_nerf["datadir"] = os.path.join(
            self.output_dir, "nerf_with_bundletrack_online"
        )
        cfg_nerf["notes"] = ""
        cfg_nerf["expname"] = "nerf_with_bundletrack_online"
        cfg_nerf["save_dir"] = cfg_nerf["datadir"]

        cfg_nerf_dir = os.path.join(self.output_dir, "config_nerf.yml")
        with open(cfg_nerf_dir, "w") as f:
            yaml.dump(cfg_nerf, f)

        # Create tracker
        self.tracker = BundleSdf(
            cfg_track_dir=cfg_track_dir,
            cfg_nerf_dir=cfg_nerf_dir,
            start_nerf_keyframes=5,
            use_gui=self.use_gui,
        )

        rospy.loginfo("BundleSDF tracker initialized")

    def _monitor_enter_key(self):
        """Monitor for ENTER key press to trigger graceful shutdown."""
        try:
            input()  # Wait for ENTER key
            rospy.loginfo("\nENTER key detected. Initiating graceful shutdown...")
            self.shutdown_requested = True

            # Write final summary to log
            if self.feature_logger:
                with open(self.feature_logger.log_path, "a") as f:
                    f.write("\n" + "=" * 60 + "\n")
                    f.write("=== SHUTDOWN: Feature point logging completed ===\n")
                    f.write(f"Total frames processed: {self.frame_count}\n")
                    f.write("=" * 60 + "\n")
                    f.flush()
                rospy.loginfo(f"Final logs written to: {self.feature_logger.log_dir}")

            # Trigger ROS shutdown
            rospy.signal_shutdown("User requested shutdown via ENTER key")
        except Exception as e:
            rospy.logwarn(f"Enter key monitor error: {e}")

    def _check_gui_health(self, event):
        """Monitor GUI process health and log warnings if it becomes unresponsive."""
        if not self.use_gui or self.tracker is None:
            return
            
        try:
            # Check if tracker has GUI-related attributes
            if not hasattr(self.tracker, 'gui_dict') or self.tracker.gui_dict is None:
                return
                
            with self.tracker.gui_lock:
                gui_alive = self.tracker.gui_dict.get("gui_alive", True)
                last_heartbeat = self.tracker.gui_dict.get("gui_last_heartbeat", None)
            
            import time
            current_time = time.time()
            
            if not gui_alive:
                if not self.gui_dead_logged:
                    rospy.logwarn(
                        "GUI process has terminated. Pose tracking continues without visualization. "
                        "Consider restarting the node if GUI is needed."
                    )
                    self.gui_dead_logged = True
            elif last_heartbeat is not None:
                time_since_heartbeat = current_time - last_heartbeat
                if time_since_heartbeat > self.gui_heartbeat_timeout:
                    rospy.logwarn_throttle(30.0,
                        f"GUI may be unresponsive (no heartbeat for {time_since_heartbeat:.1f}s). "
                        "Pose tracking continues."
                    )
                    
        except (BrokenPipeError, ConnectionRefusedError, EOFError, OSError) as e:
            if not self.gui_dead_logged:
                rospy.logwarn(f"GUI health check failed (Manager may have terminated): {e}")
                self.gui_dead_logged = True
        except Exception as e:
            rospy.logwarn_throttle(60.0, f"GUI health check error: {e}")

    def _camera_info_callback(self, msg):
        """Store camera intrinsics from CameraInfo message."""
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)
            rospy.loginfo(f"Camera intrinsics received:\n{self.K}")

    # === === === ===
    # Callback function executed at listening of new camera frame
    # === === === ===
    def _rgbd_callback(self, color_msg, depth_msg):
        """Process synchronized RGB-D messages."""
        t_callback_start = time.time()
        msg_timestamp = color_msg.header.stamp.to_sec()
        rospy.loginfo(f"RGBD callback received at {t_callback_start:.3f}, msg timestamp: {msg_timestamp:.3f}, latency: {(t_callback_start - msg_timestamp)*1000:.1f}ms")
        
        if self.K is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera intrinsics...")
            return

        self.frame_count += 1

        # Apply frame stride
        if (self.frame_count - 1) % self.frame_stride != 0:
            return

        try:
            # Convert ROS images to OpenCV format
            color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")

            # Handle depth encoding
            if depth_msg.encoding == "16UC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
                depth = depth.astype(np.float32) / 1000.0  # Convert mm to meters
            elif depth_msg.encoding == "32FC1":
                depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            else:
                rospy.logwarn(f"Unknown depth encoding: {depth_msg.encoding}")
                return

            # Resize if needed
            H0, W0 = color.shape[:2]
            if min(H0, W0) > self.shorter_side:
                scale = self.shorter_side / min(H0, W0)
                new_W = int(W0 * scale)
                new_H = int(H0 * scale)
                color = cv2.resize(
                    color, (new_W, new_H), interpolation=cv2.INTER_LINEAR
                )
                depth = cv2.resize(
                    depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST
                )
                K = self.K.copy()
                K[0, :] *= scale
                K[1, :] *= scale
            else:
                K = self.K.copy()

            H, W = color.shape[:2]

            # Get or create mask using segmenter or fallback to full-image mask
            if self.segmenter is not None:
                # Use SAM3 + Cutie segmenter
                if not self.first_frame_processed:
                    # First frame: use SAM3 for interactive mask generation
                    rospy.loginfo("First frame detected. Starting SAM3 segmentation...")

                    # Save frame to temp file for SAM3 (requires file path)
                    temp_path = "/tmp/bundlesdf_first_frame.png"
                    cv2.imwrite(temp_path, color)

                    # Get mask from SAM3 with text prompt
                    # If target_object is set, use it directly; otherwise prompt via stdin
                    mask = self.segmenter.get_first_frame_mask(
                        temp_path,
                        target_object=self.target_object,
                        wait_for_robot_home=self.with_robot,
                    )

                    if mask is None:
                        rospy.logwarn("SAM3 returned no mask. Using full-image mask.")
                        mask = np.ones((H, W), dtype=np.uint8) * 255
                    else:
                        # Resize mask to match current image size if needed
                        if mask.shape[:2] != (H, W):
                            mask = cv2.resize(
                                mask, (W, H), interpolation=cv2.INTER_NEAREST
                            )
                        rospy.loginfo(f"SAM3 mask obtained. Shape: {mask.shape}")

                    # Publish object_targeted after mask is accepted
                    self.object_targeted_pub.publish(Bool(True))
                    rospy.loginfo("Object targeted. Published /object_targeted = True")

                    self.first_mask = mask
                    self.first_frame_processed = True

                    # Start Enter key monitoring thread now that initialization is complete
                    if self.enable_feature_logging and self.enter_monitor_thread is None:
                        rospy.loginfo("Press ENTER to stop logging and shutdown gracefully")
                        self.enter_monitor_thread = threading.Thread(target=self._monitor_enter_key, daemon=True)
                        self.enter_monitor_thread.start()

                    # Initialize Cutie with the first mask
                    temp_path = "/tmp/bundlesdf_current_frame.png"
                    cv2.imwrite(temp_path, color)
                    mask = self.segmenter.process(
                        temp_path,
                        mask_numpy=self.first_mask,
                        first_frame=True,
                        wait_for_robot_home=self.with_robot,
                    )
                    rospy.loginfo("Cutie segmenter initialized with first mask")
                else:
                    # Subsequent frames: use Cutie for tracking
                    temp_path = "/tmp/bundlesdf_current_frame.png"
                    cv2.imwrite(temp_path, color)
                    mask = self.segmenter.process(temp_path)

                    # Resize mask to match current image size if needed
                    if mask.shape[:2] != (H, W):
                        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
            else:
                # No segmenter: use full-image mask
                if self.first_mask is None:
                    mask = np.ones((H, W), dtype=np.uint8) * 255
                    self.first_mask = mask
                    rospy.loginfo(
                        "Using full-image mask. Consider enabling segmenter with use_segmenter:=true"
                    )
                else:
                    mask = cv2.resize(
                        self.first_mask, (W, H), interpolation=cv2.INTER_NEAREST
                    )

            # Create frame ID string
            id_str = f"{self.frame_count:06d}"

            # Run BundleSDF tracker
            t_before_run = time.time()
            with self.lock:
                self.tracker.run(
                    color=color,
                    depth=depth,
                    K=K,
                    id_str=id_str,
                    mask=mask,
                    occ_mask=None,
                    pose_in_model=np.eye(4),
                )
            t_after_run = time.time()
            run_latency_ms = (t_after_run - t_before_run) * 1000.0
            rospy.loginfo(f"Frame {id_str}: tracker.run() took {run_latency_ms:.1f}ms")

            # Get latest pose from tracker
            if len(self.tracker.bundler._keyframes) > 0:
                latest_frame = self.tracker.bundler._keyframes[-1]
                pose_matrix = np.array(latest_frame._pose_in_model)
                # Invert to get Object Pose in Camera Frame (ob_in_cam)
                pose_matrix = np.linalg.inv(pose_matrix)

                # Publish pose
                self._publish_pose(pose_matrix, color_msg.header.stamp)

                # Publish surface points
                try:
                    surface_points = self._get_masked_surface_points(latest_frame)
                    self._publish_surface_points(surface_points, color_msg.header.stamp)
                except Exception as e_surface:
                    rospy.logwarn_throttle(
                        10.0, f"Surface points publishing failed: {e_surface}"
                    )

                # Feature point logging (for stability investigation)
                if self.enable_feature_logging and (
                    self.frame_count % self.feature_log_stride == 0
                ):
                    try:
                        # Get reference frame for RANSAC matches
                        # Use the second-to-last keyframe as reference if available
                        ref_frame = None
                        if len(self.tracker.bundler._keyframes) >= 2:
                            ref_frame = self.tracker.bundler._keyframes[-2]

                        # Extract feature points
                        ransac_points, ransac_skip_reason = (
                            self._get_ransac_inlier_points(latest_frame, ref_frame)
                        )
                        surface_points = self._get_masked_surface_points(latest_frame)

                        # Log to files
                        self.feature_logger.log_feature_points(
                            frame_id=id_str,
                            timestamp=color_msg.header.stamp.to_sec(),
                            ransac_points=ransac_points,
                            surface_points=surface_points,
                            ransac_skip_reason=ransac_skip_reason,
                        )
                    except Exception as e_log:
                        rospy.logwarn_throttle(
                            10.0, f"Feature point logging failed: {e_log}"
                        )

        except Exception as e:
            rospy.logerr(f"Error processing frame: {e}")
            import traceback

            traceback.print_exc()

    def _publish_pose(self, pose_matrix, stamp):
        """Publish pose as PoseStamped and TF."""
        # Extract rotation and translation
        from scipy.spatial.transform import Rotation

        R = pose_matrix[:3, :3]
        t = pose_matrix[:3, 3]

        # Convert rotation matrix to quaternion
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # [x, y, z, w]

        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = self.camera_frame_id

        pose_msg.pose.position.x = t[0]
        pose_msg.pose.position.y = t[1]
        pose_msg.pose.position.z = t[2]

        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

        # Also broadcast as TF
        t_msg = TransformStamped()
        t_msg.header.stamp = stamp
        t_msg.header.frame_id = self.camera_frame_id
        t_msg.child_frame_id = self.object_frame_id

        t_msg.transform.translation.x = t[0]
        t_msg.transform.translation.y = t[1]
        t_msg.transform.translation.z = t[2]

        t_msg.transform.rotation.x = quat[0]
        t_msg.transform.rotation.y = quat[1]
        t_msg.transform.rotation.z = quat[2]
        t_msg.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t_msg)

    def _normalize_point_cloud(self, points):
        """
        Normalize point cloud: center at origin and scale to unit sphere.

        Args:
            points: numpy array of shape (N, 3) containing XYZ coordinates

        Returns:
            normalized_points: numpy array of shape (N, 3), normalized to [-1, 1] range
        """
        if points is None or len(points) == 0:
            return None

        # Center at centroid
        centroid = points.mean(axis=0)
        centered = points - centroid

        # Scale by max distance from centroid
        distances = np.linalg.norm(centered, axis=1)
        max_dist = distances.max()

        if max_dist < 1e-6:  # Avoid division by zero
            return centered

        normalized = centered / max_dist
        return normalized

    def _create_pointcloud2_msg(self, points, stamp, frame_id):
        """
        Create PointCloud2 message from numpy array.

        Args:
            points: numpy array of shape (N, 3) containing XYZ coordinates
            stamp: ROS timestamp
            frame_id: coordinate frame ID

        Returns:
            PointCloud2 message
        """
        # Create header
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        # Define fields (x, y, z as float32)
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Pack point data into binary format
        cloud_data = []
        for point in points:
            cloud_data.append(
                struct.pack("fff", float(point[0]), float(point[1]), float(point[2]))
            )
        cloud_bytes = b"".join(cloud_data)

        # Create PointCloud2 message
        pc2_msg = PointCloud2()
        pc2_msg.header = header
        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.fields = fields
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 12  # 3 * float32 (4 bytes each)
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.data = cloud_bytes
        pc2_msg.is_dense = True

        return pc2_msg

    def _publish_surface_points(self, surface_points, stamp):
        """
        Publish both raw and normalized surface points as PointCloud2.

        Args:
            surface_points: numpy array of shape (N, 3) containing XYZ coordinates
            stamp: ROS timestamp
        """
        if surface_points is None or len(surface_points) == 0:
            return

        # Random sampling if we have more points than requested
        num_points = len(surface_points)
        if num_points > self.surface_point_sample_count:
            indices = np.random.choice(
                num_points, self.surface_point_sample_count, replace=False
            )
            sampled_points = surface_points[indices]
        else:
            sampled_points = surface_points

        # Publish raw point cloud
        raw_msg = self._create_pointcloud2_msg(
            sampled_points, stamp, self.camera_frame_id
        )
        self.surface_points_pub.publish(raw_msg)

        # Normalize and publish normalized point cloud
        normalized_points = self._normalize_point_cloud(sampled_points)
        if normalized_points is not None:
            normalized_msg = self._create_pointcloud2_msg(
                normalized_points, stamp, self.camera_frame_id
            )
            self.surface_points_normalized_pub.publish(normalized_msg)

    def _get_masked_surface_points(self, frame):
        """Extract 3D points within the foreground mask from frame's point cloud."""
        try:
            # Get full organized point cloud (H*W, D)
            cloud = frame.pointcloud()  # Returns Eigen::MatrixXf as numpy

            # Get frame dimensions
            H, W = frame._H, frame._W

            # Reshape to (H, W, D)
            cloud_img = cloud.reshape(H, W, -1)

            # Get mask and ensure it's 2D
            mask = np.array(frame._fg_mask)
            mask = np.squeeze(mask)  # Remove any extra dimensions
            if mask.ndim != 2:
                mask = mask.reshape(H, W)

            # Extract XYZ columns (columns 0, 1, 2)
            xyz = cloud_img[:, :, :3]

            # Ensure xyz has shape (H, W, 3) by removing extra dimensions
            if xyz.shape[-1] == 3 and len(xyz.shape) == 4:
                xyz = xyz.reshape(H, W, 3)

            # Extract z channel and ensure it's 2D
            z_channel = xyz[:, :, 2]
            z_channel = np.squeeze(z_channel)  # Remove any extra dimensions
            if z_channel.ndim != 2:
                z_channel = z_channel.reshape(H, W)

            # Ensure both arrays are 2D with matching shapes
            assert mask.shape == (H, W), f"mask shape {mask.shape} != ({H}, {W})"
            assert z_channel.shape == (H, W), f"z_channel shape {z_channel.shape} != ({H}, {W})"

            # Filter: mask > 0 AND z > 0.1 (valid depth)
            valid = (mask > 0) & (z_channel > 0.1)

            points_3d = xyz[valid]
            return points_3d  # Shape: (N_valid, 3)
        except Exception as e:
            rospy.logwarn(f"Failed to get masked surface points: {e}")
            return None

    def _get_ransac_inlier_points(self, frame, ref_frame):
        """Extract RANSAC inlier 3D points by reprojecting 2D correspondences."""
        try:
            # Check if reference frame exists
            if ref_frame is None:
                return None, "no_reference_frame"

            # Get matches between frames
            pair = (ref_frame, frame)  # Note: order matters for frame pairs
            if pair not in self.tracker.bundler._fm._matches:
                # Try reversed pair
                pair = (frame, ref_frame)
                if pair not in self.tracker.bundler._fm._matches:
                    return None, "no_matches_found"

            matches = self.tracker.bundler._fm._matches[pair]
            if len(matches) == 0:
                return None, "no_matches_found"

            # Get depth and camera intrinsics
            depth = np.array(frame._depth).squeeze()  # Remove extra dimensions
            K = np.array(frame._K).squeeze() if hasattr(frame, "_K") else np.array(self.K).squeeze()

            # Reproject 2D correspondences to 3D using depth
            points_3d = []
            for corr in matches:
                u, v = int(corr._uA), int(corr._vA)
                if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                    z = float(depth[v, u])  # Ensure scalar
                    if z > 0.1:  # Valid depth threshold
                        x = float((u - K[0, 2]) * z / K[0, 0])
                        y = float((v - K[1, 2]) * z / K[1, 1])
                        points_3d.append([x, y, z])

            if len(points_3d) == 0:
                return None, "empty_points"

            return np.array(points_3d, dtype=np.float32), ""
        except Exception as e:
            rospy.logwarn(f"Failed to get RANSAC inlier points: {e}")
            return None, "extraction_error"

    def shutdown(self):
        """Clean shutdown of tracker."""
        rospy.loginfo("Shutting down BundleSDF node...")

        # Write final summary if feature logging is enabled
        if self.enable_feature_logging and self.feature_logger:
            try:
                with open(self.feature_logger.log_path, "a") as f:
                    if not self.shutdown_requested:  # Only write if not already written
                        f.write("\n" + "=" * 60 + "\n")
                        f.write("=== SHUTDOWN: Feature point logging completed ===\n")
                        f.write(f"Total frames processed: {self.frame_count}\n")
                        f.write("=" * 60 + "\n")
                        f.flush()
                rospy.loginfo(f"Feature point logs saved to: {self.feature_logger.log_dir}")
            except Exception as e:
                rospy.logwarn(f"Failed to write final log summary: {e}")

        if self.tracker is not None:
            self.tracker.on_finish()

    def run(self):
        """Main run loop."""
        rospy.on_shutdown(self.shutdown)
        rospy.spin()


def main():
    try:
        node = BundleSdfNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
