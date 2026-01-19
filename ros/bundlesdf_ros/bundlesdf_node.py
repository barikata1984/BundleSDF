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
import numpy as np
import cv2
import yaml

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import message_filters
import tf2_ros
from geometry_msgs.msg import TransformStamped

# Add BundleSDF to Python path
BUNDLESDF_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, BUNDLESDF_DIR)

from bundlesdf import BundleSdf, set_seed
from segmentation_utils import Segmenter


class BundleSdfNode:
    """ROS Node wrapper for BundleSDF object pose tracking."""

    def __init__(self):
        rospy.init_node("bundlesdf_node", anonymous=False)

        # Parameters
        self.output_dir = rospy.get_param("~output_dir", "/tmp/bundlesdf_output")
        self.use_segmenter = rospy.get_param("~use_segmenter", False)
        self.use_gui = rospy.get_param("~use_gui", False)
        self.shorter_side = rospy.get_param("~shorter_side", 480)
        self.frame_stride = rospy.get_param("~frame_stride", 1)
        self.debug_level = rospy.get_param("~debug_level", 2)

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

        # Initialize segmenter if enabled
        if self.use_segmenter:
            rospy.loginfo("Initializing Segmenter (SAM3 + Cutie)...")
            self.segmenter = Segmenter()
            rospy.loginfo("Segmenter initialized")
        else:
            self.segmenter = None

        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publisher
        self.pose_pub = rospy.Publisher("~object_pose", PoseStamped, queue_size=10)

        # Setup config and tracker
        self._setup_tracker()

        # Subscribers with message_filters for synchronized callback
        self.color_sub = message_filters.Subscriber(self.color_topic, Image)
        self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self._camera_info_callback, queue_size=1
        )

        # Approximate time synchronizer for RGB-D
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], queue_size=10, slop=0.1
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

        cfg_bundletrack["SPDLOG"] = int(self.debug_level)
        cfg_bundletrack["depth_processing"]["percentile"] = 95
        cfg_bundletrack["erode_mask"] = 3
        cfg_bundletrack["debug_dir"] = self.output_dir + "/"
        cfg_bundletrack["bundle"]["max_BA_frames"] = 10
        cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
        cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
        cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
        cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
        cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
        cfg_bundletrack["feature_corres"]["map_points"] = True
        cfg_bundletrack["feature_corres"]["resize"] = 400
        cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
        cfg_bundletrack["keyframe"]["min_rot"] = 5
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
        cfg_nerf["mesh_resolution"] = 0.005
        cfg_nerf["down_scale_ratio"] = 1
        cfg_nerf["fs_sdf"] = 0.1
        cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
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

    def _camera_info_callback(self, msg):
        """Store camera intrinsics from CameraInfo message."""
        if self.K is None:
            self.K = np.array(msg.K).reshape(3, 3)
            rospy.loginfo(f"Camera intrinsics received:\n{self.K}")

    def _rgbd_callback(self, color_msg, depth_msg):
        """Process synchronized RGB-D messages."""
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
                        temp_path, target_object=self.target_object
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

                    self.first_mask = mask
                    self.first_frame_processed = True

                    # Initialize Cutie with the first mask
                    temp_path = "/tmp/bundlesdf_current_frame.png"
                    cv2.imwrite(temp_path, color)
                    mask = self.segmenter.process(
                        temp_path, mask_numpy=self.first_mask, first_frame=True
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

            # Get latest pose from tracker
            if len(self.tracker.bundler._keyframes) > 0:
                latest_frame = self.tracker.bundler._keyframes[-1]
                pose_matrix = np.array(latest_frame._pose_in_model)

                # Publish pose
                self._publish_pose(pose_matrix, color_msg.header.stamp)

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

    def shutdown(self):
        """Clean shutdown of tracker."""
        rospy.loginfo("Shutting down BundleSDF node...")
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
