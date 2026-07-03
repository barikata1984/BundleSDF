#!/usr/bin/env python3
# Online BundleSDF tracking node.
# Subscribes time-synchronized rgb/depth/mask, feeds each frame to BundleSdf.run(),
# and publishes the tracked object's pose as PoseStamped + TF.
# See README.md for the topic contract.

import os
import sys

import cv2
import message_filters
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

code_dir = os.path.dirname(os.path.realpath(__file__))
repo_dir = os.path.abspath(f'{code_dir}/../../..')
sys.path.append(repo_dir)

import ruamel.yaml  # noqa: E402

from bundlesdf import BundleSdf, set_seed  # noqa: E402

yaml = ruamel.yaml.YAML()


def decode_bgr(msg):
  # np.frombuffer instead of cv_bridge; honor row stride (step) and encoding.
  buf = np.frombuffer(msg.data, dtype=np.uint8)
  img = buf.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
  if msg.encoding == 'rgb8':
    img = img[:, :, ::-1]
  elif msg.encoding != 'bgr8':
    raise ValueError(f'unsupported color encoding: {msg.encoding} (expected rgb8/bgr8)')
  return np.ascontiguousarray(img)


def decode_depth(msg):
  # BundleSdf.run() expects depth in meters (see BundleTrack/scripts/data_reader.py::get_depth).
  if msg.encoding == '16UC1':
    depth_mm = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    return depth_mm.astype(np.float32) / 1e3
  if msg.encoding == '32FC1':
    return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
  raise ValueError(f'unsupported depth encoding: {msg.encoding} (expected 16UC1/32FC1)')


def decode_mask(msg):
  mask = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
  return (mask > 0).astype(np.uint8)


class BundleSdfNode:
  def __init__(self):
    set_seed(0)

    out_folder = rospy.get_param('~out_folder', '/tmp/bundlesdf_online')
    os.system(f'rm -rf {out_folder} && mkdir -p {out_folder}')

    # Config block mirrored from run_custom.py::run_one_video() / scripts/bench_milk.py.
    cfg_bundletrack = yaml.load(open(f'{repo_dir}/BundleTrack/config_ho3d.yml', 'r'))
    cfg_bundletrack['SPDLOG'] = rospy.get_param('~debug_level', 1)
    cfg_bundletrack['depth_processing']['percentile'] = 95
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

    cfg_nerf = yaml.load(open(f'{repo_dir}/config.yml', 'r'))
    cfg_nerf['continual'] = True
    cfg_nerf['trunc_start'] = 0.01
    cfg_nerf['trunc'] = 0.01
    cfg_nerf['mesh_resolution'] = 0.005
    cfg_nerf['down_scale_ratio'] = 1
    cfg_nerf['fs_sdf'] = 0.1
    cfg_nerf['far'] = cfg_bundletrack['depth_processing']['zfar']
    cfg_nerf['datadir'] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf['notes'] = ''
    cfg_nerf['expname'] = 'nerf_with_bundletrack_online'
    cfg_nerf['save_dir'] = cfg_nerf['datadir']
    cfg_nerf_dir = f'{out_folder}/config_nerf.yml'
    yaml.dump(cfg_nerf, open(cfg_nerf_dir, 'w'))

    self.erode_mask = cfg_bundletrack['erode_mask']
    self.debug_dir = cfg_bundletrack['debug_dir']
    self.target_frame = rospy.get_param('~target_frame', 'tracked_object')
    self.K = None

    rospy.loginfo('loading BundleSdf tracker, debug_dir=%s', self.debug_dir)
    self.tracker = BundleSdf(
      cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir, start_nerf_keyframes=5, use_gui=False)

    self.pose_pub = rospy.Publisher('~object_pose', PoseStamped, queue_size=1)
    self.tf_broadcaster = tf2_ros.TransformBroadcaster()

    self.info_sub = rospy.Subscriber('~camera_info_in', CameraInfo, self.on_camera_info, queue_size=1)
    rgb_sub = message_filters.Subscriber('~rgb_in', Image)
    depth_sub = message_filters.Subscriber('~depth_in', Image)
    mask_sub = message_filters.Subscriber('~mask_in', Image)
    self.sync = message_filters.ApproximateTimeSynchronizer(
      [rgb_sub, depth_sub, mask_sub], queue_size=5, slop=0.05)
    self.sync.registerCallback(self.on_frame)
    rospy.loginfo('bundlesdf_node ready; waiting for camera_info and synced rgb/depth/mask')

  def on_camera_info(self, msg):
    self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)

  def on_frame(self, rgb_msg, depth_msg, mask_msg):
    if self.K is None:
      rospy.logwarn_throttle(5, 'camera_info not received yet, dropping frame')
      return

    color = decode_bgr(rgb_msg)
    depth = decode_depth(depth_msg)
    mask = decode_mask(mask_msg)
    if self.erode_mask > 0:
      kernel = np.ones((self.erode_mask, self.erode_mask), np.uint8)
      mask = cv2.erode(mask, kernel)

    id_str = str(rgb_msg.header.stamp.to_nsec())

    self.tracker.run(color, depth, self.K, id_str, mask=mask, occ_mask=None, pose_in_model=np.eye(4))

    pose_file = f'{self.debug_dir}/ob_in_cam/{id_str}.txt'
    if not os.path.exists(pose_file):
      rospy.logwarn('frame %s produced no pose (tracker.run did not reach save), skipping publish', id_str)
      return

    ob_in_cam = np.loadtxt(pose_file).reshape(4, 4)
    self.publish_pose(ob_in_cam, rgb_msg.header)

  def publish_pose(self, ob_in_cam, header):
    quat = Rotation.from_matrix(ob_in_cam[:3, :3]).as_quat()  # (x, y, z, w)
    t = ob_in_cam[:3, 3]

    pose_msg = PoseStamped()
    pose_msg.header = header
    pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z = t
    (pose_msg.pose.orientation.x, pose_msg.pose.orientation.y,
     pose_msg.pose.orientation.z, pose_msg.pose.orientation.w) = quat
    self.pose_pub.publish(pose_msg)

    tf_msg = TransformStamped()
    tf_msg.header = header
    tf_msg.child_frame_id = self.target_frame
    (tf_msg.transform.translation.x, tf_msg.transform.translation.y,
     tf_msg.transform.translation.z) = t
    (tf_msg.transform.rotation.x, tf_msg.transform.rotation.y,
     tf_msg.transform.rotation.z, tf_msg.transform.rotation.w) = quat
    self.tf_broadcaster.sendTransform(tf_msg)


def main():
  rospy.init_node('bundlesdf_node')
  BundleSdfNode()
  rospy.spin()


if __name__ == '__main__':
  main()
