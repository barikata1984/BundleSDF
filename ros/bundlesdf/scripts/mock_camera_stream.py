#!/usr/bin/env python3
# Mock camera publisher for smoke-testing bundlesdf_node.py without a real D455.
# Replays recorded rgb/depth/mask frames from a YcbineoatReader-style video dir
# (see BundleTrack/scripts/data_reader.py) as sensor_msgs/Image + CameraInfo,
# matching the encodings that bundlesdf_node.py's decode_* functions expect.

import glob
import os

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


def make_image(data_bytes, height, width, step, encoding, header):
  msg = Image()
  msg.header = header
  msg.height = height
  msg.width = width
  msg.encoding = encoding
  msg.is_bigendian = 0
  msg.step = step
  msg.data = data_bytes
  return msg


class MockCameraStream:
  def __init__(self):
    self.video_dir = rospy.get_param('~video_dir', '/workspace/data/2022-11-18-15-10-24_milk')
    self.max_frames = int(rospy.get_param('~max_frames', 20))
    self.rate_hz = float(rospy.get_param('~rate', 2.0))
    self.frame_id = rospy.get_param('~frame_id', 'camera_color_optical_frame')

    rgb_out = rospy.get_param('~rgb_out', '/d455_1/color/image_rect')
    depth_out = rospy.get_param('~depth_out', '/d455_1/aligned_depth_to_color/image_raw')
    mask_out = rospy.get_param('~mask_out', '/sam3/mask')
    info_out = rospy.get_param('~camera_info_out', '/d455_1/color/camera_info_rect')

    self.color_files = sorted(glob.glob(f'{self.video_dir}/rgb/*.png'))
    if not self.color_files:
      raise RuntimeError(f'no rgb/*.png found under {self.video_dir}')
    if self.max_frames > 0:
      self.color_files = self.color_files[: self.max_frames]

    self.K = np.loadtxt(f'{self.video_dir}/cam_K.txt').reshape(3, 3)
    first = cv2.imread(self.color_files[0])
    self.height, self.width = first.shape[:2]

    self.rgb_pub = rospy.Publisher(rgb_out, Image, queue_size=1)
    self.depth_pub = rospy.Publisher(depth_out, Image, queue_size=1)
    self.mask_pub = rospy.Publisher(mask_out, Image, queue_size=1)
    self.info_pub = rospy.Publisher(info_out, CameraInfo, queue_size=1, latch=True)

    rospy.loginfo(
      'mock_camera_stream: video_dir=%s frames=%d rate=%.2fHz size=%dx%d',
      self.video_dir, len(self.color_files), self.rate_hz, self.width, self.height)

  def build_camera_info(self, header):
    msg = CameraInfo()
    msg.header = header
    msg.height = self.height
    msg.width = self.width
    msg.distortion_model = 'plumb_bob'
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.K = self.K.reshape(-1).tolist()
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    fx, fy = self.K[0, 0], self.K[1, 1]
    cx, cy = self.K[0, 2], self.K[1, 2]
    msg.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg

  def load_frame(self, color_file):
    base = os.path.basename(color_file)
    bgr = cv2.imread(color_file, cv2.IMREAD_COLOR)  # BGR, uint8
    depth = cv2.imread(f'{self.video_dir}/depth/{base}', cv2.IMREAD_UNCHANGED)  # uint16, mm
    mask = cv2.imread(f'{self.video_dir}/masks/{base}', cv2.IMREAD_UNCHANGED)
    if mask.ndim == 3:
      mask = mask.sum(axis=-1)
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    return (
      np.ascontiguousarray(bgr, dtype=np.uint8),
      np.ascontiguousarray(depth, dtype=np.uint16),
      np.ascontiguousarray(mask, dtype=np.uint8),
    )

  def spin(self):
    rate = rospy.Rate(self.rate_hz)
    # Publish CameraInfo once up-front so the subscriber has intrinsics ready.
    self.info_pub.publish(self.build_camera_info(self.stamped_header()))

    for i, color_file in enumerate(self.color_files):
      if rospy.is_shutdown():
        return
      bgr, depth, mask = self.load_frame(color_file)
      header = self.stamped_header()

      self.info_pub.publish(self.build_camera_info(header))
      self.rgb_pub.publish(
        make_image(bgr.tobytes(), self.height, self.width, self.width * 3, 'bgr8', header))
      self.depth_pub.publish(
        make_image(depth.tobytes(), self.height, self.width, self.width * 2, '16UC1', header))
      self.mask_pub.publish(
        make_image(mask.tobytes(), self.height, self.width, self.width, 'mono8', header))
      rospy.loginfo_throttle(2.0, 'mock_camera_stream: published frame %d/%d (%s)',
                             i + 1, len(self.color_files), os.path.basename(color_file))
      rate.sleep()

    rospy.loginfo('mock_camera_stream: replayed %d frames, done (idling)', len(self.color_files))
    rospy.spin()

  def stamped_header(self):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = self.frame_id
    return header


def main():
  rospy.init_node('mock_camera_stream')
  MockCameraStream().spin()


if __name__ == '__main__':
  main()
