#!/usr/bin/env python3
# SAM 3 streaming segmentation node.
# Subscribes RGB, tracks a single text-prompted object, publishes a mono8 mask.
# See README.md for the topic contract.

import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor


def decode_rgb(msg):
  # np.frombuffer instead of cv_bridge; honor row stride (step) and encoding.
  buf = np.frombuffer(msg.data, dtype=np.uint8)
  img = buf.reshape(msg.height, msg.step)[:, : msg.width * 3]
  img = img.reshape(msg.height, msg.width, 3)
  if msg.encoding == 'bgr8':
    img = img[:, :, ::-1]
  elif msg.encoding != 'rgb8':
    raise ValueError('unsupported encoding: %s (expected rgb8/bgr8)' % msg.encoding)
  return np.ascontiguousarray(img)


class Sam3SegmenterNode:
  def __init__(self):
    self.text_prompt = rospy.get_param('~text_prompt', '')
    if not self.text_prompt:
      raise rospy.ROSInitException('~text_prompt is required')

    model_id = rospy.get_param('~model_id', 'facebook/sam3')
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32

    rospy.loginfo('loading %s on %s', model_id, self.device)
    self.model = Sam3VideoModel.from_pretrained(model_id).to(self.device, dtype=self.dtype)
    self.processor = Sam3VideoProcessor.from_pretrained(model_id)
    self.session = self.processor.init_video_session(
      inference_device=self.device,
      processing_device='cpu',
      video_storage_device='cpu',
    )
    self.session = self.processor.add_text_prompt(
      inference_session=self.session, text=self.text_prompt)

    self.pub = rospy.Publisher('~mask_out', Image, queue_size=1)
    # buff_size large so stale frames are dropped rather than queued.
    self.sub = rospy.Subscriber(
      '~image_in', Image, self.callback, queue_size=1, buff_size=2 ** 24)
    rospy.loginfo('sam3_segmenter ready; prompt=%r', self.text_prompt)

  def select_mask(self, processed, height, width):
    masks = processed['masks']
    if masks is None or len(masks) == 0:
      return np.zeros((height, width), dtype=np.uint8)
    masks = masks.to('cpu').numpy().astype(bool)
    scores = processed.get('scores')
    if scores is None:
      idx = int(np.argmax(masks.reshape(len(masks), -1).sum(axis=1)))
    else:
      scores = scores.to('cpu').numpy()
      top = np.flatnonzero(scores == scores.max())
      # score-max, breaking ties by mask area.
      idx = int(top[np.argmax([masks[i].sum() for i in top])])
    return (masks[idx].astype(np.uint8)) * 255

  def make_mask_msg(self, mask, header):
    out = Image()
    out.header = header
    out.height, out.width = mask.shape
    out.encoding = 'mono8'
    out.is_bigendian = 0
    out.step = out.width
    out.data = mask.tobytes()
    return out

  def callback(self, msg):
    frame = decode_rgb(msg)
    inputs = self.processor(images=frame, device=self.device, return_tensors='pt').to(
      self.model.device)
    with torch.inference_mode():
      outputs = self.model(
        inference_session=self.session, frame=inputs.pixel_values[0], reverse=False)
    processed = self.processor.postprocess_outputs(
      self.session, outputs, original_sizes=inputs.original_sizes)
    mask = self.select_mask(processed, msg.height, msg.width)
    self.pub.publish(self.make_mask_msg(mask, msg.header))


def main():
  rospy.init_node('sam3_segmenter')
  Sam3SegmenterNode()
  rospy.spin()


if __name__ == '__main__':
  main()
