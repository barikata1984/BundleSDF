#!/usr/bin/env python3
# SAM 3 streaming segmentation node.
# Subscribes RGB, tracks a single text-prompted object, publishes a mono8 mask.
# See README.md for the topic contract.

import threading

import cv2
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
    if msg.encoding == "bgr8":
        img = img[:, :, ::-1]
    elif msg.encoding != "rgb8":
        raise ValueError("unsupported encoding: %s (expected rgb8/bgr8)" % msg.encoding)
    return np.ascontiguousarray(img)


class Sam3SegmenterNode:
    def __init__(self):
        self.text_prompt = rospy.get_param("~text_prompt", "")
        if not self.text_prompt:
            raise rospy.ROSInitException("~text_prompt is required")

        model_id = rospy.get_param("~model_id", "facebook/sam3")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        rospy.loginfo("loading %s on %s", model_id, self.device)
        self.model = Sam3VideoModel.from_pretrained(model_id).to(
            self.device, dtype=self.dtype
        )
        self.processor = Sam3VideoProcessor.from_pretrained(model_id)
        # Guards self.session / self.text_prompt: the gate thread rebuilds them
        # while the callback thread runs inference on them.
        self.session_lock = threading.Lock()
        self.session = self.processor.init_video_session(
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device=self.device,
            dtype=self.dtype,
        )
        self.session = self.processor.add_text_prompt(
            inference_session=self.session, text=self.text_prompt
        )

        self.check_segmentation = rospy.get_param("~check_segmentation", True)
        # Gate: without the check, publish immediately (legacy behavior); with
        # it, hold publishing until the user accepts the segmentation.
        self.accepted = not self.check_segmentation
        # Latest debug image (BGR) handed off to the main thread for display.
        self.debug_lock = threading.Lock()
        self.debug_frame = None
        if self.check_segmentation:
            self.debug_window = "sam3_segmenter prompt check: %r" % self.text_prompt
            cv2.namedWindow(self.debug_window, cv2.WINDOW_NORMAL)

        image_in = rospy.get_param("/camera_input/rgb_in")
        mask_out = rospy.get_param("/camera_input/mask_topic")
        self.pub = rospy.Publisher(mask_out, Image, queue_size=1)
        # buff_size large so stale frames are dropped rather than queued.
        self.sub = rospy.Subscriber(
            image_in, Image, self.callback, queue_size=1, buff_size=2**24
        )
        rospy.loginfo("sam3_segmenter ready; prompt=%r", self.text_prompt)

    def select_mask(self, processed, height, width):
        masks = processed["masks"]
        if masks is None or len(masks) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        masks = masks.to("cpu").numpy().astype(bool, copy=False)
        scores = processed.get("scores")
        if scores is None:
            idx = int(np.argmax(masks.reshape(len(masks), -1).sum(axis=1)))
        else:
            scores = scores.to("cpu").numpy()
            top = np.flatnonzero(scores == scores.max())
            # score-max, breaking ties by mask area.
            idx = int(top[np.argmax([masks[i].sum() for i in top])])
        return (masks[idx].astype(np.uint8)) * 255

    def show_debug(self, frame, mask):
        mask_bool = mask.astype(bool)
        green = np.zeros_like(frame)
        green[:] = (0, 255, 0)
        tracked = frame.copy()
        blended = cv2.addWeighted(frame, 0.5, green, 0.5, 0)
        tracked[mask_bool] = blended[mask_bool]
        vis = np.hstack([frame, tracked])
        # Do NOT call imshow/waitKey here: this runs on the subscriber callback
        # thread, and OpenCV's Qt backend deadlocks when imshow is called off the
        # GUI main thread. Hand the frame to main() for display instead.
        bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        with self.debug_lock:
            self.debug_frame = bgr

    def make_mask_msg(self, mask, header):
        out = Image()
        out.header = header
        out.height, out.width = mask.shape
        out.encoding = "mono8"
        out.is_bigendian = 0
        out.step = out.width
        out.data = mask.tobytes()
        return out

    def callback(self, msg):
        frame = decode_rgb(msg)
        inputs = self.processor(
            images=frame, device=self.device, return_tensors="pt"
        ).to(self.model.device)
        # Hold session_lock across inference so the gate thread cannot swap the
        # session mid-callback. Also read self.accepted under the same lock, so
        # it is consistent with the session that produced this mask.
        with self.session_lock:
            with torch.inference_mode():
                outputs = self.model(
                    inference_session=self.session,
                    frame=inputs.pixel_values[0],
                    reverse=False,
                )
            processed = self.processor.postprocess_outputs(
                self.session, outputs, original_sizes=inputs.original_sizes
            )
            accepted = self.accepted
        mask = self.select_mask(processed, msg.height, msg.width)
        # Always refresh the debug view so the user sees the current segmentation
        # while deciding; only publish once the gate is open.
        if self.check_segmentation:
            self.show_debug(frame, mask)
        if accepted:
            self.pub.publish(self.make_mask_msg(mask, msg.header))

    def gate_loop(self):
        # Runs on a dedicated daemon thread while check_segmentation is on.
        # Blocks on stdin; accepting opens the publish gate for good, rejecting
        # rebuilds the session with a new prompt and keeps the gate closed.
        while not rospy.is_shutdown():
            answer = input("Accept segmentation? [y/n]: ").strip().lower()
            if answer == "y":
                with self.session_lock:
                    self.accepted = True
                rospy.loginfo("segmentation accepted; publishing mask")
                return
            if answer != "n":
                continue
            new_prompt = ""
            while not new_prompt:
                new_prompt = input("New object prompt: ").strip()
            with self.session_lock:
                self.session = self.processor.init_video_session(
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device=self.device,
                    dtype=self.dtype,
                )
                self.session = self.processor.add_text_prompt(
                    inference_session=self.session, text=new_prompt
                )
                self.text_prompt = new_prompt
                self.accepted = False
            rospy.loginfo("prompt updated to %r", new_prompt)


def main():
    rospy.init_node("sam3_segmenter")
    node = Sam3SegmenterNode()
    if not node.check_segmentation:
        rospy.spin()
        return
    # Gate the stream on user confirmation. This blocks on stdin, so run it off
    # the main thread, which stays free for imshow/waitKey below.
    threading.Thread(target=node.gate_loop, daemon=True).start()
    # imshow/waitKey must run on the main thread. Pull the latest debug frame
    # produced by the callback thread and display it here.
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        with node.debug_lock:
            frame = node.debug_frame
        if frame is not None:
            cv2.imshow(node.debug_window, frame)
            cv2.waitKey(1)
        rate.sleep()


if __name__ == "__main__":
    main()
