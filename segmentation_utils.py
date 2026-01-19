# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

# Lazy import for SAM3 to avoid overhead if not used


class Segmenter:
    def __init__(self):
        # obtain the Cutie model with default parameters -- skipping hydra configuration
        self.cutie = get_default_model()
        # Typically, use one InferenceCore per video
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        # the processor matches the shorter edge of the input to this size
        # you might want to experiment with different sizes, -1 keeps the original size
        self.processor.max_internal_size = 480
        self.sam3 = None

    def _load_sam3(self):
        if self.sam3 is None:
            print("Loading SAM3 model for first frame segmentation...")
            try:
                import sys

                # Add third_party/sam3 to python path if not present
                sam3_pkg_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "third_party", "sam3"
                )
                if sam3_pkg_path not in sys.path:
                    sys.path.append(sam3_pkg_path)

                from sam3 import build_sam3_image_model

                from sam3.model.sam3_image_processor import Sam3Processor

                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_sam3_image_model(device=device)
                self.sam3 = Sam3Processor(model, device=device)
                print("SAM3 model loaded.")
            except Exception as e:
                print(f"Error loading SAM3: {e}")
                raise e
        return self.sam3

    def get_first_frame_mask(self, image_path, target_object=""):
        """
        Get the first frame mask using SAM3 with a text prompt.

        Args:
            image_path: Path to the image file.
            target_object: Optional text prompt for segmentation. If provided,
                           skips stdin input and uses this directly.
        """
        import subprocess
        import time
        import sys

        self._load_sam3()

        viewer_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "utils", "view_image.py"
        )

        # Load image for display and processing
        # Note: cv2.imread loads as BGR
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Save image to temp file instead of using cv2.imshow to avoid GUI conflicts
        # when BundleSDF GUI is running in a separate process
        # Encode image to bytes for pipe transfer
        success, buffer = cv2.imencode(".png", image_bgr)
        if not success:
            print("Error: Could not encode image for viewer.")
        else:
            image_bytes = buffer.tobytes()
            try:
                # Use Popen with stdin=PIPE to send image data
                proc = subprocess.Popen(
                    ["python3", viewer_script, "stdin", "Initial Frame"],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                proc.stdin.write(image_bytes)
                proc.stdin.close()

                # Pause to let viewer initialize. 2.0s should be enough for most GUI backends to settle.
                time.sleep(2.0)
            except Exception as e:
                print(f"Warning: Could not launch image viewer: {e}")

        # Use provided target_object as initial prompt, or None to trigger stdin
        initial_prompt = target_object.strip() if target_object else None

        # Loop until user is satisfied with the mask
        while True:
            # Ensure stdout is flushed before printing prompt
            sys.stdout.flush()

            if initial_prompt:
                # Use the CLI-provided target object (first iteration only)
                prompt = initial_prompt
                print("\n" + "=" * 50)
                print(f"Using CLI-provided target object: '{prompt}'")
                print("=" * 50 + "\n")
                initial_prompt = None  # Clear so next iteration uses stdin
            else:
                # Interactive mode: ask user for input
                print("\n" + "=" * 50)
                print("Input image displayed in popup window.")
                print(
                    "Enter a text prompt to segment the object (e.g., 'milk carton', 'hand', 'cat')"
                )
                print("=" * 50 + "\n")

                prompt = input("Enter text prompt: ").strip()
                if not prompt:
                    print("Empty prompt provided. Using default.")

            # Convert to RGB for SAM3 (which uses PIL/RGB internally via set_image)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            state = self.sam3.set_image(image_pil)
            state = self.sam3.set_text_prompt(prompt, state)

            # state["masks"] is a boolean tensor of shape [N, H, W]
            masks = state.get("masks", None)

            if masks is None or masks.numel() == 0:
                print("No objects found by SAM3.")
                combined_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            else:
                combined_mask = masks.any(dim=0).cpu().numpy().astype(np.uint8)

            vis_image = image_bgr.copy()

            if combined_mask.ndim > 2:
                combined_mask = combined_mask.squeeze()

            if combined_mask.shape[:2] != vis_image.shape[:2]:
                print(
                    f"Warning: Mask shape {combined_mask.shape} does not match image shape {vis_image.shape[:2]}. Resizing mask."
                )
                combined_mask = cv2.resize(
                    combined_mask,
                    (vis_image.shape[1], vis_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

            if combined_mask.max() > 0:
                red_mask = np.zeros_like(vis_image)
                red_mask[:, :, 2] = 255  # Red channel
                mask_bool = combined_mask > 0
                vis_image[mask_bool] = cv2.addWeighted(
                    vis_image[mask_bool], 0.5, red_mask[mask_bool], 0.5, 0
                ).reshape(-1, 3)
                print("Object found! Mask overlay created.")
            else:
                print("Result is empty (no object found). Saving raw image.")

            if combined_mask is not None:
                raw_mask_path = os.path.abspath("mask_result_raw.png")
                raw_mask_vis = (combined_mask * 255).astype(np.uint8)
                cv2.imwrite(raw_mask_path, raw_mask_vis)
                print(f"Raw mask saved to: {raw_mask_path}")

            # Prepare visualization for viewer
            success, buffer = cv2.imencode(".png", vis_image)
            if success:
                image_bytes = buffer.tobytes()
                # Display the result mask in a separate window (non-blocking)
                try:
                    proc = subprocess.Popen(
                        ["python3", viewer_script, "stdin", "Predicted Mask"],
                        stdin=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    proc.stdin.write(image_bytes)
                    proc.stdin.close()

                    # Pause to let viewer logs (Qt warnings etc) print before asking for validation
                    time.sleep(2.0)
                except Exception as e:
                    print(f"Warning: Could not launch mask viewer: {e}")
            else:
                print("Error: Could not encode mask preview.")

            # --- Validation Check ---
            # Ensure stdout is flushed so prompt appears at bottom
            sys.stdout.flush()
            print("\n" + "=" * 50)
            valid = input("Is this mask acceptable? (y/n): ").strip().lower()
            if valid == "y" or valid == "yes":
                print("Mask accepted. Proceeding...")
                break
            else:
                print("Mask rejected. Please try another prompt.")
                print("=" * 50 + "\n")

        return (combined_mask * 255).astype(np.uint8)

    @torch.inference_mode()
    @torch.amp.autocast("cuda")
    def process(self, image_path, mask_numpy=None, first_frame=False):
        # load the image as RGB; normalization is done within the model
        image = Image.open(image_path)
        image = to_tensor(image).cuda().float()

        if first_frame and mask_numpy is None:
            # Use SAM3 to get the mask if it's the first frame and no mask is provided
            mask_numpy = self.get_first_frame_mask(image_path)

            # Check for early exit signal (mask preview)
            # if mask_numpy is None:
            #    print("Early exit from Segmenter process.")
            #    return None

        if mask_numpy is not None:
            # mask for the first frame
            # NOTE: this should be a grayscale mask or a indexed (with/without palette) mask,
            # and definitely NOT a colored RGB image
            # https://pillow.readthedocs.io/en/stable/handbook/concepts.html: mode "L" or "P"

            # mask = Image.fromarray(mask_numpy)
            # assert mask.mode in ['L', 'P']

            # the number of objects is determined by counting the unique values in the mask
            # common mistake: if the mask is resized w/ interpolation, there might be new unique values
            objects = np.unique(mask_numpy)
            # background "0" does not count as an object
            objects = objects[objects != 0].tolist()

            mask = torch.from_numpy(mask_numpy).cuda()

            # if mask is passed in, it is memorized
            # if not all objects are specified, we propagate the unspecified objects using memory
            output_prob = self.processor.step(image, mask, objects=objects)
        else:
            # otherwise, we propagate the mask from memory
            output_prob = self.processor.step(image)

        # convert output probabilities to an object mask
        mask = self.processor.output_prob_to_mask(output_prob)

        print("=== === === CUTIE predicted a mask === === ===")

        return mask.cpu().numpy().astype(np.uint8)

    def run(self, mask_file=None):
        # Legacy method to keep compatibility if needed, but we are moving to process()
        return (cv2.imread(mask_file, -1) > 0).astype(np.uint8)
