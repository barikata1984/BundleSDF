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

    def get_first_frame_mask(self, image_path):
        """
        Interactively get the first frame mask using SAM3 with a text prompt.
        """
        self._load_sam3()

        # Load image for display and processing
        # Note: cv2.imread loads as BGR
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Save image to temp file instead of using cv2.imshow to avoid GUI conflicts
        # when BundleSDF GUI is running in a separate process
        # Saving to CWD so it is accessible from host via the mounted volume
        temp_image_path = os.path.abspath("input_preview.png")
        cv2.imwrite(temp_image_path, image_bgr)

        # Use external viewer to avoid GUI/thread conflicts in main process
        import subprocess

        viewer_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "utils", "view_image.py"
        )
        try:
            subprocess.run(["python3", viewer_script, temp_image_path], check=False)
        except Exception as e:
            print(f"Warning: Could not launch image viewer: {e}")

        print("\n" + "=" * 50)
        print(f"Input image saved to: {temp_image_path}")
        print(
            f"Please view this image files on your host machine to determine the prompt."
        )
        print(
            "Enter a text prompt to segment the object (e.g., 'milk carton', 'hand', 'cat')"
        )
        print("=" * 50 + "\n")

        prompt = input("Enter text prompt: ").strip()
        if not prompt:
            print("Empty prompt provided. Using default.")
            # Depending on SAM3 behavior, empty prompt might fail or do something else.
            # Assuming 'object' or similar generic? Or just return empty?
            # Let's assume user provides something. If empty, warn.

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
            # Combine all masks into a single object (label 1)
            # Assuming we are engaging with one object of interest which might have multiple parts
            combined_mask = masks.any(dim=0).cpu().numpy().astype(np.uint8)

        # --- Visualization and Escape ---
        # Create user-friendly visualization
        vis_image = image_bgr.copy()

        print(
            f"Debug: Image shape: {vis_image.shape}, Combined mask shape (raw): {combined_mask.shape}"
        )

        # Robustly handle mask shape
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

        print(f"Debug: Final Combined mask shape: {combined_mask.shape}")

        if combined_mask.max() > 0:
            # Create red overlay for the object
            red_mask = np.zeros_like(vis_image)
            red_mask[:, :, 2] = 255  # Red channel

            # Apply mask overlay
            mask_bool = combined_mask > 0
            # Blend original image and red mask
            # vis_image[mask_bool] = 0.5 * vis_image[mask_bool] + 0.5 * red_mask[mask_bool]
            vis_image[mask_bool] = cv2.addWeighted(
                vis_image[mask_bool], 0.5, red_mask[mask_bool], 0.5, 0
            ).reshape(-1, 3)
            print("Object found! Mask overlay created.")
        else:
            print("Result is empty (no object found). Saving raw image.")

        if combined_mask is not None:
            raw_mask_path = os.path.abspath("mask_result_raw.png")
            # Scale mask to 0-255 for visibility if it's boolean or 0/1
            raw_mask_vis = (combined_mask * 255).astype(np.uint8)
            cv2.imwrite(raw_mask_path, raw_mask_vis)
            print(f"Raw mask saved to: {raw_mask_path}")

        preview_path = os.path.abspath("mask_result_preview.png")
        cv2.imwrite(preview_path, vis_image)
        print(f"Mask preview saved to: {preview_path}")

        # print("Stopping execution as requested by user to inspect the mask.")
        # Returning None to signal early exit without crashing multiprocessing
        # The caller (process function) should handle this and return/exit gracefully
        # return None
        # --------------------------------

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
