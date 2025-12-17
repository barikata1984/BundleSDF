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


class Segmenter():
    def __init__(self):
        # obtain the Cutie model with default parameters -- skipping hydra configuration
        self.cutie = get_default_model()
        # Typically, use one InferenceCore per video
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        # the processor matches the shorter edge of the input to this size
        # you might want to experiment with different sizes, -1 keeps the original size
        self.processor.max_internal_size = 480
        
    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def process(self, image_path, mask_numpy=None):
        # load the image as RGB; normalization is done within the model
        image = Image.open(image_path)
        image = to_tensor(image).cuda().float()

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
        
        return mask.cpu().numpy().astype(np.uint8)

    def run(self, mask_file=None):
        # Legacy method to keep compatibility if needed, but we are moving to process()
        return (cv2.imread(mask_file, -1)>0).astype(np.uint8)
