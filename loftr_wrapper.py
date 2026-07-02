# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,zmq,pdb,sys,time,torchvision
code_dir = os.path.dirname(os.path.realpath(__file__))
import argparse
import cv2
import torch,imageio
import torch.nn.functional as F
from copy import deepcopy
from Utils import *


def _autocast_cuda():
  # torch>=1.10 exposes torch.amp.autocast; torch.cuda.amp.autocast is deprecated in torch 2.x
  if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
    return torch.amp.autocast('cuda')
  return torch.cuda.amp.autocast(enabled=True)


class LoftrRunner:
  def __init__(self, backend:str|None=None):
    if backend is None:
      backend = os.environ.get('BUNDLESDF_MATCHER', 'eloftr')
    self.backend = backend.lower()
    # only the eloftr backbone requires input dims divisible by 32
    self.pad_multiple = 32 if self.backend == 'eloftr' else 8
    if self.backend == 'eloftr':
      self._init_eloftr()
    elif self.backend == 'loftr':
      self._init_loftr()
    else:
      raise ValueError(f"Unknown matcher backend {backend!r} (expected 'eloftr' or 'loftr')")
    logging.info(f"LoftrRunner backend={self.backend}")


  def _init_eloftr(self):
    from BundleTrack.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, reparameter
    cfg = deepcopy(full_default_cfg)
    self.matcher = LoFTR(config=cfg)
    ckpt = f'{code_dir}/BundleTrack/EfficientLoFTR/weights/eloftr_outdoor.ckpt'
    self.matcher.load_state_dict(torch.load(ckpt)['state_dict'])
    self.matcher = reparameter(self.matcher)  # RepVGG deploy conversion, required by upstream
    self.matcher = self.matcher.eval().cuda()


  def _init_loftr(self):
    from BundleTrack.LoFTR.src.loftr import LoFTR, default_cfg
    default_cfg['match_coarse']['thr'] = 0.2
    self.matcher = LoFTR(config=default_cfg)
    self.matcher.load_state_dict(torch.load(f'{code_dir}/BundleTrack/LoFTR/weights/outdoor_ds.ckpt')['state_dict'])
    self.matcher = self.matcher.eval().cuda()


  def _pad_to_multiple(self, img):
    m = self.pad_multiple
    h, w = img.shape[-2:]
    ph = (m - h % m) % m
    pw = (m - w % m) % m
    if ph or pw:
      img = F.pad(img, (0, pw, 0, ph), value=0)
    return img


  @torch.no_grad()
  def predict(self, rgbAs:np.ndarray, rgbBs:np.ndarray):
    '''
    @rgbAs: (N,H,W,C)
    '''
    h0, w0 = rgbAs.shape[1], rgbAs.shape[2]
    h1, w1 = rgbBs.shape[1], rgbBs.shape[2]
    image0 = torch.from_numpy(rgbAs).permute(0,3,1,2).float().cuda()
    image1 = torch.from_numpy(rgbBs).permute(0,3,1,2).float().cuda()
    if image0.shape[-1]==3:
      image0 = torchvision.transforms.functional.rgb_to_grayscale(image0)
      image1 = torchvision.transforms.functional.rgb_to_grayscale(image1)
    image0 = image0/255.0
    image1 = image1/255.0
    # zero-pad bottom/right so the backbone stride divides the input; origin stays fixed
    image0 = self._pad_to_multiple(image0)
    image1 = self._pad_to_multiple(image1)
    last_data = {'image0': image0, 'image1': image1}
    logging.info(f"image0: {last_data['image0'].shape}")

    batch_size = 64
    ret_keys = ['mkpts0_f','mkpts1_f','mconf','m_bids']
    with _autocast_cuda():
      i_b = 0
      for b in range(0,len(last_data['image0']),batch_size):
        tmp = {'image0': last_data['image0'][b:b+batch_size], 'image1': last_data['image1'][b:b+batch_size]}
        with torch.no_grad():
          self.matcher(tmp)
        tmp['m_bids'] += i_b
        for k in ret_keys:
          if k not in last_data:
            last_data[k] = []
          last_data[k].append(tmp[k])
        i_b += len(tmp['image0'])

    logging.info("net forward")

    for k in ret_keys:
      last_data[k] = torch.cat(last_data[k],dim=0)

    total_n_matches = len(last_data['mkpts0_f'])
    mkpts0 = last_data['mkpts0_f'].cpu().numpy()
    mkpts1 = last_data['mkpts1_f'].cpu().numpy()
    mconf = last_data['mconf'].cpu().numpy()
    pair_ids = last_data['m_bids'].cpu().numpy()
    logging.info(f"mconf, {mconf.min()} {mconf.max()}")
    logging.info(f'pair_ids {pair_ids.shape}')
    corres = np.concatenate((mkpts0.reshape(-1,2),mkpts1.reshape(-1,2),mconf.reshape(-1,1)),axis=-1).reshape(-1,5).astype(np.float32)

    # drop matches that landed in the padded region (outside the original image extent)
    if corres.shape[0]>0:
      valid = (corres[:,0]<w0)&(corres[:,1]<h0)&(corres[:,2]<w1)&(corres[:,3]<h1)
      corres = corres[valid]
      pair_ids = pair_ids[valid]

    logging.info(f'corres: {corres.shape}')
    corres_tmp = []
    for i in range(len(rgbAs)):
      cur_corres = corres[pair_ids==i]
      corres_tmp.append(cur_corres)
    corres = corres_tmp

    del last_data, image0, image1
    torch.cuda.empty_cache()

    return corres
