import torch
from loguru import logger


def detect_NaN(feat_0, feat_1):
  logger.info(f'NaN detected in feature')
  logger.info(f"#NaN in feat_0: {torch.isnan(feat_0).int().sum()}, #NaN in feat_1: {torch.isnan(feat_1).int().sum()}")
  feat_0[torch.isnan(feat_0)] = 0
  feat_1[torch.isnan(feat_1)] = 0
