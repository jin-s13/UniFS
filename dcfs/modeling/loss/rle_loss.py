# Copyright (c) OpenMMLab. All rights reserved.
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models import LOSSES
from mmpose.models.utils.realnvp import RealNVP
from fvcore.nn import smooth_l1_loss

@LOSSES.register_module()
class RLELoss(nn.Module):
    """RLE Loss.

    `Human Pose Regression With Residual Log-Likelihood Estimation
    arXiv: <https://arxiv.org/abs/2107.11291>`_.

    Code is modified from `the official implementation
    <https://github.com/Jeff-sjtu/res-loglikelihood-regression>`_.

    Args:
        use_target_weight (bool): Option to use weighted MSE loss.
            Different joint types may have different target weights.
        size_average (bool): Option to average the loss by the batch_size.
        residual (bool): Option to add L1 loss and let the flow
            learn the residual error distribution.
        q_dis (string): Option for the identity Q(error) distribution,
            Options: "laplace" or "gaussian"
    """

    def __init__(self,
                 use_target_weight=False,
                 size_average=True,
                 residual=True,
                 q_dis='laplace',
                 loss_weight=1.0,
                 **kwargs
                 ):
        super(RLELoss, self).__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual
        self.q_dis = q_dis
        self.loss_weight = loss_weight

        self.flow_model = RealNVP()

    def forward(self, output, target, target_weight=None,**kwargs):
        """Forward function.

        Note:
            - batch_size: N
            - num_keypoints: K
            - dimension of keypoints: D (D=2 or D=3)

        Args:
            output (torch.Tensor[N, K, D*2]): Output regression,
                    including coords and sigmas.
            target (torch.Tensor[N, K, D]): Target regression.
            target_weight (torch.Tensor[N, K, D]):
                Weights across different joint types.
        """
        batch_size = output.shape[0]
        output = output.reshape(batch_size, -1, 4)
        target = target.reshape(batch_size, -1, 2)
        B,N=output.shape[:2]
        pred = output[:, :, :2]

        loss_box_reg = smooth_l1_loss(
            pred,
            target,
            0.0,
            reduction="sum",
        )



        sigma = output[:, :, 2:4].sigmoid()

        error = (pred - target) / (sigma + 1e-5)
        # (B, K, 2)
        log_phi = self.flow_model.log_prob(error.reshape(-1, 2))
        log_phi = log_phi.reshape(target.shape[0], target.shape[1], 1)
        log_sigma = torch.log(sigma).reshape(target.shape[0], target.shape[1],
                                             2)
        nf_loss = log_sigma - log_phi

        if self.residual:
            assert self.q_dis in ['laplace', 'gaussian', 'strict']
            if self.q_dis == 'laplace':
                loss_q = torch.log(sigma * 2) + torch.abs(error)
            else:
                loss_q = torch.log(
                    sigma * math.sqrt(2 * math.pi)) + 0.5 * error**2

            loss = nf_loss + loss_q
        else:
            loss = nf_loss

        if self.use_target_weight:
            assert target_weight is not None
            loss *= target_weight

        if self.size_average:
            loss /= B*N
        loss *= self.loss_weight
        return loss.sum()*0.1+loss_box_reg

