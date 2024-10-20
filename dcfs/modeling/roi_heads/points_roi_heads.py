import torch
import logging
import numpy as np
from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou,PolygonMasks,Keypoints
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from .poins_fast_rcnn import PointsFastRCNNOutputs,UnifyFastRCNNOutputs
from .mask_head import build_mask_head
from .roi_heads import Res5ROIHeads,ROI_HEADS_REGISTRY,select_foreground_proposals

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from detectron2.layers import (
    Conv2d,
    get_norm,
)
import fvcore.nn.weight_init as weight_init
from mmdet.models.utils import LearnedPositionalEncoding

logger = logging.getLogger(__name__)


@ROI_HEADS_REGISTRY.register()
class PointsRes5ROIHeads(Res5ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON
        self.box_class_loss = cfg.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE
        self.bbox_points_num = cfg.MODEL.ROI_BOX_HEAD.BBOX_POINTS_NUM
        self.seg_points_num = cfg.MODEL.ROI_BOX_HEAD.SEG_POINTS_NUM
        self.keypoint_points_num = cfg.MODEL.ROI_BOX_HEAD.KEYPOINT_POINTS_NUM
        self.reg_weights=cfg.MODEL.ROI_BOX_HEAD.REG_WEIGHTS
        if cfg.MODEL.ROI_BOX_HEAD.FREEZE_REG:
            self.reg_weights=0
        self.use_rle_loss=cfg.MODEL.ROI_BOX_HEAD.USE_RLE_LOSS
        # fmt: on
        # assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.out_channels = out_channels
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg,use_rle_loss=self.use_rle_loss
        )
        # if self.mask_on:
        #     del self.mask_head


    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x)
        # print('res5:', x.size())
        return x
    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes
                if self.mask_on:
                    gt_masks = PolygonMasks(
                            [[np.zeros(self.seg_points_num*2)]] * len(sampled_idxs)
                    )
                    proposals_per_image.gt_masks = gt_masks
                if self.keypoint_on:
                    gt_keypoints = Keypoints(
                        targets_per_image.gt_boxes.tensor.new_zeros(
                            (len(sampled_idxs), self.keypoint_points_num,3)
                        )
                    )
                    proposals_per_image.gt_keypoints = gt_keypoints

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt
    def forward(self, images, features, proposals, targets=None, fs_class=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas,pred_seg_deltas,pred_keypoint_deltas = self.box_predictor(
            box_features
        )

        # del feature_pooled

        outputs = PointsFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            pred_seg_deltas,
            pred_keypoint_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_class_loss,
            fs_class,
            bbox_points_num=self.bbox_points_num,
            seg_points_num=self.seg_points_num,
            keypoint_points_num=self.keypoint_points_num,
            cls_agnostic_bbox_reg=self.cls_agnostic_bbox_reg,
            reg_weights=self.reg_weights,
            use_rle=self.use_rle_loss,
        )

        if self.training:
            del features
            losses = outputs.losses()
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            pred_instances = self.forward_with_given_boxes(features,
                                                           pred_instances,
                                                           outputs)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances,outputs):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.


        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        assert len(instances)==1 # only support batch size 1

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            # x_pooled = x.mean(dim=[2, 3])
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            pre_seg_points_delta=self.box_predictor.mask_pred(x)
            if self.keypoint_on:
                pre_keypoint_points_delta=self.box_predictor.keypoint_pred(x)
            rois = torch.cat([x.pred_boxes.tensor for x in instances])
            if len(rois) == 0:
                instances[0].pred_masks = instances[0].scores #tensor([])
                return instances
            pre_seg_points,_=outputs.get_regression_boxes(rois, pre_seg_points_delta,num_point=self.seg_points_num)
            if self.keypoint_on:
                pre_keypoint_points,_=outputs.get_regression_boxes(rois, pre_keypoint_points_delta,num_point=self.keypoint_points_num)
                if not self.cls_agnostic_bbox_reg:
                    pre_keypoint_points=pre_keypoint_points.reshape(-1, self.num_classes, self.keypoint_points_num * 2)
                    pre_keypoint_points=pre_keypoint_points[torch.arange(len(pre_keypoint_points)), instances[0].pred_classes]
                instances[0].pred_keypoints=pre_keypoint_points
            # pre_seg_points: [num,64]; transfer polygen to mask
            if not self.cls_agnostic_bbox_reg:
                pre_seg_points = pre_seg_points.reshape(-1, self.num_classes, self.seg_points_num * 2)
                pre_seg_points = pre_seg_points[torch.arange(len(pre_seg_points)), instances[0].pred_classes]
            instances[0].pred_masks=pre_seg_points
            return instances
            # return self.mask_head(x, instances)
        else:
            return instances



