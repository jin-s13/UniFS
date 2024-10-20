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
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_HEADS_OUTPUT_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from .deform_fast_rcnn import DeformFastRCNNOutputs
from .mask_head import build_mask_head
from .roi_heads import Res5ROIHeads,ROI_HEADS_REGISTRY,select_foreground_proposals
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding, build_transformer_layer_sequence
from mmdet.models.utils.transformer import inverse_sigmoid
from torch.nn import functional as F
from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    DeformConv,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)
import fvcore.nn.weight_init as weight_init
logger = logging.getLogger(__name__)

decoder = dict(
    type='DinoTransformerDecoder',
    num_layers=1,
    return_intermediate=True,
    transformerlayers=dict(
        type='DetrTransformerDecoderLayer',
        feedforward_channels=1024,
        attn_cfgs=[
            dict(
                type='MultiheadAttention',
                embed_dims=256,
                num_heads=8,
                dropout=0),  # 0.1 for DeformDETR
            dict(
                type='MultiScaleDeformableAttention',
                embed_dims=256,
                num_levels=1,
                dropout=0),  # 0.1 for DeformDETR
        ],
        ffn_cfgs=dict(
            type='FFN',
            feedforward_channels=1024,  # 1024 for DeformDETR
            num_fcs=2,
            ffn_drop=0,  # 0.1 for DeformDETR
            act_cfg=dict(type='ReLU', inplace=True)),
        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                         'ffn', 'norm')))


@ROI_HEADS_REGISTRY.register()
class DeformRes5ROIHeads(Res5ROIHeads):
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
        self.box_class_loss = cfg.MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        pooler_resolution=(1,1)
        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER

        self.reg_channel=256
        norm = cfg.MODEL.RESNETS.NORM
        # self.down_sample_res4 = Conv2d(out_channels//2, self.reg_channel,
        #                              kernel_size=1, stride=1, bias=False,norm=get_norm(norm, self.reg_channel))
        self.down_sample_res5 = Conv2d(out_channels, self.reg_channel,
                                     kernel_size=1, stride=1, bias=False,norm=get_norm(norm, self.reg_channel))
        for layer in [
            # self.down_sample_res4,
            self.down_sample_res5]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

        self.up_sample_cls = nn.Linear(self.reg_channel, out_channels)
        nn.init.normal_(self.up_sample_cls.weight, std=0.01)
        for l in [self.up_sample_cls]:
            nn.init.constant_(l.bias, 0)


        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, self.reg_channel, self.num_classes, self.cls_agnostic_bbox_reg,input_size_cls=out_channels
        )
        
        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )
        self.level_embeds = nn.Parameter(
            torch.Tensor(1, self.reg_channel))
        # self.two_stage_num_proposals = \
        #     cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE

        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.reg_channel // 2,
            temperature=20,
            normalize=True)
        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO
        self.decoder = build_transformer_layer_sequence(decoder)

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def prepare_tansformer_input(self,
                                 mlvl_feats,
                                 img_metas,
                                 ):
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            _,img_h, img_w= img_metas[img_id]['image'].shape
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_pos_embeds = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(
                    0))
            mlvl_pos_embeds.append(
                self.positional_encoding(mlvl_masks[-1]))
        attn_mask = None
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1,
                                                                    -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)

        memory = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)

        return memory, attn_mask, mask_flatten, spatial_shapes, \
            level_start_index, valid_ratios

    def _bbox_forward(self, feature_pooled,x, rois, img_metas, training=True):
        num_images = len(img_metas)
        reference_points = []
        factors = []
        for img_id in range(num_images):
            _,img_h, img_w= img_metas[img_id]['image'].shape
            factor= x[0].new_tensor(
            [img_w, img_h, img_w, img_h])
            factors.append(factor)
            index = rois[:, 0] == img_id
            reference_point = rois[index, 1:] / factor
            # x1y1x2y2->cxcywh
            reference_point[:, 2:] = reference_point[:,
                                     2:] - reference_point[:, :2]
            reference_point[:, :2] = reference_point[:,
                                     :2] + reference_point[:, 2:] / 2
            reference_points.append(reference_point.unsqueeze(0))
        reference_points = torch.cat(reference_points, dim=0)

        # import matplotlib.pyplot as plt
        # points=reference_points[0][0].cpu().numpy()
        # plt.figure()
        # plt.scatter(points[:,0],points[:,1])
        # plt.show()

        memory, attn_mask, mask_flatten, spatial_shapes, \
            level_start_index, valid_ratios = \
            self.prepare_tansformer_input(x, img_metas)

        query = feature_pooled.reshape(num_images, -1,
                                              self.reg_channel).transpose(
            0, 1)

        hs, inter_references = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=[self.box_predictor.bbox_pred],
        )

        hs = hs.permute(0, 2, 1, 3)
        outputs_coords = []
        outputs_classes = []
        for lvl in range(hs.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            tmp = self.box_predictor.bbox_pred(hs[lvl])
            # # tmp=tmp.sigmoid()
            if reference.shape[-1] == 4:
                tmp =tmp+ reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            tmp=tmp.sigmoid()
            factors_tmp=torch.cat([factors[i].unsqueeze(0).repeat(tmp.shape[1],1).unsqueeze(0) for i in range(num_images)],dim=0)
            outputs_coord = tmp * factors_tmp

            # cxcywh->x1y1x2y2
            outputs_coord[..., :2] = outputs_coord[...,
                                     :2] - outputs_coord[..., 2:] / 2
            outputs_coord[..., 2:] = outputs_coord[...,
                                     :2] + outputs_coord[..., 2:]
            # outputs_coord = outputs_coord.reshape(-1, 4)
            # outputs_coord = self.box2box_transform.get_deltas(
            #      rois[:, 1:],outputs_coord.reshape(-1, 4))

            outputs_coord = outputs_coord.reshape(-1, 4)
            if lvl == 0:
                hs_tmp = hs[lvl].clone()
                hs_tmp=self.up_sample_cls(hs_tmp)
                if self._do_cls_dropout:
                    hs_tmp = F.dropout(hs_tmp, self._dropout_ratio, training=training)
                outputs_class = self.box_predictor.cls_score1(hs_tmp)
                outputs_classes.append(outputs_class.reshape(-1,
                                                             outputs_class.shape[-1]))
            else:
                outputs_classes.append(None)
            outputs_coords.append(outputs_coord)

        bbox_results = dict(
            cls_score=outputs_classes[-1],
            bbox_pred=outputs_coords[-1],
            bbox_feats=hs[-1])
        return bbox_results

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

    def forward(self, batched_inputs, features, proposals, targets=None, fs_class=None):
        """
        See :class:`ROIHeads.forward`.
        """
        # del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets


        proposal_nums=[len(p) for p in proposals]
        min_num_proposals=min(proposal_nums)
        for i in range(len(proposals)):
            if proposal_nums[i]>min_num_proposals:
                proposals[i]=proposals[i][:min_num_proposals]
                print('Warning: proposals num is not equal!',min_num_proposals,proposal_nums[i])

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        if self._do_cls_dropout:
            feature_pooled = F.dropout(feature_pooled, self._dropout_ratio, training=self.training)
        pred_class_logits1=self.box_predictor.cls_score(feature_pooled)

        box_features = self.down_sample_res5(box_features)
        # memory=[self.down_sample_res4(features[f]) for f in self.in_features]
        memory=[]
        memory.append(self.down_sample_res5(self.res5(features['res4'])))


        feature_pooled = box_features.mean(dim=[2, 3])

        rois = torch.cat(
            [proposal_boxes[i].tensor for i in range(len(proposal_boxes))], dim=0
        )
        # range idx
        batch_idx=torch.cat([torch.full_like(proposal_boxes[i].tensor[:,0],i) for i in range(len(proposal_boxes))],dim=0)
        rois=torch.cat([batch_idx[:,None],rois],dim=1)
        # pooled to 1x1
        bbox_results = self._bbox_forward(feature_pooled,memory, rois,batched_inputs)
        pred_proposal_deltas=bbox_results['bbox_pred']
        pred_class_logits2=bbox_results['cls_score']
        pred_class_logits=[pred_class_logits1,pred_class_logits2]
        # pred_class_logits, pred_proposal_deltas = self.box_predictor(
        #     feature_pooled,memory
        # )
        del feature_pooled

        outputs = DeformFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.box_class_loss,
            fs_class
        )

        if self.training:
            del features
            losses = outputs.losses()
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
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

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


