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
from .points_roi_heads import PointsRes5ROIHeads

logger = logging.getLogger(__name__)

@ROI_HEADS_REGISTRY.register()
class UnifyRes5ROIHeads(PointsRes5ROIHeads):
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        decoder_num=cfg.MODEL.ROI_BOX_HEAD.DECODER_NUM
        self.bilinear_detach=True
        self.embed_dims=256
        decoder=dict(
            type='TransformerLayerSequence',
            num_layers=decoder_num,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=
                    dict(
                        type='MultiheadAttention',
                        embed_dims=self.embed_dims,
                        num_heads=8,
                        dropout=0.1)
                ,
                feedforward_channels=1024,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))
        self.decoder = build_transformer_layer_sequence(decoder)
        self.downsample1 = Conv2d(
                self.out_channels//2,
                self.embed_dims,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm('BN', self.embed_dims),
            )
        self.downsample2 = Conv2d(
                self.out_channels,
                self.embed_dims,
                kernel_size=1,
                stride=1,
                bias=False,
                norm=get_norm('BN', self.embed_dims),
            )
        self.box_predictor = ROI_HEADS_OUTPUT_REGISTRY.get(output_layer)(
            cfg, self.out_channels, self.num_classes, self.cls_agnostic_bbox_reg,
            use_rle_loss=self.use_rle_loss,
            embed_dims=self.embed_dims
        )
        self.img_pos_encoder = LearnedPositionalEncoding(self.embed_dims // 2,100,100)
        self.foreground_embedding = nn.Embedding(1, self.embed_dims)
        for layer in [self.downsample1,self.downsample2]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

        self.det_weights = cfg.MODEL.ROI_BOX_HEAD.DET_WEIGHTS
        self.seg_weights = cfg.MODEL.ROI_BOX_HEAD.SEG_WEIGHTS
        self.pose_weights = cfg.MODEL.ROI_BOX_HEAD.POSE_WEIGHTS
        self.refined = cfg.MODEL.ROI_BOX_HEAD.REFINED
        self.angle_stride = cfg.MODEL.ROI_BOX_HEAD.ANGLE_STRIDE
        self.use_angle_loss = cfg.MODEL.ROI_BOX_HEAD.USE_ANGLE_LOSS
    def bilinear_gather(self, feat_map, index):
        '''
        input:
            index: FloatTensor, N*2 (w coordinate, h coordinate)
            feat_map: C*H*W
        return:
            bilinear inperpolated features: C*N
        '''
        assert feat_map.ndim == 3
        height, width = feat_map.shape[1:]
        w, h = index[..., 0], index[..., 1]

        h_low = torch.floor(h)
        h_low = torch.clamp(h_low, min=0, max=height - 1)
        h_high = torch.where(h_low >= height - 1, h_low, h_low + 1)
        h = torch.where(h_low >= height - 1, h_low, h)

        w_low = torch.floor(w)
        w_low = torch.clamp(w_low, min=0, max=width - 1)
        w_high = torch.where(w_low >= width - 1, w_low, w_low + 1)
        w = torch.where(w_low >= width - 1, w_low, w)

        h_low = h_low.long()
        w_low = w_low.long()
        h_high = h_high.long()
        w_high = w_high.long()

        if self.bilinear_detach:
            h_low = h_low.detach()
            w_low = w_low.detach()
            h_high = h_high.detach()
            w_high = w_high.detach()

        lh = h - h_low  # N
        lw = w - w_low
        hh = 1 - lh
        hw = 1 - lw

        v1 = feat_map[:, h_low, w_low]  # C * N
        v2 = feat_map[:, h_low, w_high]
        v3 = feat_map[:, h_high, w_low]
        v4 = feat_map[:, h_high, w_high]

        w1 = hh * hw  # N
        w2 = hh * lw
        w3 = lh * hw
        w4 = lh * lw
        w1, w2, w3, w4 = [x.unsqueeze(0) for x in [w1, w2, w3, w4]]

        val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4  # C*N
        return val
    def extract_support_feats(self,
                              feats,
                              gt_points,
                              featmap_strides,
                              has_vis=False):
        support_feats = []
        for j in range(len(gt_points)):
            if  not has_vis:
                points = gt_points[j].reshape(-1, 2).clone()
                feat = self.bilinear_gather(feats,
                                            points / featmap_strides)
            else:
                points = gt_points[j][..., :2].reshape(-1, 2).clone()
                points_vis = gt_points[j][..., -1].reshape(-1).clone()
                points_vis=points_vis>0
                feat = self.bilinear_gather(feats,
                                            points / featmap_strides)
                feat = feat * points_vis[None, :]+self.foreground_embedding.weight.repeat(len(points),1)\
                           .transpose(0,1)*(~points_vis[None, :])
            support_feats.append(feat)
        support_feats = torch.stack(support_feats).mean(dim=0)
        return support_feats
    def prepare_single_support_feats(self,features,target,featmap_strides):
        support_bbox_feats=self.foreground_embedding.weight.repeat(self.bbox_points_num,1).transpose(0,1)
        support_seg_feats=self.foreground_embedding.weight.repeat(self.seg_points_num,1).transpose(0,1)
        support_keypoint_feats=self.foreground_embedding.weight.repeat(self.keypoint_points_num,1).transpose(0,1)

        if target.has("gt_boxes_points"):
            support_bbox_feats = self.extract_support_feats(
                features, target.gt_boxes_points, featmap_strides)
        if target.has("gt_masks"):
            gt_masks_points = target.gt_masks.polygons
            if len(gt_masks_points) > 0:
                gt_masks_points=torch.stack([torch.Tensor(x[0]).to(features.device) for x in gt_masks_points])
                support_seg_feats = self.extract_support_feats(
                    features, gt_masks_points, featmap_strides)
        if target.has("gt_keypoints"):
            gt_keypoints_points = target.gt_keypoints.tensor
            if len(gt_keypoints_points) > 0:
                support_keypoint_feats = self.extract_support_feats(
                    features, gt_keypoints_points, featmap_strides, has_vis=True)
        return support_bbox_feats,support_seg_feats,support_keypoint_feats
    def prepare_support_feats(self,features,targets,featmap_strides):
        reg_feats = self.downsample1(features['res4'])
        reg_feats = reg_feats + self.img_pos_encoder(reg_feats)
        batch_num=len(targets)
        support_feats = []
        for i in range(batch_num):
            support_bbox_feats, support_seg_feats, support_keypoint_feats = self.prepare_single_support_feats(
                reg_feats[i], targets[i], featmap_strides)
            support_feats.append(torch.cat([support_bbox_feats, support_seg_feats, support_keypoint_feats],dim=-1))
        return support_feats
    def get_query_regs(self,query_feats,support_feats,pred_class=None):
        num_object=query_feats.size(0)
        if pred_class is None or isinstance(support_feats,torch.Tensor):
            query=support_feats.unsqueeze(0).expand(num_object,-1,-1).permute(2,0,1)
        else:
            query=torch.stack([support_feats[k.item()] for k in pred_class]).permute(2,0,1)
        query=query.split([self.bbox_points_num,self.seg_points_num,self.keypoint_points_num],dim=0)
        query_regs=[]
        key = query_feats.flatten(2).permute(2, 0, 1)
        for i in range(len(query)):
            # emb=torch.cat([query[i]+key.mean(0).unsqueeze(0),key],dim=0)
            emb=query[i]+key.mean(0).unsqueeze(0)
            query_reg = self.decoder(
                emb,
                key,
                key,
            ).permute(1,0,2)
            query_regs.append(query_reg)
        query_regs = torch.cat(query_regs, dim=1)
        return query_regs
    def get_model_init_support_features(self,features,targets):
        support_features = {}
        gt_classes = targets[0].gt_classes
        gt_classes = torch.unique(gt_classes)
        targets = targets[0]
        for cate in gt_classes:
            selected_inds = \
                torch.where(targets.gt_classes == cate)[0]
            targets_i = targets[selected_inds]
            support_feat = self.prepare_support_feats(features,
                                                      [targets_i],
                                                      self.feature_strides[
                                                          self.in_features[0]])
            if cate.item() not in support_features:
                support_features[cate.item()] = []
            support_features[cate.item()].append(support_feat[0])
        return support_features
    def change_pose_vis(self,targets,selected_class):
        vis = []
        for i in range(len(targets)):
            if targets[i].has("gt_keypoints"):
                selected_inds = \
                torch.where(targets[i].gt_classes == selected_class)[0]
                targets_i = targets[i][selected_inds]
                gt_keypoints = targets_i.gt_keypoints.tensor
                vis.append((gt_keypoints[..., 2] > 0).sum(0)>0)
        vis=vis[0] & vis[1]
        vis=vis.int().unsqueeze(0)
        for i in range(len(targets)):
            if targets[i].has("gt_keypoints"):
                selected_inds = \
                torch.where(targets[i].gt_classes == selected_class)[0]
                targets[i][selected_inds].gt_keypoints.tensor[..., 2] = (
                        vis * (targets[i][selected_inds].gt_keypoints.tensor[
                                     ..., 2] > 0)).int()
        return targets



    def forward_train(self, images, features, proposals,
                      targets=None, fs_class=None,get_features=False):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        if get_features:
            support_features = self.get_model_init_support_features(features, targets)
            return support_features

        support_feats = self.prepare_support_feats(features,
                                                   targets,
                                                   self.feature_strides[self.in_features[0]],
                                                   )


        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # del targets
        num_batch = len(proposals)

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
        fg_inds = torch.nonzero(
            (gt_classes >= 0) & (gt_classes <  self.num_classes)
        ).squeeze(1)
        if len(fg_inds) == 0:
                del features
                loss = torch.tensor(0.0, device=box_features.device)
                losses = {
                            "loss_cls": loss,
                            # "loss_point": loss,
                            'loss_det':loss,
                            'loss_seg':loss,
                            'loss_kpt':loss
                        }
                return [], losses


        query_feats = self.downsample2(box_features[fg_inds])
        query_feats=self.img_pos_encoder(query_feats)+query_feats
        p_num=[len(p) for p in proposals]
        batch_index = [torch.ones(p, device=gt_classes.device)*i for i,p in enumerate(p_num)]
        batch_index = torch.cat(batch_index, dim=0)[fg_inds]

        query_feats=[query_feats[batch_index==i] for i in range(num_batch)]

        query_regs=[]
        for i in range(num_batch):
            num_object=query_feats[i].size(0)
            if num_object==0:
                continue
            index=(i+1)%num_batch
            query_reg = self.get_query_regs(query_feats[i],support_feats[index])
            query_regs.append(query_reg)
        try:
            query_regs = torch.cat(query_regs, dim=0)
        except:
            del features
            loss = torch.tensor(0.0, device=box_features.device)
            losses = {
                "loss_cls": loss,
                # "loss_point": loss,
                'loss_det':loss,
                'loss_seg':loss,
                'loss_kpt':loss
            }
            return [], losses

        # feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas,pred_seg_deltas,pred_keypoint_deltas = \
            self.box_predictor(
            box_features,
            query_regs
        )

        # del feature_pooled
        p_num_sum=[len(proposals[0])]
        for i in range(1,num_batch):
            p_num_sum.append(sum(p_num[:i]))
        proposals_gt=[p[fg_inds[batch_index==i]%p_num_sum[i]] for i,p in enumerate(proposals)]
        outputs = UnifyFastRCNNOutputs(
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
            det_weights=self.det_weights,
            seg_weights=self.seg_weights,
            pose_weights=self.pose_weights,
            angle_stride=self.angle_stride,
            use_rle=self.use_rle_loss,
            proposals_gt=proposals_gt
        )


        del features
        losses = outputs.losses()
        return [], losses

    def simple_test(self, images, features, proposals,fs_class=None, support_feats=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        pred_class_logits=self.box_predictor.cls_score(box_features.mean(dim=[2, 3]))
        pred_class= pred_class_logits[:,:-1].argmax(dim=1)


        query_feats = self.downsample2(box_features)
        query_feats = self.img_pos_encoder(query_feats) + query_feats

        query_regs = self.get_query_regs(query_feats, support_feats,pred_class)


        # feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        pred_class_logits, pred_proposal_deltas,pred_seg_deltas,pred_keypoint_deltas = \
            self.box_predictor(
            box_features,
            query_regs
        )

        # del feature_pooled
        outputs = UnifyFastRCNNOutputs(
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
            use_angle_loss=self.use_angle_loss,
        )

        pred_instances, filter_inds = outputs.inference(
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_detections_per_img,
        )
        pred_instances[0].scores_emb=pred_class_logits[filter_inds[0]]
        if len(filter_inds[0])>0:
            pred_instances[0].point_det=outputs.pre_box_points[filter_inds[0]].reshape(len(filter_inds[0]),-1,2)
        # else:
        #     pred_instances[0].point_det = pred_instances[0].scores
        if self.det_weights>0 and self.refined:
            pred_instances = self.forward_with_given_boxes(features,
                                                               pred_instances,
                                                               outputs,
                                                           support_feats,
                                                           )
        else:
            rois=proposals[0].proposal_boxes.tensor[filter_inds[0]]
            # rois = torch.cat([x.pred_boxes.tensor for x in pred_instances])
            if len(rois) == 0:
                pred_instances[0].pred_masks = pred_instances[0].scores  # tensor([])
                pred_instances[0].pred_keypoints = pred_instances[0].scores
                return pred_instances, {}
            pre_seg_points_delta=outputs.pred_seg_deltas[filter_inds[0]]
            pre_keypoint_points_delta=outputs.pred_keypoint_deltas[filter_inds[0]]
            pre_seg_points, _ = outputs.get_regression_boxes(rois,
                                                             pre_seg_points_delta,
                                                             num_point=self.seg_points_num)
            pre_keypoint_points, _ = outputs.get_regression_boxes(rois,
                                                                  pre_keypoint_points_delta,
                                                                  num_point=self.keypoint_points_num)
            pred_instances[0].pred_keypoints = pre_keypoint_points
            pred_instances[0].pred_masks = pre_seg_points

        return pred_instances, {}
    def forward(self, images, features, proposals,
                targets=None, fs_class=None,support_feats=None,get_features=False):
        if get_features:
            return self.forward_train(images, features, proposals,
                                      targets, fs_class,get_features)
        if self.training:
            return self.forward_train(images, features, proposals,
                                      targets, fs_class,)
        else:
            return self.simple_test(images, features, proposals,
                                     support_feats=support_feats)

    def forward_with_given_boxes(self, features, instances,outputs,support_feats):
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

        rois = torch.cat([x.pred_boxes.tensor for x in instances])
        if len(rois) == 0:
            instances[0].pred_masks = instances[0].scores #tensor([])
            instances[0].pred_keypoints = instances[0].scores
            return instances

        features = [features[f] for f in self.in_features]
        box_features = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
        query_feats = self.downsample2(box_features)
        query_feats = self.img_pos_encoder(query_feats) + query_feats

        x = self.get_query_regs(query_feats, support_feats,instances[0].pred_classes)

        _, _,pre_seg_points_delta,pre_keypoint_points_delta = \
            self.box_predictor(
            box_features,
            x
        )

        pre_seg_points,_=outputs.get_regression_boxes(rois, pre_seg_points_delta,num_point=self.seg_points_num)
        pre_keypoint_points,_=outputs.get_regression_boxes(rois, pre_keypoint_points_delta,num_point=self.keypoint_points_num)
        instances[0].pred_keypoints=pre_keypoint_points
        instances[0].pred_masks=pre_seg_points
        return instances



