import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from dcfs.modeling.roi_heads import build_roi_heads
import random
from .unify_point_helper import get_box_point,get_polygon_point
__all__ = ["PointsFasterRCNN"]
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.layers import paste_masks_in_image
from detectron2.structures import Instances
from detectron2.utils.memory import retry_if_cuda_oom
from pycocotools import mask as mask_utils
import cv2
import numpy as np


from shapely.geometry import MultiPoint
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
def poly_to_mask(polygons, height, width):
    masks=[]
    for polygon in polygons:
        rles = mask_utils.frPyObjects([polygon], height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle).astype(np.bool)
        mask = torch.from_numpy(mask)
        masks.append(mask)
    mask=torch.stack(masks, dim=0)
    return mask
# perhaps should rename to "resize_instance"
def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """

    # Converts integer tensors to float temporaries
    #   to ensure true division is performed when
    #   computing scale_x and scale_y.
    if isinstance(output_width, torch.Tensor):
        output_width_tmp = output_width.float()
    else:
        output_width_tmp = output_width

    if isinstance(output_height, torch.Tensor):
        output_height_tmp = output_height.float()
    else:
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances((output_height, output_width), **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes

    output_boxes.scale(scale_x, scale_y)
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]
    if results.has("point_det"):
        results.point_det[:,:, 0] *= scale_x
        results.point_det[:,:, 1] *= scale_y
        vis_ones = torch.ones((len(results), results.point_det.shape[1], 1), dtype=torch.float32, device=results.point_det.device)
        results.point_det = torch.cat([results.point_det, vis_ones], dim=-1)
    else:
        results.point_det=torch.zeros((len(results),32,3),dtype=torch.float32,device=results.scores.device)

    if results.has("pred_masks"):
        polygons=results.pred_masks
        if len(polygons)>0:
            polygons=polygons.reshape(len(polygons),-1,2)
            polygons[:, :, 0] *= scale_x
            polygons[:, :, 1] *= scale_y
            results.point_seg=polygons

            vis_ploygons=torch.ones((len(polygons),polygons.shape[1],1),dtype=torch.float32,device=polygons.device)
            results.point_seg=torch.cat([polygons,vis_ploygons],dim=-1)
            # results.pred_keypoints=torch.cat([polygons,vis_ploygons],dim=-1)

            polygons=polygons.reshape(len(polygons),-1).detach().cpu().numpy()
            polygons=np.array(polygons,dtype=np.float64)
            mask=poly_to_mask(polygons, output_height, output_width)
            # to device
            mask=mask.to(results.pred_masks.device)
            results.pred_masks=mask
        else:
            results.pred_masks=torch.zeros((len(results),output_height,output_width),dtype=torch.bool,device=results.scores.device)
            results.point_seg=torch.zeros((len(results),32,3),dtype=torch.float32,device=results.scores.device)


    if results.has("pred_keypoints"):
        if len(results.pred_keypoints)==0:
            results.pred_keypoints=torch.zeros((len(results),52,3),dtype=torch.float32,device=results.pred_keypoints.device)
        else:
            if len(results.pred_keypoints.shape)==2:
                results.pred_keypoints=results.pred_keypoints.reshape(len(results.pred_keypoints),-1,2)
                vis_ones=torch.ones((len(results.pred_keypoints),results.pred_keypoints.shape[1],1),dtype=torch.float32,device=results.pred_keypoints.device)
                results.pred_keypoints=torch.cat([results.pred_keypoints,vis_ones],dim=-1)
            results.pred_keypoints[:, :, 0] *= scale_x
            results.pred_keypoints[:, :, 1] *= scale_y

    return results

@META_ARCH_REGISTRY.register()
class UnifyPointsFasterRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)
        self.seg_points_num = cfg.MODEL.ROI_BOX_HEAD.SEG_POINTS_NUM
        self.bbox_points_num = cfg.MODEL.ROI_BOX_HEAD.BBOX_POINTS_NUM
        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

        if cfg.MODEL.ROI_BOX_HEAD.FREEZE_REG:
            for p in self.roi_heads.box_predictor.point_pred.parameters():
                p.requires_grad = False
            for p in self.roi_heads.decoder.parameters():
                p.requires_grad = False
            for p in self.roi_heads.downsample1.parameters():
                p.requires_grad = False
            for p in self.roi_heads.downsample2.parameters():
                p.requires_grad = False
            print("froze roi_box_head bbox_pred parameters")



        if cfg.MODEL.ROI_MASK_HEAD.FREEZE_WITHOUT_PREDICTOR \
                and cfg.MODEL.ROI_MASK_HEAD.FREEZE:
            # Both frozen doesn't make sense and likely indicates that we forgot to
            # modify a config, so better to early error.
            assert False

        if cfg.MODEL.ROI_MASK_HEAD.FREEZE_WITHOUT_PREDICTOR:
            frozen_names = []
            for n, p in self.roi_heads.mask_head.named_parameters():
                if 'predictor' not in n:
                    p.requires_grad = False
                    frozen_names.append(n)
        self.index=0
    def polygons2bbox(self, polygons):
        x_min = polygons[..., 0].min(dim=-1)[0]
        y_min = polygons[..., 1].min(dim=-1)[0]
        x_max = polygons[..., 0].max(dim=-1)[0]
        y_max = polygons[..., 1].max(dim=-1)[0]
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    def prepare_data(self, batched_inputs):
        for batched_input in batched_inputs:
            # plt.imshow(batched_input["image"].cpu().numpy().transpose(1,2,0))
            # save_flag=0
            if "instances" not in batched_input:
                continue
            instances = batched_input["instances"]
            if not instances.has("gt_masks"):
                continue
            gt_masks = instances.gt_masks.polygons
            mask_points=[]
            bboxes_new = []
            labels_new=[]
            for i in range(len(gt_masks)):
                    mask_i=gt_masks[i]
                    mask=[]
                    #merge list mask
                    if len(mask_i)>1:
                        for j in range(len(mask_i)):
                            mask.extend(mask_i[j])
                        mask=np.array(mask)
                    else:
                        mask=mask_i[0]
                    mask_point=get_polygon_point(mask.reshape(1,-1,2),num_point=self.seg_points_num)
                    bbox_new=self.polygons2bbox(mask_point)
                    mask_points.append([mask_point.reshape(-1).cpu().numpy()])
                    bboxes_new.append(bbox_new)
                    labels_new.append(instances.gt_classes[i])
            if len(mask_points)>0:
                instances.gt_boxes.tensor=torch.cat(bboxes_new,dim=0)
                instances.gt_masks.polygons=mask_points
                instances.gt_classes=torch.tensor(labels_new)

        for batched_input in batched_inputs:
            if "instances" not in batched_input:
                continue
            instances = batched_input["instances"]
            gt_boxes = instances.gt_boxes.tensor
            if gt_boxes.shape[0]>0:
                gt_boxes_points = get_box_point(gt_boxes,self.bbox_points_num).flatten(1)
                instances.gt_boxes_points = gt_boxes_points
        return batched_inputs
    def forward(self, batched_inputs,support_features=None,get_features=False):
        if get_features:
            batched_inputs = self.prepare_data(batched_inputs)
            gt_instances = [x["instances"].to(self.device) for x in
                            batched_inputs]
            fs_class = [x["fs_class"] for x in batched_inputs]
            return self.get_support_features(batched_inputs, gt_instances, fs_class)
        if not self.training:
            return self.inference(batched_inputs,support_features)
        assert "instances" in batched_inputs[0]
        assert "fs_class" in batched_inputs[0]

        batched_inputs = self.prepare_data(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        fs_class = [x["fs_class"] for x in batched_inputs]
        proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances, fs_class)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs,support_features=None):
        assert not self.training
        _, _, results, image_sizes = self._forward_once_(batched_inputs, None,support_features=support_features)
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def get_support_features(self,batched_inputs, gt_instances=None, fs_class=None):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}

        support_features=self.roi_heads(images,
                       features_de_rcnn,
                       proposals,
                       gt_instances,
                       fs_class,
                       get_features=True
                       )
        return support_features
    def _forward_once_(self, batched_inputs, gt_instances=None, fs_class=None,support_features=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances)

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images,
                                                  features_de_rcnn,
                                                  proposals,
                                                  gt_instances,
                                                  fs_class,
                                                  support_features)

        return proposal_losses, detector_losses, results, images.image_sizes

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
