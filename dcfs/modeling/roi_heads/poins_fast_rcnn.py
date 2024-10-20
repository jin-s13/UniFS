"""Implement the CosineSimOutputLayers and  FastRCNNOutputLayers with FC layers."""

import torch
import logging
import numpy as np
from torch import nn
from torch.nn import functional as F
from fvcore.nn import smooth_l1_loss
from detectron2.utils.registry import Registry
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage
import copy
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding, build_transformer_layer_sequence
from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, ROI_HEADS_OUTPUT_REGISTRY
from .fast_rcnn import fast_rcnn_inference
from mmdet.models.losses.iou_loss import ciou_loss
from ..loss import RLELoss
import random
import math
from ..meta_arch.unify_point_helper import get_box_point,get_polygon_point



class PointsFastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        pred_seg_deltas,
        pred_keypoint_deltas,
        proposals,
        smooth_l1_beta,
        box_class_loss,
        fs_class,
        bbox_points_num=16,
        seg_points_num=32,
        keypoint_points_num=60,
        cls_agnostic_bbox_reg=False,
        reg_weights=1,
        det_weights=1,
        seg_weights=1,
        pose_weights=1,
        use_rle=False,
        angle_stride=2,
        proposals_gt=None,
        use_angle_loss=True,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.use_rle=use_rle
        self.use_angle_loss=use_angle_loss


        self.pred_seg_deltas = pred_seg_deltas
        self.pred_keypoint_deltas = pred_keypoint_deltas
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_class_loss_type = box_class_loss
        self.fs_class = fs_class

        self.bbox_points_num = bbox_points_num
        self.seg_points_num = seg_points_num
        self.keypoint_points_num = keypoint_points_num

        self.reg_weights = reg_weights
        self.det_weights = det_weights
        self.seg_weights = seg_weights
        self.pose_weights = pose_weights

        self.angle_stride=angle_stride

        proposals_init = proposals
        if proposals_gt is not None:
            proposals=proposals_gt

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]
        self.pre_box_points,self.pre_box=self.get_regression_boxes(self.proposals.tensor,
                                                        self.pred_proposal_deltas,
                                                        num_point=self.bbox_points_num)
        if self.pred_seg_deltas is not None:
            self.pre_seg_points,self.pre_box_seg_temp=self.get_regression_boxes(self.proposals.tensor,
                                                            self.pred_seg_deltas,
                                                            num_point=self.seg_points_num)
        if self.pred_keypoint_deltas is not None:
            self.pred_keypoint_points,_=self.get_regression_boxes(self.proposals.tensor,
                                                            self.pred_keypoint_deltas,
                                                            num_point=self.keypoint_points_num)
            self.pre_box_kpt_temp=self.proposals.tensor.unsqueeze(1)


        self.do_seg=False
        self.do_keypoint=False
        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            self.gt_boxes = get_box_point(gt_boxes.tensor,
                                            self.bbox_points_num).flatten(1)
            # self.gt_boxes_points = get_box_point(self.gt_boxes.tensor, self.bbox_points_num).flatten(1)
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals_init], dim=0)
        if self.seg_weights > 0:
            if proposals[0].has("gt_masks"):
                self.do_seg=True
                gt_masks = [p[0] for ps in proposals for p in ps.gt_masks.polygons]
                gt_masks=np.array(gt_masks)
                self.gt_masks = torch.Tensor(gt_masks).cuda()

        if self.pose_weights > 0:
            if proposals[0].has("gt_keypoints"):
                self.do_keypoint=True
                gt_keypoints = [p.gt_keypoints.tensor for p in proposals]
                gt_keypoints = torch.cat(gt_keypoints)
                self.gt_keypoints = gt_keypoints

    def get_regression_boxes(self, rois, loc_pred,num_point=None):
        if loc_pred is None:
            return None,None

        # offset = offset_pred
        # offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        # wh_ = torch.stack((w_, h_), dim=-1).unsqueeze(1)  # N_batch * 1 * 2
        # offset_point = centers + offset * wh_ * 0.1* 2
        points_dim=4 if self.use_rle else 2
        loc_pred = loc_pred.reshape(loc_pred.shape[0],-1,num_point, points_dim)[..., :2]
        x_shift = loc_pred[..., 0] * w_.unsqueeze(1).unsqueeze(1)
        y_shift = loc_pred[..., 1] * h_.unsqueeze(1).unsqueeze(1)

        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_point * 2
        shifted_offset_point = centers.unsqueeze(1) + shifts

        x_min, _ = torch.min(shifted_offset_point[..., 0], dim=-1)
        y_min, _ = torch.min(shifted_offset_point[..., 1], dim=-1)
        x_max, _ = torch.max(shifted_offset_point[..., 0], dim=-1)
        y_max, _ = torch.max(shifted_offset_point[..., 1], dim=-1)
        iou_boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1)

        return shifted_offset_point.flatten(1),iou_boxes
    def get_points_delta(self, rois, points_gt,num_point=None):
        if points_gt is None:
            return None

        # offset = offset_pred
        # offset = offset.reshape(rois.shape[0], -1, 2)
        centers_x = (rois[:, 0] + rois[:, 2]) / 2
        centers_y = (rois[:, 1] + rois[:, 3]) / 2
        w_, h_ = (rois[:, 2] - rois[:, 0] + 1), (rois[:, 3] - rois[:, 1] + 1)
        centers = torch.stack((centers_x, centers_y), dim=-1).unsqueeze(1)
        points_gt = points_gt.reshape(points_gt.shape[0],-1,num_point, 2)
        x_shift = (points_gt[..., 0] - centers[..., 0].unsqueeze(-1)) / w_.unsqueeze(1).unsqueeze(1)
        y_shift = (points_gt[..., 1] - centers[..., 1].unsqueeze(-1)) / h_.unsqueeze(1).unsqueeze(1)
        shifts = torch.stack((x_shift, y_shift), dim=-1)  # N_batch * num_point * 2
        return shifts.flatten(1)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction="mean"
        )

    def dc_loss_v1(self):
        """
        Compute loss for the decoupling classifier.
        Return scalar Tensor for single image.

        Args:
          x: predicted class scores in [-inf, +inf], x's size: N x (1+C), where N is the
           number of region proposals of one image.
          y: ground-truth classification labels in [0, C-1], y's size: N x 1, where [0,C-1]
           represent foreground classes and C-1 represents the background class.
          m: image-level label vector and its element is 0 or 1, m's size: 1 x (1+C)

        Returns:
        loss
        """
        self._log_accuracy()

        loss = 0
        bg_label = self.pred_class_logits.shape[1]-1
        num_instances = self.pred_class_logits.shape[0]
        num_classes = self.pred_class_logits.shape[1]

        for i in range(int(num_instances/512)):
            start_ind = i*512
            end_ind = 511 + i*512
            x = self.pred_class_logits[start_ind:end_ind+1,:]
            y = self.gt_classes[start_ind:end_ind+1]
            m = torch.zeros(1, num_classes).to(self.gt_classes.device)
            m[0,-1] = 1
            m[0, self.fs_class[i]] = 1
            N = x.shape[0]

            # positive head
            pos_ind = y!=bg_label
            pos_logit= x[pos_ind,:]
            pos_score = F.softmax(pos_logit, dim=1) # Eq. 4
            pos_loss = F.nll_loss(pos_score.log(), y[pos_ind], reduction="sum") #Eq. 5

            # negative head
            neg_ind = y==bg_label
            neg_logit = x[neg_ind,:]
            neg_score = F.softmax(m.expand_as(neg_logit)*neg_logit, dim=1) #Eq. 8
            neg_loss = F.nll_loss(neg_score.log(), y[neg_ind], reduction="sum")  #Eq. 9

            # total loss
            loss += (pos_loss + neg_loss)/N #Eq. 6

        return loss/(num_instances/512)

    def dc_loss(self):
        """
        Compute the decoupling classification loss for box classification.
        The implementation functions of dc_loss and dc_loss_v1 are exactly the same, but the code of the dc_loss is more concise.

        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        bg_class_ind = self.pred_class_logits.shape[1] - 1


        fg_class = self.gt_classes != bg_class_ind
        bg_class = self.gt_classes == bg_class_ind

        num_instances = self.pred_class_logits.shape[0]
        num_classes = self.pred_class_logits.shape[1]

        knonw_class_mask = torch.zeros(num_instances, num_classes).to(self.gt_classes.device)
        knonw_class_mask[fg_class,:] = 1

        for i in range(int(num_instances/512)):
            start_ind = i*512
            end_ind = 511 + i*512
            known_class_ind = copy.deepcopy(self.fs_class[i])
            known_class_ind.append(bg_class_ind)

            tmp = knonw_class_mask[start_ind:end_ind+1, known_class_ind]
            tmp[bg_class[start_ind:end_ind+1],:] = 1

            knonw_class_mask[start_ind:end_ind+1, known_class_ind] = tmp


        pred_logits = self.pred_class_logits * knonw_class_mask
        loss = F.cross_entropy(pred_logits, self.gt_classes, reduction="mean")

        return loss
    def polygons2bbox(self, polygons):
        x_min = polygons[..., 0].min(dim=-1)[0]
        y_min = polygons[..., 1].min(dim=-1)[0]
        x_max = polygons[..., 0].max(dim=-1)[0]
        y_max = polygons[..., 1].max(dim=-1)[0]
        return torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    def points_loss(self,pred_points,pred_points_delta, gt_points):

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)

        if len(fg_inds)==0:
            loss=pred_points_delta.sum()*0.0
            return loss,loss

        gt_points_final = gt_points[fg_inds]

        box_dim = gt_points_final.size(-1)

        do_keypoint = False
        if box_dim==3:
            #keypoints loss
            do_keypoint = True
            keypoint_vis = gt_points_final[..., 2]
            gt_points_final = gt_points_final[..., :2].flatten(1)
            box_dim = gt_points_final.size(-1)

        cls_agnostic_bbox_reg = self.cls_agnostic_bbox_reg
        device = self.pred_proposal_deltas.device

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )
        if len(gt_points_final)==0:
            loss=gt_points_final.sum()*0.0
            return loss,loss

        gt_points_final_delta=self.get_points_delta(self.proposals.tensor[fg_inds],gt_points_final,num_point=box_dim//2)

        if len(gt_points_final_delta)==0:
            loss=gt_points_final_delta.sum()*0.0
            return loss,loss

        if not do_keypoint:
            loss_box_reg = smooth_l1_loss(
                pred_points_delta[fg_inds[:, None], gt_class_cols],
                gt_points_final_delta,
                self.smooth_l1_beta,
                reduction="sum",
            )
            loss_iou_reg = ciou_loss(
                self.polygons2bbox(
                    pred_points[fg_inds[:, None], gt_class_cols].reshape(
                        len(gt_points_final), -1, 2)),
                self.polygons2bbox(
                    gt_points_final.reshape(len(gt_points_final), -1, 2)),
            ) * 0.2
        else:
            loss_box_reg = smooth_l1_loss(
                pred_points_delta[fg_inds].reshape(len(gt_points_final_delta),-1,2),
                gt_points_final_delta.reshape(len(gt_points_final_delta),-1,2),
                self.smooth_l1_beta,
                reduction=None,
            )
            loss_iou_reg = gt_points_final_delta.sum()*0.0
            loss_box_reg = loss_box_reg.sum(-1)*keypoint_vis
            loss_box_reg = loss_box_reg.sum()*50.0

        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg*self.reg_weights,loss_iou_reg*self.reg_weights
    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        l1_loss,iou_loss = self.points_loss(self.pre_box_points,
                                             self.pred_proposal_deltas,
                                             self.gt_boxes)

        if self.do_seg:
            seg_l1_loss, seg_iou_loss= self.points_loss(self.pre_seg_points,
                                                            self.pred_seg_deltas,
                                                         self.gt_masks)
        else:
            seg_l1_loss,seg_iou_loss = torch.tensor(0).cuda(),torch.tensor(0).cuda()
        if self.do_keypoint:
            keypoint_l1_loss,keypoint_iou_loss = self.points_loss(self.pred_keypoint_points,
                                                            self.pred_keypoint_deltas,
                                                            self.gt_keypoints)
        else:
            keypoint_l1_loss,keypoint_iou_loss = torch.tensor(0).cuda(),torch.tensor(0).cuda()


        if self.box_class_loss_type=="DC":
            loss_cls=self.dc_loss()
        else:
            loss_cls=self.softmax_cross_entropy_loss()

        return {
            "loss_cls": loss_cls,
            "loss_det": l1_loss*self.det_weights,
            "loss_seg": seg_l1_loss*self.seg_weights,
            "loss_kpt": keypoint_l1_loss*self.pose_weights,
            # "loss_point": l1_loss*self.det_weights + seg_l1_loss*self.seg_weights + keypoint_l1_loss*self.pose_weights,
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        boxes = self.pre_box
        return boxes.view(num_pred, -1).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
def calculate_angles(contour_points, stride=1):
    # assert stride in [1, 2]
    contour_points=contour_points.reshape(contour_points.shape[0],-1,2)
    M, N, _ = contour_points.shape

    # Shift points to handle stride

    pi_minus_1 = torch.roll(contour_points, shifts=stride, dims=1)
    pi_plus_1 = torch.roll(contour_points, shifts=-1*stride, dims=1)


    # Calculate vectors
    v1 = contour_points - pi_minus_1
    v2 = pi_plus_1 - contour_points

    # Calculate dot product and norms
    dot_product = torch.sum(v1 * v2, dim=-1)
    norm_product = torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1)

    # Clip the dot product to ensure it stays within the valid range [-1, 1]
    eps = 1e-6
    cos_value = dot_product / (norm_product + eps)
    cos_value = torch.clamp(cos_value, -1.0, 1.0)

    # Calculate angles：cos(theta/2)
    angles = torch.acos(cos_value)
    angles=torch.cos(angles/2.0)

    return angles

def unify_loss(pred, target, beta, reduction="mean",use_angle_loss=True,keypoint_vis=None,angle_stride=2):
    l1_loss=smooth_l1_loss(pred, target, beta, reduction=None)
    if beta>10:
        l1_loss=l1_loss*100.0
    l1_loss=l1_loss.reshape(l1_loss.shape[0],-1,2).sum(-1)
    if use_angle_loss:
        angle_loss=[]
        for stride in range(1,angle_stride+1):
            gt_angles=calculate_angles(target,stride)
            pred_angles=calculate_angles(pred,stride)
            al=smooth_l1_loss(pred_angles,gt_angles,0.5,reduction=None)
            angle_loss.append(al)
        angle_loss=torch.stack(angle_loss).mean(0)
        loss_reg=l1_loss+angle_loss*0.1
    if keypoint_vis is not None:
        if keypoint_vis.sum() > 0:
            loss_reg = loss_reg * keypoint_vis
            loss_reg = loss_reg.sum()
        else:
            loss_reg = loss_reg.sum() * 0.0
    else:
        loss_reg = loss_reg.sum()
    return loss_reg


class UnifyFastRCNNOutputs(PointsFastRCNNOutputs):
    def points_loss(self, pred_points, pred_points_delta, gt_points):

        fg_inds = torch.arange(len(gt_points))
        if len(fg_inds) == 0:
            loss = pred_points_delta.sum() * 0.0
            return loss, loss

        gt_points_final = gt_points[fg_inds]

        box_dim = gt_points_final.size(-1)

        do_keypoint = False
        if box_dim == 3:
            # keypoints loss
            do_keypoint = True
            keypoint_vis = gt_points_final[..., 2]
            gt_points_final = gt_points_final[..., :2].flatten(1)
            box_dim = gt_points_final.size(-1)

        cls_agnostic_bbox_reg = self.cls_agnostic_bbox_reg
        device = self.pred_proposal_deltas.device

        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )
        if len(gt_points_final) == 0:
            loss = gt_points_final.sum() * 0.0
            return loss, loss

        gt_points_final_delta = self.get_points_delta(
            self.proposals.tensor[fg_inds], gt_points_final,
            num_point=box_dim // 2)

        if len(gt_points_final_delta) == 0:
            loss = gt_points_final_delta.sum() * 0.0
            return loss, loss


        if not do_keypoint:
            loss_box_reg = unify_loss(
                pred_points_delta[fg_inds[:, None], gt_class_cols],
                gt_points_final_delta,
                self.smooth_l1_beta,
                reduction="sum",
                use_angle_loss=self.use_angle_loss,
                angle_stride=self.angle_stride
            )
            loss_iou_reg = ciou_loss(
                self.polygons2bbox(
                    pred_points[fg_inds[:, None], gt_class_cols].reshape(
                        len(gt_points_final), -1, 2)),
                self.polygons2bbox(
                    gt_points_final.reshape(len(gt_points_final), -1, 2)),
            ) * 0.2
        else:
            loss_box_reg = unify_loss(
                pred_points_delta[fg_inds].reshape(
                    len(gt_points_final_delta), -1, 2),
                gt_points_final_delta.reshape(len(gt_points_final_delta),
                                              -1, 2),
                self.smooth_l1_beta,
                reduction=None,
                use_angle_loss=self.use_angle_loss,
                keypoint_vis=keypoint_vis,
                angle_stride=self.angle_stride
            )
            loss_iou_reg = gt_points_final_delta.sum() * 0.0


        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg * self.reg_weights, loss_iou_reg * self.reg_weights


@ROI_HEADS_OUTPUT_REGISTRY.register()
class UnifyFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4,use_rle_loss=False,
        embed_dims=None
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(UnifyFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.mask_on      = cfg.MODEL.MASK_ON
        self.bbox_points_num = cfg.MODEL.ROI_BOX_HEAD.BBOX_POINTS_NUM
        self.seg_points_num = cfg.MODEL.ROI_BOX_HEAD.SEG_POINTS_NUM
        self.keypoint_points_num = cfg.MODEL.ROI_BOX_HEAD.KEYPOINT_POINTS_NUM
        self.max_points_num = max(self.bbox_points_num , self.seg_points_num , self.keypoint_points_num)
        pooler_resolution = 4
        self.use_rle_loss=use_rle_loss
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        num_points_reg_classes = 4 if self.use_rle_loss else 2

        embed_dims = input_size if embed_dims is None else embed_dims
        self.embed_dims = embed_dims
        self.point_pred = nn.Sequential(
                                           nn.Linear(embed_dims*self.max_points_num,input_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(input_size, num_bbox_reg_classes * self.max_points_num*num_points_reg_classes),
                                        )

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.point_pred[0].weight, std=0.001)
        nn.init.normal_(self.point_pred[2].weight, std=0.001)
        for l in [self.cls_score,self.point_pred[0],self.point_pred[2]]:
            nn.init.constant_(l.bias, 0)
        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x,query_regs):
        x_mean = x.mean(dim=[2, 3])
        # if query_regs.dim() > 2:
        #     x = torch.flatten(query_regs, start_dim=1)
        x_det=query_regs[:,:self.bbox_points_num].flatten(1)
        x_seg=query_regs[:,self.bbox_points_num:self.bbox_points_num+self.seg_points_num].flatten(1)
        x_kp=query_regs[:,self.bbox_points_num+self.seg_points_num:].flatten(1)
        # add zero padding
        x_det = F.pad(x_det, (0, (self.max_points_num-self.bbox_points_num)*self.embed_dims), "constant", 0)
        x_seg = F.pad(x_seg, (0, (self.max_points_num-self.seg_points_num)*self.embed_dims), "constant", 0)
        x_kp = F.pad(x_kp, (0, (self.max_points_num-self.keypoint_points_num)*self.embed_dims), "constant", 0)

        proposal_deltas = self.point_pred(x_det)[..., :self.bbox_points_num*2]
        seg_deltas = self.point_pred(x_seg)[..., :self.seg_points_num*2]
        keypoint_deltas = self.point_pred(x_kp)[..., :self.keypoint_points_num*2]


        if self._do_cls_dropout:
            x_mean = F.dropout(x_mean, self._dropout_ratio, training=self.training)
        scores = self.cls_score(x_mean)

        return scores, proposal_deltas,seg_deltas,keypoint_deltas

@ROI_HEADS_OUTPUT_REGISTRY.register()
class PointsFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4,use_rle_loss=False
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(PointsFastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)
        self.mask_on      = cfg.MODEL.MASK_ON
        self.bbox_points_num = cfg.MODEL.ROI_BOX_HEAD.BBOX_POINTS_NUM
        self.seg_points_num = cfg.MODEL.ROI_BOX_HEAD.SEG_POINTS_NUM
        self.keypoint_points_num = cfg.MODEL.ROI_BOX_HEAD.KEYPOINT_POINTS_NUM
        self.total_points_num = self.bbox_points_num + self.seg_points_num + self.keypoint_points_num
        pooler_resolution = 4
        self.use_rle_loss=use_rle_loss
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        num_points_reg_classes = 4 if self.use_rle_loss else 2
        self.bbox_pred = nn.Sequential(
                                           nn.Linear(input_size*pooler_resolution*pooler_resolution, input_size),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(input_size, num_bbox_reg_classes * self.total_points_num*num_points_reg_classes),
                                        )

        nn.init.normal_(self.cls_score.weight, std=0.01)
        for l in [self.cls_score]:
            nn.init.constant_(l.bias, 0)

        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x):
        x_mean = x.mean(dim=[2, 3])
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)


        if self.mask_on:
            seg_deltas = self.mask_pred(x)
        else:
            seg_deltas = None

        if self.keypoint_on:
            keypoint_deltas = self.keypoint_pred(x)
        else:
            keypoint_deltas = None

        if self._do_cls_dropout:
            x_mean = F.dropout(x_mean, self._dropout_ratio, training=self.training)
        scores = self.cls_score(x_mean)

        return scores, proposal_deltas,seg_deltas,keypoint_deltas

