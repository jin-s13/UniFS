from .box_head import ROI_BOX_HEAD_REGISTRY, build_box_head
from .roi_heads import (
    ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, build_roi_heads, select_foreground_proposals)
from .deform_roi_heads import DeformRes5ROIHeads
from .deform_fast_rcnn import DeformFastRCNNOutputLayers,DeformFastRCNNOutputs
from .poins_fast_rcnn import PointsFastRCNNOutputLayers,PointsFastRCNNOutputs
from .points_roi_heads import PointsRes5ROIHeads
from .unify_points_roi_heads import UnifyRes5ROIHeads