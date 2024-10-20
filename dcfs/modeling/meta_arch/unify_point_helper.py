import torch

from shapely.geometry import Polygon,Point
import numpy as np
import torch.nn as nn
import functools
import matplotlib.pyplot as plt
from typing import Any, Optional, Tuple, Type
import math
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C


def uniformsample(pgtnp_px2, newpnum):
    # borrowed from https://github.com/zju3dv/snake
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2

    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            # pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp
def get_box_point( bboxes, num_point, order_index=None):
    if order_index is None:
        order_index = torch.arange(num_point)
    xl, yl, xr, yr = bboxes.split((1, 1, 1, 1), dim=-1)
    points = torch.cat([xl, yl, xr, yl, xr, yr, xl, yr], -1)
    points = get_polygon_point(points, num_point)
    points = points.type_as(bboxes)
    points[:,:num_point, :] = points[:,order_index, :]
    return points

def sort_polygon( polygon):
    polygon = polygon.reshape(-1, 2)
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    center = [xs.mean(), ys.mean()]
    ref_vec = [-1, 0]
    sort_func = functools.partial(clockwiseangle_and_distance, origin=center, ref_vec=ref_vec)
    sorted_polygon = sorted(polygon.tolist(), key=sort_func)
    return np.array(sorted_polygon)

def clockwiseangle_and_distance(point, origin=[0, 0], ref_vec=[1, 0]):
    import math
    vector = [point[0] - origin[0], point[1] - origin[1]]
    lenvector = math.hypot(vector[0], vector[1])
    if lenvector == 0:
        return -math.pi, 0
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * ref_vec[0] + normalized[1] * ref_vec[1]  # x1*x2 + y1*y2
    diffprod = ref_vec[1] * normalized[0] - ref_vec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    return angle, lenvector

def get_cw_poly(poly):
    return poly[::-1] if Polygon(poly).exterior.is_ccw else poly

def unify_origin_polygon( poly):
    new_poly = np.zeros_like(poly)
    xmin = poly[:, 0].min()
    xmax = poly[:, 0].max()
    ymin = poly[:, 1].min()
    ymax = poly[:, 1].max()
    tcx = (xmin + xmax) / 2
    tcy = ymin
    dist = (poly[:, 0] - tcx) ** 2 + (poly[:, 1] - tcy) ** 2
    min_dist_idx = dist.argmin()
    new_poly[:(poly.shape[0] - min_dist_idx)] = poly[min_dist_idx:]
    new_poly[(poly.shape[0] - min_dist_idx):] = poly[:min_dist_idx]
    return new_poly

def get_polygon_point( points_total, num_point,spline_num=10):
    spline_num=10 if num_point>4 else 100
    spline_poly_num=num_point*spline_num
    num=len(points_total)
    if isinstance(points_total,torch.Tensor):
        points_total = points_total.reshape(num,-1, 2)
        points_total = points_total.cpu().numpy()
    else:
        points_total = np.array(points_total).reshape(num,-1, 2)
    output=[]
    for p_id in range(num):
        points=points_total[p_id]
        if num_point==2:
            points=[points.min(0),points.max(0)]
        else:
            points = uniformsample(points, spline_poly_num)
            if points.shape[0]==0:
                points=np.zeros((spline_poly_num,2))
            tt_idx = np.argmin(np.power(points - points[0], 2).sum(axis=1))
            valid_polygon = np.roll(points, -tt_idx, axis=0)[::spline_num]
            cw_valid_polygon = get_cw_poly(valid_polygon)

            points_clock = unify_origin_polygon(cw_valid_polygon).reshape(-1)
            points_clock = sort_polygon(points_clock)[::-1]
            start_point = points_clock[0]
            tt_idx = np.argmin(np.power(cw_valid_polygon - start_point, 2).sum(axis=1))
            points = np.roll(cw_valid_polygon, -tt_idx, axis=0)



        points = np.array(points)
        if points.shape[0]!=num_point:
            if points.shape[0]==0:
                points=np.zeros((num_point,2))
            else:
                points=points[:num_point] if points.shape[0]>num_point else np.concatenate([points,np.tile(points[-1],(num_point-points.shape[0],1))],0)
        points = torch.from_numpy(points).unsqueeze(0)
        output.append(points)

    output=torch.cat(output).cuda().float()

    return output

def calculate_angle(p1, p2, p3):
    """Calculate the angle between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)

    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)

    # Clip the dot product to ensure it stays within the valid range [-1, 1]
    cos_value=dot_product / norm_product
    eps = 1e-6
    if 1.0 < cos_value < 1.0 + eps:
        cos_value = 1.0
    elif -1.0 - eps < cos_value < -1.0:
        cos_value = -1.0

    angle = np.arccos(cos_value)
    return np.degrees(angle)
def contour_discretization(contour_points, N,spline_num):
    """Discretize the contour and extract N important points."""
    M = len(contour_points)
    sampled_points = []
    sampled_points_angles = []
    for i in range(M):
        pi_minus_1 = contour_points[i - 1] if i > 0 else contour_points[M - 1]
        pi = contour_points[i]
        pi_plus_1 = contour_points[(i + 1) % M]

        angle_i = calculate_angle(pi_minus_1, pi, pi_plus_1)

        # if angle_i > 0.01:  # Discard almost linear points
        sampled_points.append((pi, angle_i))
        sampled_points_angles.append(angle_i)
    sampled_points_angles = np.array(sampled_points_angles)

    index_sorted = np.argsort(sampled_points_angles)
    # delete sampled_points_angles>179 or sampled_points_angles<1
    index_sorted1 = index_sorted[sampled_points_angles[index_sorted] < 179]
    if len(index_sorted1) >3:
        index_sorted = index_sorted1

    return index_sorted[:N]


