import os
import io
import json
import copy
import torch
import logging
import itertools
import contextlib
import numpy as np
from tabulate import tabulate
from collections import OrderedDict
import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from detectron2.structures import BoxMode
from detectron2.utils import comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import create_small_table
from detectron2.data.datasets.coco import convert_to_coco_json
from dcfs.evaluation.evaluator import DatasetEvaluator


class COCOEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, tasks, distributed, output_dir=None):
        self._tasks = tasks
        self._distributed = distributed
        self._output_dir = output_dir
        self._dataset_name = dataset_name
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'")
            cache_path = convert_to_coco_json(dataset_name, output_dir)
            self._metadata.json_file = cache_path
        self._is_splits = "all" in dataset_name or "base" in dataset_name \
            or "novel" in dataset_name
        self._base_classes = [
            8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35,
            36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54,
            55, 56, 57, 58, 59, 60, 61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80,
            81, 82, 84, 85, 86, 87, 88, 89, 90,
        ]
        self._novel_classes = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21,
                               44, 62, 63, 64, 67, 72]

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)
        self._do_evaluation = "annotations" in self._coco_api.dataset

    def reset(self):
        self._predictions = []
        self._coco_results = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(
                    instances, input["image_id"])
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))
            if not comm.is_main_process():
                return {}

        if len(self._predictions) == 0:
            self._logger.warning(
                "[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(
                self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(self._predictions, f)

        self._results = OrderedDict()
        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate self._predictions on the instance detection task.
        Fill self._results with the metrics of the instance detection task.
        """
        self._logger.info("Preparing results for COCO format ...")
        self._coco_results = list(
            itertools.chain(*[x["instances"] for x in self._predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in self._coco_results:
                result["category_id"] = reverse_id_mapping[result["category_id"]]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, self._dataset_name+"_coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(self._tasks):
            if self._is_splits:
                self._results[task] = {}
                for split, classes, names in [
                        ("all", None, self._metadata.get("thing_classes")),
                        ("base", self._base_classes, self._metadata.get("base_classes")),
                        ("novel", self._novel_classes, self._metadata.get("novel_classes"))]:
                    if "all" not in self._dataset_name and \
                            split not in self._dataset_name:
                        continue

                    self._logger.info(f'Evaluating results for {task} and dataset {split}') 

                    coco_eval = (
                        _evaluate_predictions_on_coco(
                            self._coco_api, self._coco_results, task, classes,
                        )
                        if len(self._coco_results) > 0
                        else None  # cocoapi does not handle empty results very well
                    )
                    res_ = self._derive_coco_results(coco_eval, task, class_names=names)
                    res = {}
                    for metric in res_.keys():
                        # if len(metric) <= 4:
                            if split == "all":
                                res[metric] = res_[metric]
                            elif split == "base":
                                res["b"+metric] = res_[metric]
                            elif split == "novel":
                                res["n"+metric] = res_[metric]
                    self._results[task].update(res)
                if task != "count":
                    # add "AP" if not already in
                    if "AP" not in self._results[task]:
                        if "nAP" in self._results[task]:
                            self._results[task]["AP"] = self._results[task]["nAP"]
                        else:
                            self._results[task]["AP"] = self._results[task]["bAP"]
            else:
                coco_eval = (
                    _evaluate_predictions_on_coco(
                        self._coco_api, self._coco_results, task,
                    )
                    if len(self._coco_results) > 0
                    else None  # cocoapi does not handle empty results very well
                )
                res = self._derive_coco_results(
                    coco_eval, task,
                    class_names=self._metadata.get("thing_classes")
                )
                self._results[task] = res
    '''
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

        if coco_eval is None:
            self._logger.warn("No predictions from the model! Set scores to -1")
            return {metric: -1 for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100) \
                for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + \
                create_small_table(results)
        )

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(
            *[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results
    '''
    def compute_count(self,coco_eval ,class_names,score_thr=[0.05,0.3,0.5]):
        """
        Compute count for object detection.
        """
        coco_gt=coco_eval.cocoGt
        coco_dt=coco_eval.cocoDt
        cates=coco_eval.cocoGt.getCatIds(class_names)
        mae={thr:[] for thr in score_thr}
        mse={thr:[] for thr in score_thr}
        gt_annos=coco_gt.imgToAnns
        dt_annos=coco_dt.imgToAnns
        cates2id=dict(zip(cates,range(len(cates))))
        gt_cates_nums=[]
        for img_id in gt_annos:
            gt_cates_num=np.zeros(len(cates))
            dt_cates_num={thr:np.zeros(len(cates)) for thr in score_thr}
            for j in range(len(gt_annos[img_id])):
                cate=gt_annos[img_id][j]['category_id']
                if cate in cates2id:
                    gt_cates_num[cates2id[cate]]+=1
            gt_cates_nums.append(gt_cates_num)
            for j in range(len(dt_annos[img_id])):
                cate=dt_annos[img_id][j]['category_id']
                score=dt_annos[img_id][j]['score']
                for thr in score_thr:
                    if cate in cates2id:
                        if score>thr:
                            dt_cates_num[thr][cates2id[cate]]+=1
            for i in score_thr:
                mae[i].append(np.abs(dt_cates_num[i]-gt_cates_num))
                mse[i].append(np.square(dt_cates_num[i]-gt_cates_num))
        # gt_cates_nums_vis=np.stack(gt_cates_nums)>1
        mae={thr:np.stack(mae[thr]) for thr in score_thr}
        mse={thr:np.stack(mse[thr]) for thr in score_thr}

        mae_mean_final={thr:np.mean(mae[thr]) for thr in score_thr}
        mse_mean_final={thr:np.mean(mse[thr]) for thr in score_thr}
        mae_sigma_final={thr:np.sqrt(np.std(mae[thr])) for thr in score_thr}
        mse_sigma_final={thr:np.sqrt(np.std(mse[thr])) for thr in score_thr}

        results={}
        data_names=['mae_m','mse_m','mae_s','mse_s']
        for i,data in enumerate([mae_mean_final,mse_mean_final,mae_sigma_final,mse_sigma_final]):
            for j,thr in enumerate(score_thr):
                results[data_names[i]+'_'+str(thr)]=data[thr]

        return results


    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
            """
            Derive the desired score numbers from summarized COCOeval.

            Args:
                coco_eval (None or COCOEval): None represents no predictions from model.
                iou_type (str):
                class_names (None or list[str]): if provided, will use it to predict
                    per-category AP.

            Returns:
                a dict of {metric name: score}
            """

            metrics = {
                "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
                "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
                "count":["MAE","MSE"]
            }[iou_type]

            if iou_type == "count":
                return self.compute_count(coco_eval, class_names)

            if coco_eval is None:
                self._logger.warn("No predictions from the model!")
                return {metric: float("nan") for metric in metrics}

            # the standard metrics
            results = {
                metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
                for idx, metric in enumerate(metrics)
            }
            self._logger.info(
                "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
            )
            if not np.isfinite(sum(results.values())):
                self._logger.info("Note that some metrics cannot be computed.")

            if class_names is None or len(class_names) <= 1:
                return results
            # Compute per-category AP
            # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
            precisions = coco_eval.eval["precision"]
            # precision has dims (iou, recall, cls, area range, max dets)
            assert len(class_names) == precisions.shape[2]

            results_per_category = []
            # TODO(): Rewrite this more modularly
            results_per_category_AP50 = []
            for idx, name in enumerate(class_names):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category.append(("{}".format(name), float(ap * 100)))

                # Compute for AP50
                # 0th first index is IOU .50
                precision = precisions[0, :, idx, 0, -1]
                precision = precision[precision > -1]
                ap = np.mean(precision) if precision.size else float("nan")
                results_per_category_AP50.append(("{}".format(name), float(ap * 100)))

            table = _tabulate_per_category(results_per_category)
            self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

            tableAP50 = _tabulate_per_category(results_per_category_AP50, "AP50")
            self._logger.info("Per-category {} AP50: \n".format(iou_type) + tableAP50)

            results.update({"AP-" + name: ap for name, ap in results_per_category})
            # Update AP50
            results.update({"AP50-" + name: ap for name, ap in results_per_category_AP50})
            return results 

def _tabulate_per_category(result_per_cat, AP_TYPE='AP'):
    """Given a list of results per category, returns a table with max 6 columns

    :param result_per_cat: Results per category. List of tuples like ('AP-car', 0.42)
    :type result_per_cat: List[Tuple(str,int)]
    :param AP_TYPE: Name of the performance metric the results are for, defaults to 'AP'
    :type AP_TYPE: str, optional
    """
    N_COLS = min(6, len(result_per_cat) * 2)
    results_flatten = list(itertools.chain(*result_per_cat))
    results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        results_2d,
        tablefmt="pipe",
        floatfmt=".3f",
        headers=["category", AP_TYPE] * (N_COLS // 2),
        numalign="left",
    )
    return table

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
 
    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [ 
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "cou:::::;;;;;nts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoint = instances.has("pred_keypoints")
    if has_keypoint:
            keypoints = instances.pred_keypoints
            keypoints = keypoints.reshape(num_instance, -1).tolist()
    has_point=instances.has("point_det")
    if has_point:
        point_det=instances.point_det
        point_det=point_det.reshape(num_instance, -1).tolist()
        point_seg=instances.point_seg
        point_seg=point_seg.reshape(num_instance, -1).tolist()


    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
           result["segmentation"] = rles[k]
        if has_keypoint:
            result["keypoints"] = keypoints[k]
            result["num_keypoints"] = sum(k != 0 for k in keypoints[k][2::3])
        if has_point:
            result["point_det"]=point_det[k]
            result["point_seg"]=point_seg[k]

        results.append(result)
    return results

def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, catIds=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0
    if iou_type =='count':
        iou_type='bbox'

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if catIds is not None:
        coco_eval.params.catIds = catIds
    if iou_type == "keypoints":
        coco_eval.params.kpt_oks_sigmas = np.ones(52)/10*3.5

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
