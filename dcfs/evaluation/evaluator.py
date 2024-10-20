import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock
from dcfs.dataloader import MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import cv2

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def build_suppport_prototypes_voc(dataloader, model):
    support_features, all_labels = [], []
    with inference_context(model), torch.no_grad():
        for index in range(len(dataloader.dataset)):
            if index>100:
                break
            inputs = [dataloader.dataset[index]]
            if len(inputs[0]['instances']) == 0:
                continue
            support_feature = model(inputs, get_features=True)
            for cate in support_feature:
                support_features.append(support_feature[cate][0])
        support_features = torch.stack(support_features).mean(0)
    return support_features
def build_suppport_prototypes(dataloader,model,kpt_num):

        support_features, all_labels = {}, []
        pose_vis={}
        with inference_context(model), torch.no_grad():
            for index in range(len(dataloader.dataset)):
                inputs = [dataloader.dataset[index]]
                if len(inputs[0]['instances']) == 0:
                    continue
                support_feature=model(inputs,get_features=True)
                for cate in support_feature:
                    if cate not in support_features:
                        support_features[cate]=[]
                    support_features[cate].append(support_feature[cate][0])
                # load support images and gt-boxes
                gt_classes = inputs[0]['instances'].gt_classes
                gt_classes = torch.unique(gt_classes)
                targets=inputs[0]['instances']
                for cate in gt_classes:
                    selected_inds = \
                    torch.where(targets.gt_classes == cate)[0]
                    targets_i = targets[selected_inds]
                    vis_points = targets_i.gt_keypoints.tensor[..., 2] > 0
                    vis_points = vis_points.sum(0)
                    if cate.item() not in pose_vis:
                        pose_vis[cate.item()]=[]
                    pose_vis[cate.item()].append(vis_points)
            # concat
            for cate in support_features:
                support_features[cate] = torch.stack(support_features[cate]).mean(0)
                pose_vis[cate] = torch.stack(pose_vis[cate]).sum(0)>0
            support_features_values = torch.stack([support_features[k] for k in support_features]).mean(0)
            for cate in support_features:
                support_features[cate][:,:-kpt_num] = support_features_values[:,:-kpt_num]
            # all_labels = torch.cat(all_labels, dim=0)

        return support_features,pose_vis
def adjust_kpy_vis(targets,pose_vis):
    category_id=targets.gt_classes
    for j,cate in enumerate(category_id):
        targets.gt_keypoints.tensor[j][...,2]*=pose_vis[cate.item()]
    return targets

def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    if cfg.TEST.META_TEST:
        # if cfg.DATASETS has MODEL_INIT
        model_init_dataset_name = cfg.DATASETS.MODEL_INIT[0] if cfg.DATASETS.MODEL_INIT else None
        if model_init_dataset_name is None:
            model_init_dataset_name = cfg.DATASETS.TRAIN[0]
        model_init_dataset = build_detection_test_loader(cfg, model_init_dataset_name)
        kpt_num=cfg.MODEL.ROI_BOX_HEAD.KEYPOINT_POINTS_NUM
        if 'voc' not in model_init_dataset_name:
            support_features,pose_vis=build_suppport_prototypes(model_init_dataset,model,kpt_num)
        else:
            support_features=build_suppport_prototypes_voc(model_init_dataset,model)
        logger.info( "*************Model Init done*********************" )


    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            if cfg.TEST.META_TEST:
                outputs = model(inputs,support_features)
                if len(inputs[0]['instances']) > 0 and  'voc' not in model_init_dataset_name:
                    inputs[0]['instances']=adjust_kpy_vis(inputs[0]['instances'],pose_vis)
            else:
                outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)
            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
