import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from dcfs.config import get_cfg, set_global_cfg
from dcfs.evaluation import DatasetEvaluators, verify_results
from dcfs.engine import DefaultTrainer, default_argument_parser, default_setup


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        tasks = ["bbox",]
        if cfg.MODEL.MASK_ON:
            tasks.append("segm")
        if cfg.MODEL.KEYPOINT_ON:
            tasks.append("keypoints")
        tasks.append("count")
        tasks=tuple(tasks)
            
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from dcfs.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, tasks, True, output_folder))
        if evaluator_type == "pascal_voc":
            from dcfs.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

def add_new_config(cfg):
    cfg.MODEL.ROI_BOX_HEAD.BBOX_POINTS_NUM=16
    cfg.MODEL.ROI_BOX_HEAD.SEG_POINTS_NUM=32
    cfg.MODEL.ROI_BOX_HEAD.REG_WEIGHTS=1.0
    cfg.MODEL.ROI_BOX_HEAD.DET_WEIGHTS=1
    cfg.MODEL.ROI_BOX_HEAD.SEG_WEIGHTS=1
    cfg.MODEL.ROI_BOX_HEAD.POSE_WEIGHTS=1
    cfg.MODEL.ROI_BOX_HEAD.USE_RLE_LOSS=False
    cfg.SOLVER.LOG_PERIOD=20
    cfg.MODEL.ROI_BOX_HEAD.FREEZE_REG=False
    cfg.MODEL.ROI_BOX_HEAD.KEYPOINT_POINTS_NUM=60
    cfg.DATASETS.MODEL_INIT=None
    cfg.TEST.META_TEST=False
    cfg.MODEL.ROI_BOX_HEAD.REFINED=True
    cfg.DATALOADER.REPEAT_TIMES=5
    cfg.MODEL.ROI_BOX_HEAD.ANGLE_STRIDE=2
    cfg.MODEL.ROI_BOX_HEAD.DECODER_NUM=2
    cfg.INPUT.CUSROMARGU=False
    cfg.MODEL.ROI_BOX_HEAD.USE_ANGLE_LOSS=True
    return cfg

def setup(args):
    cfg = get_cfg()
    cfg=add_new_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.dist_url = args.dist_url+'{}'.format(args.port)
    print(args.dist_url)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
