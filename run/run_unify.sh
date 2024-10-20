#!/usr/bin/env bash
NET=$1
NUNMGPU=$2
EXPNAME=$3

DETWEIGHT=1
SEGWEIGHT=1
KPTWEIGHT=1


SAVEDIR=workspace/DCFS/coco-seg/${EXPNAME}_${DETWEIGHT}_${SEGWEIGHT}_${KPTWEIGHT} #<-- change it to you path
PRTRAINEDMODEL=pretrained_models/           #<-- change it to you path


if [ "$NET"x = "r101"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-101.pkl                            
  IMAGENET_PRETRAIN_TORCH=${PRTRAINEDMODEL}/resnet101-cd907fc2.pth
fi

if [ "$NET"x = "r50"x ]; then
  IMAGENET_PRETRAIN=${PRTRAINEDMODEL}/MSRA/R-50.pkl                             
  IMAGENET_PRETRAIN_TORCH=${IMAGENET_PRETRAIN}/resnet50-19c8e357.pth           
fi


# ------------------------------- Base Pre-train ---------------------------------- #
python3 main.py --num-gpus ${NUNMGPU} --config-file configs/points_unify/dcfs_det_r101_base.yaml \
    --opts MODEL.WEIGHTS ${IMAGENET_PRETRAIN}       \
           MODEL.MASK_ON 'True'  \
           MODEL.ROI_BOX_HEAD.DET_WEIGHTS  ${DETWEIGHT}\
           MODEL.ROI_BOX_HEAD.SEG_WEIGHTS  ${SEGWEIGHT}\
           MODEL.ROI_BOX_HEAD.POSE_WEIGHTS  ${KPTWEIGHT}\
           OUTPUT_DIR ${SAVEDIR}/dcfs_seg_${NET}_base


# ------------------------------ Model Preparation -------------------------------- #
python3 tools/model_surgery.py --dataset coco --method remove     \
    --src-path ${SAVEDIR}/dcfs_seg_${NET}_base/model_final.pth                        \
    --save-dir ${SAVEDIR}/dcfs_seg_${NET}_base                                        \
    --param-name roi_heads.box_predictor.cls_score

BASE_WEIGHT=${SAVEDIR}/dcfs_seg_${NET}_base/model_reset_remove.pth


# ------------------------------ Novel Fine-tuning For Det & Seg -------------------------------- #
# --> 1. TFA-like, i.e. run seed0~9 (10times) for FSIS on COCO (20 classes)
classloss="DC" # "CE"
for shot in 1 5
do
  for seed in `seq 0 1 9`
    do
        TRAIN_NOVEL_NAME=unify_trainval_novel_${shot}shot_seed${seed}
        TEST_NOVEL_NAME=unify_test_novel
        CONFIG_PATH=configs/points_unify/dcfs_fsod_${NET}_novel_${shot}shot_seedx.yaml
        OUTPUT_DIR=${SAVEDIR}/dcfs_fsis_${NET}_novel/tfa-like-${classloss}/${shot}shot_seed${seed}
        python3 main.py --num-gpus ${NUNMGPU} --config-file ${CONFIG_PATH}   --port  28${shot}${seed}      \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}           \
                       MODEL.MASK_ON 'True'     \
                       MODEL.ROI_BOX_HEAD.REFINED True \
                       MODEL.ROI_BOX_HEAD.DET_WEIGHTS  ${DETWEIGHT}\
                       MODEL.ROI_BOX_HEAD.SEG_WEIGHTS  ${SEGWEIGHT}\
                       MODEL.ROI_BOX_HEAD.POSE_WEIGHTS  ${KPTWEIGHT}\
                       MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE ${classloss} \
                       DATASETS.TRAIN "('"${TRAIN_NOVEL_NAME}"',)" \
                       DATASETS.TEST  "('"${TEST_NOVEL_NAME}"',)"  \
                       TEST.PCB_MODELPATH  ${PRTRAINEDMODEL}/resnet101-cd907fc2.pth \
                       TEST.PCB_MODELTYPE $NET                                \
                       MODEL.ROI_BOX_HEAD.FREEZE_REG ${FROZE_REG}  \
                       DATALOADER.REPEAT_TIMES ${REPEAT}   \
                       MODEL.ROI_BOX_HEAD.DECODER_NUM 1 \
           MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True &
#        rm ${OUTPUT_DIR}/model_final.pth
    done
    wait
done




# ------------------------------ Novel Fine-tuning For Kpt -------------------------------- #
# --> 1. TFA-like, i.e. run seed0~9 (10times) for FSIS on COCO (20 classes)
DETWEIGHT=1
SEGWEIGHT=1
KPTWEIGHT=0
SAVEDIR=workspace/DCFS/coco-seg/${EXPNAME}_${DETWEIGHT}_${SEGWEIGHT}_${KPTWEIGHT} #<-- change it to you path
classloss="DC" # "CE"
for shot in 1 5
do
  for seed in `seq 0 1 9`
    do
        TRAIN_NOVEL_NAME=unify_trainval_novel_${shot}shot_seed${seed}
        TEST_NOVEL_NAME=unify_test_novel
        CONFIG_PATH=configs/points_unify/dcfs_fsod_${NET}_novel_${shot}shot_seedx.yaml
        OUTPUT_DIR=${SAVEDIR}/dcfs_fsis_${NET}_novel/tfa-like-${classloss}/${shot}shot_seed${seed}
        python3 main.py --num-gpus ${NUNMGPU} --config-file ${CONFIG_PATH}   --port  28${shot}${seed}      \
                --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}           \
                       MODEL.MASK_ON 'True'     \
                       MODEL.ROI_BOX_HEAD.REFINED True \
                       MODEL.ROI_BOX_HEAD.DET_WEIGHTS  ${DETWEIGHT}\
                       MODEL.ROI_BOX_HEAD.SEG_WEIGHTS  ${SEGWEIGHT}\
                       MODEL.ROI_BOX_HEAD.POSE_WEIGHTS  ${KPTWEIGHT}\
                       MODEL.ROI_BOX_HEAD.BBOX_CLS_LOSS_TYPE ${classloss} \
                       DATASETS.TRAIN "('"${TRAIN_NOVEL_NAME}"',)" \
                       DATASETS.TEST  "('"${TEST_NOVEL_NAME}"',)"  \
                       TEST.PCB_MODELPATH  ${PRTRAINEDMODEL}/resnet101-cd907fc2.pth \
                       TEST.PCB_MODELTYPE $NET                                \
                       MODEL.ROI_BOX_HEAD.FREEZE_REG ${FROZE_REG}  \
                       DATALOADER.REPEAT_TIMES ${REPEAT}   \
                       MODEL.ROI_BOX_HEAD.DECODER_NUM 1 \
           MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG True &
#        rm ${OUTPUT_DIR}/model_final.pth
    done
    wait
done