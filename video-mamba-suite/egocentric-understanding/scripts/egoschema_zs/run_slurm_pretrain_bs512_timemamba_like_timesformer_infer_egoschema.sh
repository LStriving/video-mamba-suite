#!/usr/bin/env bash

export MASTER_PORT=$((12000 + $RANDOM % 20000))
set -x


OUTPUT_DIR="./work_dirs/lavila_pretrain_baseline_vitb_bs512_timemamba_like_timesformer"
DATA_ROOT="s-in-hdd:s3://videos/ego4d/videos_short320_chunked_15s/"
DATA_ROOT_VAL="/mnt/petrelfs/chenguo/data/egoschema/videos"
VIDEO_CHUNK_LENGTH=15
CLIP_LENGTH=16
CLIP_STRIDE=16

PARTITION=$1
JOB_NAME=$2
GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-1}
# SRUN_ARGS=${SRUN_ARGS:-"--quotatype=spot --async -o ${OUTPUT_DIR}/slurm.log"}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}  # Any arguments from the forth one are captured by this

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u engine/main_lavila_pretrain.py \
    --root  ${DATA_ROOT} \
    --model CLIP_TimeMamba_like_timesformer \
    --root-val ${DATA_ROOT_VAL} \
    --output-dir ${OUTPUT_DIR} \
    --video-chunk-length ${VIDEO_CHUNK_LENGTH} \
    --clip-length ${CLIP_LENGTH} \
    --clip-stride ${CLIP_STRIDE} \
    --batch-size 1 \
    --use-flash-attn \
    --use-fast-conv1 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --resume /mnt/petrelfs/chenguo/workspace/video-mamba-suite-data/model_zoo/clip_timemamba_vanilla_base_ego4d4m_bs512_f4.pt \
    --evaluate \
    --evaluate-task egoschema \
    --dataset-val egoschema  \
    --val-metadata ./datasets/Ego4D/egoschema_questions.json




    

