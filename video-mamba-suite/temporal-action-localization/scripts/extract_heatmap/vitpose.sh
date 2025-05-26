python libs/utils/inference_keypoints_api.py \
 --model_path /mnt/cephfs/home/zhoukai/Codes/vfss/vfss_keypoint/models/pytorch/vitpose_trace.pt \
 --kalman True \
 --normal_kalman False \
 --sigma 4 \
 --output_root /mnt/cephfs/dataset/swallow_heatmap56_sigma4_vitpose_kalman