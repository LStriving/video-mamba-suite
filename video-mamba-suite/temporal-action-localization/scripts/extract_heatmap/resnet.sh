python libs/utils/inference_keypoints_api.py \
 --model_path /mnt/cephfs/home/zhoukai/Codes/vfss/vfss_keypoint/models/pytorch/resnet_trace.pt \
 --kalman True \
 --sigma 4 \
 --output_root /mnt/cephfs/dataset/swallow_heatmap56_sigma4_resnet_kalman