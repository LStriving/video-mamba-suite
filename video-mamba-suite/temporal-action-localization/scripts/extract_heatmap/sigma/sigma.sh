sigmas=(6 8 1 2 3 5)
for sigma in "${sigmas[@]}"
do 
    echo "Processing sigma: $sigma"
    # extract heatmaps
    python libs/utils/inference_keypoints_api.py \
    --sigma $sigma \
    --output_root /mnt/cephfs/dataset/swallow_heatmap56_sigma${sigma} \
    --kalman True 

    echo "Extracted heatmaps for sigma: $sigma" > output/sigma_${sigma}_heatmap_extraction.log
    echo "CMD: python libs/utils/inference_keypoints_api.py \
    --sigma $sigma \
    --output_root /mnt/cephfs/dataset/swallow_heatmap56_sigma${sigma} \
    --kalman True " >> output/sigma_${sigma}_heatmap_extraction.log
done

