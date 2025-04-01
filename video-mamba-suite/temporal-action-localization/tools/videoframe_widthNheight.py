import os
import cv2

def check_video_dimensions(root_dir):
    """
    检查数据集目录下所有视频的帧宽和帧高是否一致
    
    参数:
        root_dir: 数据集根目录路径，结构应为:
            --root_dir
            ---train/val/test
            -----video001.mp4
            ...
    """
    # 存储所有视频的尺寸信息
    dimensions = {}
    inconsistent_videos = []
    first_dimension = None
    
    # 遍历根目录下的所有子目录（train/val/test）
    for subset in os.listdir(root_dir):
        subset_path = os.path.join(root_dir, subset)
        if not os.path.isdir(subset_path):
            continue
            
        print(f"\n检查子集: {subset}")
        
        # 遍历子目录中的所有视频文件
        for video_file in os.listdir(subset_path):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
                
            video_path = os.path.join(subset_path, video_file)
            
            # 使用OpenCV读取视频属性
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"警告: 无法打开视频文件 {video_path}")
                continue
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # 记录尺寸信息
            dimensions[video_path] = (width, height)
            print(f"{video_file}: {width}x{height}")
            
            # 检查是否与第一个视频尺寸一致
            if first_dimension is None:
                first_dimension = (width, height)
            elif (width, height) != first_dimension:
                inconsistent_videos.append(video_path)
    
    # 输出检查结果
    if not dimensions:
        print("\n未找到任何视频文件")
        return
        
    print("\n检查结果:")
    print(f"共检查 {len(dimensions)} 个视频文件")
    print(f"基准尺寸: {first_dimension[0]}x{first_dimension[1]}")
    
    if inconsistent_videos:
        print("\n以下视频的尺寸不一致:")
        for video in inconsistent_videos:
            print(f"{video}: {dimensions[video][0]}x{dimensions[video][1]}")
    else:
        print("\n所有视频的尺寸一致")

if __name__ == "__main__":
    # 使用示例 - 替换为您的实际数据集路径
    dataset_root = "/mnt/cephfs/dataset/MMA-52"
    check_video_dimensions(dataset_root)