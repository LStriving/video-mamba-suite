#!/bin/bash

# 启用 globstar 以支持 ** 匹配子目录
shopt -s globstar

# 遍历当前目录及子目录中的所有 AVI 文件
for file in data/swallow/external_processed/videos/*.avi; do
    # 检查是否为文件（避免处理同名的目录）
    if [ -f "$file" ]; then
        # 生成输出文件名（将 .avi 替换为 .mp4）
        output_file="${file%.avi}.mp4"

        # 检查是否存在：
        # 检查输出文件是否已存在
        if [ -f "$output_file" ]; then
            echo "跳过，输出文件已存在：$output_file"
            continue
        fi
        
        echo "正在转换文件：$file"
        
        # 使用 FFmpeg 进行转换
        ffmpeg -i "$file" -c:v libx264 -preset medium -c:a aac -b:a 128k -y "$output_file"
        
        # 检查转换是否成功
        if [ $? -eq 0 ]; then
            echo "转换完成：$output_file"
        else
            echo "转换失败：$file"
        fi
    fi
done

# 恢复 globstar 设置
shopt -u globstar