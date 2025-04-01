'''Convert dataset annotation label from csv to json format
input_dir:
    |-annotations
    |--label_name.txt
    |--train.csv
    |--val.csv
    |--test.csv

label_name.txt: (row record)
label_id label_name

train/val/test.csv
id,video_id,start_frame,end_frame,class,total_frames,fps

output json format:
database: {
    video_id:{
        subset: $subset,
        duration: $duration_in_sec,
        fps: $fps,
        total_frames: $total_frames,
        annotations: [
            {
                segment(frames): [$start_frame, $end_frame],
                segment: [$start_time, $end_time],
                label: $label_name,
                label_id: $label_id
            },
            ...
        ]
    },
    ...
}
'''
import os
import argparse
import json
import pandas as pd

def convert_csv2json(args):
    root = args.root
    assert os.path.isdir(root), f'root {root} should exist and be a directory'
    ann_files = os.listdir(root)
    subsets = [i.split('.csv')[0] for i in ann_files if '.csv' in i]
    ann_files = [os.path.join(root, i) for i in ann_files if '.csv' in i]
    print(ann_files)
    
    label_file = os.path.join(root, 'label_name.txt')
    assert os.path.exists(label_file)
    label_mapping = get_label_mapping(label_file)
    start_index = sorted(label_mapping.keys())[0]
    print(f'Total {len(label_mapping)} actions, index start from {start_index}')
    
    data = {'database':{}}
    for file,subset in zip(ann_files,subsets):
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        print(f'{subset}: {len(df)} records')
        subset_data = single_file_convert(df, subset, label_mapping)
        data['database'].update(subset_data)
    
    print(f"Total {len(data['database'])} videos")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_json(data, args.output)
    
def get_label_mapping(path) -> dict:
    label_mapping = {}
    with open(path, 'r')as f:
        for line in f:
            label_id, label_name = line.strip().split("\t")
            label_mapping[label_id] = label_name
    return label_mapping

def single_file_convert(dataframe, subset_name, label_mapping) -> dict:
    data = {}
    
    for _, row in dataframe.iterrows():
        video_id = row['video_id']
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        total_frames = row['total_frames']
        fps = row['fps']
        label_id = row['class']
        duration_in_sec = total_frames / fps

        if video_id not in data:
            data[video_id] = {
                'subset': subset_name,
                'duration': duration_in_sec,
                'fps': fps,
                'total_frames': total_frames,
                'annotations': []
            }

        data[video_id]['annotations'].append({
            'segment(frames)': [start_frame, end_frame],
            'segment': [start_frame / fps, end_frame / fps],
            'label': label_mapping.get(str(label_id), 'unknown'),
            'label_id': label_id
        })

    return data

def write_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",'-i', type=str, help='root to annotation directory')
    parser.add_argument("--output",'-o', type=str, help='output path to the annotation json file')
    
    args = parser.parse_args()
    convert_csv2json(args)

'''
Usage:
    python tools/convert_label_csv2json.py -i /mnt/cephfs/dataset/MMA-52/annotations -o /mnt/cephfs/dataset/MMA-52/annotations/label.json
'''