# Video Mamba Suite: Dense Video Captioning

Implementation for Mamba-based PDVC (ICCV 2021) 
[[paper]](https://arxiv.org/abs/2108.07781)

**With additional supports:**
* Mamba-based feature encoder
* DDP Training (not supported in the official repo)

**Table of Contents:**
* [Preparation](#preparation)
* [Training and Validation](#training-and-validation)
  + [Download Video Features](#download-video-features)
  + [Dense Video Captioning](#dense-video-captioning)
  + [Video Paragraph Captioning](#video-paragraph-captioning)
* [Performance](#performance)
  + [Dense video captioning](#dense-video-captioning)
* [Acknowledgement](#acknowledgement)


## Preparation
1. Install pytorch and dependencies. The code is run successfully on (1) torch==1.13.1+cu117 or (2) torch==2.1.2+cu118, other pytorch versions may also work. 
```bash
conda install ffmpeg
pip install -r requirement.txt
```

2. Compile the deformable attention layer (requires GCC >= 5.4). 
```bash
cd pdvc/ops
sh make.sh
```

3. Install mamba follow the main README.md, make sure you can import mamba successfully
```bash
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba_new import Mamba 
```

4. Make sure the submodules SODA and pycocoevalcap exists, which are used for evaluation.
```bash
from densevid_eval3.SODA.soda import SODA
```
If the modules do not exist, please make sure you have cloned the repo with '--recursive', or run
```bash
cd path/to/your/video-mamba-suite/
git submodule update --init --recursive
```


## Training 

### Download Video Features

```bash
cd data/anet/features
bash download_anet_c3d.sh
# bash download_anet_tsn.sh
# bash download_i3d_vggish_features.sh
# bash download_tsp_features.sh
```


The preprocessed C3D features have been uploaded to [baiduyun drive](https://pan.baidu.com/s/1Ehvq1jNiJrhgA00mOG25zQ?pwd=fk2p)

### Download YouCook2 Video Features
```bash
cd data/yc2/features
bash download_yc2_tsn_features.sh
```


### Dense Video Captioning
1. PDVC-DeformableTransformer with learnt proposals
```
# E.g. Train on ANet dataset with 8gpus
torchrun --nproc_per_node=8 train.py \
--cfg_path cfgs/anet_c3d_pdvc.yml \ 
--disable_cudnn 1 \
--save_dir /path/to/your/folder/anet_c3d_pdvc_deformableTransformer_8gpus/ \
--encoder_type deformable \ 
```

2. PDVC-Mamba with learnt proposals

```
# E.g. Train on ANet dataset with 8gpus
torchrun --nproc_per_node=8 train.py \
--cfg_path cfgs/anet_c3d_pdvc.yml \ 
--disable_cudnn 1 \
--save_dir /path/to/your/folder/anet_c3d_pdvc_mamba_8gpus/ \
--encoder_type mamba-dbm \ 

```

```
# E.g. Train on YouCook dataset with 1gpu
torchrun --nproc_per_node=1 train.py \
--cfg_path cfgs/yc2_tsn_pdvc.yml \ 
--disable_cudnn 1 \
--save_dir /path/to/your/folder/yc2_tsn_pdvc_mamba_1gpu/ \
--encoder_type mamba-dbm \ 

```


### Video Paragraph Captioning
1. PDVC-Mamba with gt-proposals

```
# E.g. Train on ANet dataset with 8gpus
torchrun --nproc_per_node=8 train.py \
--cfg_path cfgs/anet_c3d_pdvc_gt.yml \ 
--disable_cudnn 1 \
--criteria_for_best_ckpt pc \
--save_dir /path/to/your/folder/anet_c3d_pc_mamba_8gpus/ \
--encoder_type mamba-dbm \ 
```

## Performance
### Dense video captioning on ANet (with learnt proposals)

|  Model | Features | config_path |   Url   | Recall | Precision |    B-4   | M | R |  C | SODA |
|  ----  |  ----    |   ----  |  ----  |  ----   |  ----  |   ----  |  ----  |  ----  |  ----  | ---- |
| PDVC-Deformable   | C3D  | cfgs/anet_c3d_pdvc.yml |  todo |  51.74   |  56.11  | 1.75  |  6.73  |  14.73  | 26.07  |  5.47  |
| PDVC-Mamba   | C3D  | cfgs/anet_c3d_pdvc.yml | todo  |  52.45   |  56.33  | 1.76 |  7.16 | 14.83 | 26.77 |   5.27  |   


### Dense video captioning on YouCook2 (with learnt proposals)
|  Model | Features | config_path |   Url   | Recall | Precision |    B-4   | M | R |  C | SODA |
|  ----  |  ----    |   ----  |  ----  |  ----   |  ----  |   ----  |  ----  |  ----  |  ----  | ---- |
| PDVC-Deformable   | TSN  | cfgs/yc2_tsn_pdvc.yml |  todo |  23.00  |  31.12  | 0.73  |  4.25  |  9.31  | 20.48  |  4.02  |
| PDVC-Mamba   | TSN  | cfgs/yc2_tsn_pdvc.yml | todo  |  25.27   |  32.41  | 0.86 |  4.44 | 9.62 | 21.90 |  4.32  |


Notes:
'B-4', 'M', 'R', 'C' refers to 'BLEU-4', 'METEOR', 'ROUGE-L' and 'CIDER'. More details can be found in [PDVC](https://github.com/ttengwang/PDVC/tree/main)

## Acknowledgement

The codebase is based on [PDVC](https://github.com/ttengwang/PDVC/tree/main).
We thanks the authors for their efforts.
