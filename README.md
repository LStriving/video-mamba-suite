<div align="center">
<img src="./assets/logo_trans.png" style='width: 25%'>
<h2></img><a href="https://arxiv.org/abs/2403.09626">Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding</a></h2>

[Guo Chen](https://scholar.google.com/citations?user=lRj3moAAAAAJ), [Yifei Huang](https://scholar.google.com/citations?user=RU8gNcgAAAAJ), [Jilan Xu](https://scholar.google.com/citations?user=mf2U64IAAAAJ), [Baoqi Pei](), [Zhe Chen](https://scholar.google.com/citations?user=j1rq_lYAAAAJ), [Zhiqi Li](https://scholar.google.com/citations?user=H2fJLqEAAAAJ), [Jihao Wang](), [Kunchang Li](https://scholar.google.com/citations?user=D4tLSbsAAAAJ), [Tong Lu]() and [Limin Wang](https://scholar.google.com/citations?user=HEuN8PcAAAAJ).

</div>




## Introduction
![teaser](./assets/teaser.jpg)
Understanding videos is one of the fundamental directions in computer vision research, with extensive efforts dedicated to exploring various architectures such as RNN, 3D CNN, and Transformers.
The newly proposed architecture of state space model, e.g, Mamba, shows promising traits to extend its success in long sequence modeling to video modeling. 
To assess whether Mamba can be a viable alternative to Transformers in the video understanding domain, in this work, we conduct a comprehensive set of studies, probing different roles Mamba can play in modeling videos, while investigating diverse tasks where Mamba could exhibit superiority. 
We categorize Mamba into four roles for modeling videos, deriving a **Video Mamba Suite** composed of 14 models/modules, and evaluating them on 12 video understanding tasks. Our extensive experiments reveal the strong potential of Mamba on both video-only and video-language tasks while showing promising efficiency-performance trade-offs.
We hope this work could provide valuable data points and insights for future research on video understanding.



## 📢 News


(2024/04/01) The code of related tasks for [Video Temporal Adapter](#mamba-as-video-temporal-adapter) and [Spatial-temporal modeling](#mamba-for-spatial-temporal-modeling) has been released. They contain `multi-instance retrieval`, `action recognition`, `long video quesion-answer` on [EK100](https://epic-kitchens.github.io/2024), [Kinetics400](https://deepmind.google/), and [EgoSchema](https://egoschema.github.io/). 
If there exist bug or missing packages, please submit a new issue to tell me.

(2024/03/21) The code of related tasks for [Video Temporal Modeling](#mamba-for-video-temporal-modeling) and [Cross-modal Interaction](#mamba-for-cross-modal-interaction) has been released. 

(2024/03/15) 🔄The repository is public.

(2024/03/12) 🔄The repository is created.



## Preliminary Installation

1. Install the preliminary requirements.

```bash
# clone video-mamba-suite
git clone --recursive https://github.com/OpenGVLab/video-mamba-suite.git

# create environment
conda create -n video-mamba-suite python=3.9
conda activate video-mamba-suite

# install pytorch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirement.txt

# install mamba
# cd causal-conv1d
# python setup.py develop
# cd ..
pip install causal-conv1d==1.1.1
cd mamba
python setup.py develop
cd ..
```

2. For each tasks, enter their folders to follow installation instructions.

3. If `requirement.txt` is missing some libraries, please propose an issue as soon as possible.



## Supported tasks

### Mamba for Video Temporal Modeling
| task                                                                                       | supported datasets                               |
| :----------------------------------------------------------------------------------------- | :----------------------------------------------- |
| [Temporal Action Localization](./video-mamba-suite/temporal-action-localization/README.md) | ActivityNet, HACS Segment, FineAction, THUMOS-14 |
| [Temporal Action Segmentation](./video-mamba-suite/temporal-action-segmentation/README.MD) | GTEA, Breakfast, 50salads                        |
| [Video dense captioning](./video-mamba-suite/video-dense-captioning/README.md)             | ActivityNet, YouCook2                            |
| [Video paragraph captioning](./video-mamba-suite/video-dense-captioning/README.md)         | ActivityNet, YouCook2                            |
| [Action Anticipation](./video-mamba-suite/action-anticipation/README.md)                   | Epic-kitchen-100                                 |


### Mamba for Cross-modal Interaction
| task                                                                               | supported datasets       |
| :--------------------------------------------------------------------------------- | :----------------------- |
| [Video Temporal Grounding](./video-mamba-suite/video-temporal-grounding/README.md) | QvHighlight, Charade-STA |
| [Highlight Detection](./video-mamba-suite/video-temporal-grounding/README.md)      | QvHighlight              |



### Mamba as Video Temporal Adapter

| task                                                                                          | supported datasets |
| :-------------------------------------------------------------------------------------------- | :----------------- |
| [Zero-shot/Finetuned Multi-instance Retrieval](./video-mamba-suite/egocentric-understanding/) | Epic-kitchen-100   |
| [Finetuned Action Recognition](./video-mamba-suite/egocentric-understanding/)                 | Epic-kitchen-100   |
| [Long-form Video Question-Answer](./video-mamba-suite/egocentric-understanding/)              | EgoSchema          |


### Mamba for Spatial-temporal Modeling

| task                                                                                          | supported datasets |
| :-------------------------------------------------------------------------------------------- | :----------------- |
| [Zero-shot/Finetuned Multi-instance Retrieval](./video-mamba-suite/egocentric-understanding/) | Epic-kitchen-100   |
| [Finetuned Action Recognition](./video-mamba-suite/action-recognition/)                       | Kinetics-400       |



## Model Zoo

The models of [Mamba as Video Temporal Adapter](#mamba-as-video-temporal-adapter) and [Mamba for Spatial-temporal Modeling](#mamba-for-spatial-temporal-modeling).


| Model                   | Pretrain-I | Pretrain-V | Finetuned | #Frame | Weight                                                                                                                            |
| ----------------------- | ---------- | ---------- | --------- | ------ | --------------------------------------------------------------------------------------------------------------------------------- |
| TimeSformer-B (Vanilla) | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_vanilla_base_bs512_f4.pt)                  |
| TimeSformer-B (Frozen)  | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_frozen_base_bs512_f4.pt)                   |
| TimeMamba-B (Vanilla)   | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_vanilla_base_bs512_f4.pt)                    |
| TimeMamba-B (Frozen)    | CLIP-400M  | Ego4D-4M   | -         | 4      | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_frozen_base_bs512_f4.pt)                     |
| TimeSformer-B (Vanilla) | CLIP-400M  | Ego4D-4M   | EK100-CLS | 4->16  | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_vanilla_base_bs512_f4_ft_ek100_cls_f16.pt) |
| TimeSformer-B (Vanilla) | CLIP-400M  | Ego4D-4M   | EK100-MIR | 4->16  | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timesformer_vanilla_base_bs512_f4_ft_ek100_mir_f16.pt) |
| TimeMamba-B (Vanilla)   | CLIP-400M  | Ego4D-4M   | EK100-CLS | 4->16  | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_vanilla_base_bs512_f4_ft_ek100_cls_f16.pt)   |
| TimeMamba-B (Vanilla)   | CLIP-400M  | Ego4D-4M   | EK100-MIR | 4->16  | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/clip_timemamba_vanilla_base_bs512_f4_ft_ek100_mir_f16.pt)   |
| ViT-T                   | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vit_tiny_bs512_f16.pt)                                 |
| ViT-S                   | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vit_small_bs512_f16.pt)                                |
| ViViM-T                 | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vivim_tiny_bs512_f16.pt)                               |
| ViViM-S                 | deit,IN1K  | Ego4D-4M   | -         | 16     | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vit_small_bs512_f16.pt)                                |
| ViViM-T                 | deit,IN1K  | -          | K400      | 16x3x4 | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vivim_tiny_k400_f16.pt)                                |
| ViViM-S                 | deit,IN1K  | -          | K400      | 16x3x4 | 🤗 [HF link](https://huggingface.co/cg1177/video-mamba-suite/blob/main/deit_vivim_small_k400_f16.pt)                               |


## Cite

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{2024videomambasuite,
      title={Video Mamba Suite: State Space Model as a Versatile Alternative for Video Understanding}, 
      author={Guo Chen, Yifei Huang, Jilan Xu, Baoqi Pei, Zhe Chen, Zhiqi Li, Jiahao Wang, Kunchang Li, Tong Lu, Limin Wang},
      year={2024},
      eprint={2403.09626},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


## License

This project is released under the [MIT License](./LICENSE)

## Acknowledgement

This repository is built based on [ActionFormer](https://github.com/happyharrycn/actionformer_release), [UniVTG](https://github.com/showlab/UniVTG), [ASFormer](https://github.com/ChinaYi/ASFormer), [PDVC](https://github.com/ttengwang/PDVC), [Testra](https://github.com/zhaoyue-zephyrus/TeSTra), [MAT](https://github.com/Echo0125/Memory-and-Anticipation-Transformer), [AVION](https://github.com/zhaoyue-zephyrus/AVION), [InternVideo](https://github.com/OpenGVLab/InternVideo), [EgoSchema](https://github.com/egoschema/EgoSchema), [ViM](https://github.com/hustvl/Vim) and [Mamba](https://github.com/state-spaces/mamba) repository.
