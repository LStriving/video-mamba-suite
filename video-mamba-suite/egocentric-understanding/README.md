# Video Mamba Suite: Egocentric Understanding


## Installation

See [INSTALL.md](docs/INSTALL.md) to install this code.


## Main results

### 1. Zero-shot Multi-instance Retrieval


#### 1.1 Video Temporal Adapter

| Method                |  V2T mAP |  T2V mAP |  Avg mAP | V2T nDCG | T2V nDCG | Avg nDCG |
|-----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| TimeSformer (Vanilla) |   29.2   |   21.8   |   25.5   |   30.1   |   27.1   |   28.6   |
| TimeSformer (Frozen)  |   29.8   |   22.2   |   26.0   |   30.6   |   27.5   |   29.0   |
| TimeMamba (Vanilla, Ours)   |   30.3   |   22.1   |   26.2   |   30.9   |   27.5   |   29.2   |
| TimeMamba (Frozen, Ours)    | **30.7** | **22.8** | **26.8** | **31.3** | **27.8** | **29.5** |

#### 1.2 Spatial-Temporal Modeling

| Method  | #F |  V2T mAP  |  T2V mAP  |  Avg mAP  |  V2T nDCG |  T2V nDCG |  Avg nDCG |
|---------|----|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| ViT-T   | 4  |   15.50   |   11.10   |   13.30   |   22.48   |   19.66   |   21.07   |
| ViT-B   | 4  |   25.08   |   18.49   |   21.79   |   27.80   |   24.87   |   26.34   |
| ViT-T   | 16 |   20.47   |   15.29   |   17.88   |   25.74   |   22.89   |   24.31   |
| ViT-S   | 16 |   23.80   |   17.60   |   20.70   |   27.40   |   24.40   |   25.90   |
| ViViM-T (Ours) | 16 |   23.31   |   17.21   |   20.26   |   27.40   |   24.30   |   25.80   |
| ViViM-S (Ours) | 16 | **26.00** | **19.60** | **22.80** | **28.20** | **25.30** | **26.70** |

### 2. Long-term Video Question-Answer on EgoSchema


#### 2.1 Video Temporal Adapter

| Method                | #Frame | Frame Sampling | Acc (full set) |
|-----------------------|:--------:|:--------:|:--------:|
| TimeSformer (Vanilla) | 16 |  Uniform |   38.52  |  
| TimeMamba (Ours) | 16 |  Uniform |   38.70     | 



### 3. Finetuned Multi-instance Retrieval

| Method                |  V2T mAP |  T2V mAP |  Avg mAP | V2T nDCG | T2V nDCG | Avg nDCG |
|-----------------------|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| TimeSformer (Vanilla) |   52.2  |   44.2   |   48.2   |   64.0   |   61.5   |   62.7   |
| TimeMamba (Ours)   |   **52.4**   |   **45.4**   |   **48.9**   |   **65.9**   |   **63.3**   |   **64.6**   |

### 4. Finetuned Action Recognition
| Method                | Verb Top1 | Noun Top1 | Action Top1 | Action Top5 |
|-----------------------|:---------:|:---------:|:-----------:|:-----------:|
| TimeSformer (Vanilla) |    65.2   |    55.0   |     44.5    |     62.4    |
| TimeMamba (Ours)   |  **68.5** |  **55.8** |   **46.1**  |   **63.8**  |


For more details, please refer to [MODEL_ZOO](./docs/MODEL_ZOO.md).




## Acknowledgement

The codebase is based on [AVION](https://github.com/zhaoyue-zephyrus/AVION).
We thanks the authors for their efforts.

