
---

# HTR-LTR: Multi-Temporal Feature Integration for Gait Recognition

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/11433429)
[![Framework](https://img.shields.io/badge/Framework-OpenGait-orange)](https://github.com/ShiqiYu/OpenGait)
[![License](https://img.shields.io/badge/License-Academic%20Use-green)](#license)

Official implementation of:

**Multi-Temporal Feature Integration: Learning Complementary Motion and Semantic Representations for Gait Recognition**
Published in *IEEE Access, 2026*

🔗 Paper:
[https://ieeexplore.ieee.org/document/11433429](https://ieeexplore.ieee.org/document/11433429)

---

## Overview

This repository presents **HTR-LTR**, a multi-temporal feature integration framework designed for robust gait recognition. The method jointly learns:

* **Motion dynamics**
* **Semantic representations**

By integrating these complementary features, HTR-LTR achieves improved recognition performance under challenging conditions.

The implementation is built upon the **OpenGait** framework.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/MDROKONMIA/HTR_LTR.git
cd HTR_LTR
```

---

### 2. Install Dependencies

#### Required Packages

* PyTorch ≥ 1.10
* torchvision
* pyyaml
* tensorboard
* opencv-python
* tqdm
* py7zr
* kornia
* einops

---

### Option A: Install with Conda (Recommended)

```bash
conda install tqdm pyyaml tensorboard opencv kornia einops -c conda-forge
conda install pytorch==1.10 torchvision -c pytorch
```

---

### Option B: Install with pip

```bash
pip install tqdm pyyaml tensorboard opencv-python kornia einops
pip install torch==1.10 torchvision==0.11
```

---

## Dataset Preparation

Follow the instructions in:

```
2.prepare_dataset.md
```

Ensure datasets are formatted according to the **OpenGait** structure before training.

**Note:** For the CCPG dataset, multiple sequences of the same subject captured by the same camera were merged into a single sequence.

---

## Training

To train the HTR-LTR model:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
--nproc_per_node=1 \
--master-port=29501 \
opengait/main.py \
--cfgs ./configs/HTR_LTR/CASIAB.yaml \
--phase train
```

### Arguments

| Argument             | Description                   |
| -------------------- | ----------------------------- |
| CUDA_VISIBLE_DEVICES | GPU ID(s) to use              |
| --nproc_per_node     | Number of GPUs                |
| --master-port        | Port for distributed training |
| --cfgs               | Path to config file           |
| --phase train        | Training mode                 |

**Optional:**

```bash
--log_to_file
```

Saves terminal logs to a file.

---

## Testing

Download pretrained checkpoints:
[https://drive.google.com/file/d/1-HxQPEKeo1oMRknaZRvQ5-jzS3wZET_Y/view?usp=sharing](https://drive.google.com/file/d/1-HxQPEKeo1oMRknaZRvQ5-jzS3wZET_Y/view?usp=sharing)

Run evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
--nproc_per_node=1 \
--master-port=29501 \
opengait/main.py \
--cfgs ./configs/HTR_LTR/CASIAB.yaml \
--phase test
```

**Optional:**

```bash
--iter <checkpoint_iteration>
```

Example scripts are available in:

```
test.sh
```

---

## ⚠️ Notes on Distributed Training

When using Distributed Data Parallel (DDP), abnormal termination may leave residual (zombie) processes.

Clean them with:

```bash
sh misc/clean_process.sh
```

---

## Citation

If you use this work, please cite:

### HTR-LTR

```bibtex
@article{Mia2026,
  title={Multi-temporal Feature Integration: Learning Complementary Motion and Semantic Representations for Gait Recognition},
  author={Mia, Md. Rokon and Uddin, Md. Zasim and Alnajjar, Fady and Swavaf, Muhammed and Ahad, Md Atiqur Rahman},
  journal={IEEE Access},
  year={2026},
  doi={10.1109/ACCESS.2026.3672992}
}
```

---

### OpenGait

```bibtex
@inproceedings{Fan_2023_CVPR,
  author={Fan, Chao and Liang, Junhao and Shen, Chuanfu and Hou, Saihui and Huang, Yongzhen and Yu, Shiqi},
  title={OpenGait: Revisiting Gait Recognition Towards Better Practicality},
  booktitle={CVPR},
  year={2023},
  pages={9707--9716}
}
```

```bibtex
@article{fan2025opengait,
  title={OpenGait: A Comprehensive Benchmark Study for Gait Recognition Towards Better Practicality},
  author={Fan, Chao and Hou, Saihui and Liang, Junhao and Shen, Chuanfu and Ma, Jingzhe and Jin, Dongyang and Huang, Yongzhen and Yu, Shiqi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025}
}
```

---

## Acknowledgements

This work builds upon the **OpenGait** framework developed by researchers from the Chinese Academy of Sciences.

---

## License

This project is intended **for academic research purposes only**.
**Commercial use is not permitted.**

---
