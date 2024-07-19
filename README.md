# Few-Shot Domain Adaptive Object Detection for Microscopic Imagest - MICCAI-2024
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Sumayya Inayat](https://www.linkedin.com/in/sumayya-inayat/), [Nimra Dilawar](https://www.linkedin.com/in/nimradilawar/), [Waqas Sultani](https://www.linkedin.com/in/waqas-sultani-ph-d-3549bb60), [Mohsen Ali](https://mohsenali.github.io/)

#### **Information Technology University (ITU) Lahore, Pakistan**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.07633)
[![Project Page](https://img.shields.io/badge/IML-Project_Page-blue)](https://im.itu.edu.pk/few-shot-daodmi/)

[comment]: [![Weights](https://img.shields.io/badge/Model-Weights-87CEEB)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muhammad_sohail_mbzuai_ac_ae/Ei-vfphu1RVOs0Zkle8vSD8Bub6XxaPxBnwEY2j5uWCVbQ?e=cjbFIU)

## Framewrok

<p align="center">
  <img style="width: 70%;" src="tool/ProposedFrameWork_.png" alt="Framework">
</p>

## Motivation
Few-shot domain adaptive object detection (FSDAOD) addresses the challenge of adapting object detectors to target domains with limited labeled data. We propose a novel FSDAOD strategy for microscopic imaging. Our contributions include: 1) a domain adaptive class balancing strategy for few shot scenarios; 2) multi-layer instance-level inter and intra-domain alignment by enhancing similarity between the instances of classes regardless of the domain and enhancing dissimilarity when it’s not. Furthermore, an instance-level classification loss is applied in the middle layers of the object detector to enforce the retention of features necessary for the correct classification regardless of the domain. Extensive experimental results with competitive baselines indicate the effectiveness of our proposed framework by achieving state-of-the-art results on two public microscopic datasets.

## Contents
1. [Installation Instructions](#Installation-instruction)
2. [Dataset Preparation](#Dataset-Preparation)
3. [Dataset Generation](##Dataset-Generation)
4. [Training](#Training)
5. [Evaluation](#Evaluation)
6. [Quantitative Results](#Quantitative-Results)
7. [Qualitative Results](#Qualitative-Results)
8. [Citation](#Citation)


## Installation instruction
- We use Python 3.7, CUDA >= 11.0
- You can also create enviornment using `pip install requirements.txt`

```bash
conda create -n i2da python=3.7
conda activate i2da
pip install requirements.txt
```

## Dataset Preparation
- M5-Malaria-Dataset: Download the dataset from [Link](https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022).
- Raabin-WBC Dataset: Download the dataset from [Link](https://raabindata.com/).

- Convert the malaria data into YOLOFormat format using <kbd>preprocessing/full_m5_To_yolo.py</kbd>
- Convert the raabin data into YOLOFormat format using <kbd>preprocessing/orig_raabin2yolo.py</kbd>
- Datasets are stored in the <kbd>./datasets</kbd> folder like following structure:


- datasets/
  - m5-malaria/
    - images/
      - test/
      - train/
        - 1000x/
          - Malaria_CM1_21Jun2021101548_0001_127.9_10.9_1000x.png
          - ...
      - val/
    - labels/
      - test/
      - train/
        - 1000x/
          - Malaria_CM1_21Jun2021101548_0001_127.9_10.9_1000x.txt
          - ...
      - val/

## 👁️💬 Dataset Generation
- Data Generation with Class Balancing Cut Paste Augmentation

<p align="center">
  <img style="width: 60%;"src="tool/Augmentation_.png">
</p>

- Following will generate a folder for malaria <kbd>HCM_tar_aug</kbd> and raabin <kbd>Rabin_tar_aug</kbd>
- <kbd>preprocessing/cut_mix_m5.py</kbd>


## Training
Add path of folder containing augmented dataset in <kbd>data/m5.yaml</kbd>
- For Complete Training
```bash
train.py --cfg ./models/yolov5x.yaml \ --hyp ./data/hyp.finetune.yaml \ --epoch 100 --batch 4 --data ./data/m5.yaml

```
- Train model for only Domain Alignment
```bash
train_contrat.py --cfg ./models/yolov5x.yaml \ --hyp ./data/hyp.finetune.yaml \ --epoch 100 --batch 4 --data ./data/m5.yaml

```
## Evaluation
```bash
test.py --weights ./runs/train/exp2/weights  \ --batch-size 2 --task test --data ./data/m5.yaml

```

## 🔍 Quantitative Results
<p align="center">
  <img style="width: 60%;"src="tool/Malaria_HCM2LCM_test.png">
</p>
<p align="center">
  <img style="width: 60%;" src="tool/Rabin_WBC_HCM2LCM_test.png">
</p>


## 📊 Qualitative Results
<p align="center">
  <img style="width: 60%;" src="tool/Qualitative_Results.png">
</p>


## 📜 Citation
```bibtex
@article{inayat2024few,
    title={Few-Shot Domain Adaptive Object Detection for Microscopic Images},
    author={Inayat, Sumayya and Dilawar, Nimra and Sultani, Waqas and Ali, Mohsen},
    journal={arXiv preprint arXiv:2407.07633},
    year={2024}
}
```

## 📜 Acknowledgement









