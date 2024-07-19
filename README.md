# Few-Shot Domain Adaptive Object Detection for Microscopic Imagest - MICCAI-2024
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Sumayya Inayat](https://www.linkedin.com/in/sumayya-inayat/), [Nimra Dilawar](https://www.linkedin.com/in/nimradilawar/), [Waqas Sultani](https://www.linkedin.com/in/waqas-sultani-ph-d-3549bb60, [Mohsen Ali](https://mohsenali.github.io/)

#### **<sup>1</sup>Information Technology University (ITU) Lahore, Pakistan**

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2407.07633)
[![Project Page](https://img.shields.io/badge/IML-Project_Page-blue)](https://im.itu.edu.pk/few-shot-daodmi/)

[comment]: [![Weights](https://img.shields.io/badge/Model-Weights-87CEEB)](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/muhammad_sohail_mbzuai_ac_ae/Ei-vfphu1RVOs0Zkle8vSD8Bub6XxaPxBnwEY2j5uWCVbQ?e=cjbFIU)

## Framewrok

<p align="center">
  <img style="width: 70%;" src="tool/ProposedFrameWork_.png" alt="Framework">
</p>

## Motivation
Few-shot domain adaptive object detection (FSDAOD) addresses the challenge of adapting object detectors to target domains with limited labeled data. We propose a novel FSDAOD strategy for microscopic imaging. Our contributions include: 1) a domain adaptive class balancing strategy for few shot scenarios; 2) multi-layer instance-level inter and intra-domain alignment by enhancing similarity between the instances of classes regardless of the domain and enhancing dissimilarity when it’s not. Furthermore, an instance-level classification loss is applied in the middle layers of the object detector to enforce the retention of features necessary for the correct classification regardless of the domain. Extensive experimental results with competitive baselines indicate the effectiveness of our proposed framework by achieving state-of-the-art results on two public microscopic datasets.

### Installation
Coming Soon

### Datasets
Coming Soon

### Training
Coming Soon

### Evaluation
Coming Soon

## 👁️💬 Augmentation
<p align="center">
  <img style="width: 60%;"src="tool/Augmentation_.png">
</p>

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





























<div id="preview1" class="g-b g-b--t1of2 split split-preview" style="height: 538px;">
<div id="preview" class="preview-html" preview="" debounce="150"><h1 class="code-line" data-line-start="0" data-line-end="1"><a id="FewShot_Domain_Adaptive_Object_Detection_for_Microscopic_Images_0"></a>Few-Shot Domain Adaptive Object Detection for Microscopic Images</h1>
<p class="has-line-data" data-line-start="2" data-line-end="3"><a href="https://pytorch.org/"><img src="./Miccai_Readme file_files/Framework-PyTorch-orange.svg" alt="Framework: PyTorch"></a></p>
<h4 class="code-line" data-line-start="4" data-line-end="5"><a id="Contributions_4"></a>Contributions</h4>
<ul>
<li class="has-line-data" data-line-start="5" data-line-end="6">We propose a novel FDAOD strategy for microscopic imaging.</li>
<li class="has-line-data" data-line-start="6" data-line-end="7">We propose a domain adaptive class balancing cut paste (CBCP) strategy for few shot scenario; multi-layer instance-level inter and intra-domain alignment by enhancing similarity between the instances of classes regardless of the domain and enhance dissimilarity when it’s not.</li>
<li class="has-line-data" data-line-start="7" data-line-end="9">We propose Intra-Inter-Domain Feature Alignment technique; I2DA, that addresses (a) the domain shift between similar class cells by aligning the inter-domain feature level representations of cells coming from same classes, and (b) Intra-Domain Feature Consistency at the cell level to learn distinguishable features for each class because the foreground cells in microscopic datasets possess high visual similarity with the background cells.</li>
</ul>
<h2 class="code-line" data-line-start="9" data-line-end="10"><a id="Contents_9"></a>Contents</h2>
<ol>
<li class="has-line-data" data-line-start="10" data-line-end="11"><a href="https://dillinger.io/#installation-instructions">Installation Instructions</a></li>
<li class="has-line-data" data-line-start="11" data-line-end="12"><a href="https://dillinger.io/#dataset-preparation">Dataset Preparation</a></li>
<li class="has-line-data" data-line-start="12" data-line-end="13"><a href="https://dillinger.io/#training">Training</a></li>
<li class="has-line-data" data-line-start="13" data-line-end="14"><a href="https://dillinger.io/#evaluation">Evaluation</a></li>
<li class="has-line-data" data-line-start="14" data-line-end="15"><a href="https://dillinger.io/#citation">Citation</a></li>
</ol>
<h2 class="code-line" data-line-start="17" data-line-end="18"><a id="Installation_Instructions_17"></a>Installation Instructions</h2>
<ul>
<li class="has-line-data" data-line-start="18" data-line-end="19">We use Python 3.7, CUDA &gt;= 11.0</li>
</ul>
<pre><code class="has-line-data" data-line-start="20" data-line-end="24">conda create -n i2da python=3.7
conda activate i2da
pip install requirements.txt
</code></pre>
<h2 class="code-line" data-line-start="24" data-line-end="25"><a id="Dataset_Preparation_24"></a>Dataset Preparation</h2>
<ul>
<li class="has-line-data" data-line-start="26" data-line-end="27"><strong>M5-Malaria-Dataset</strong>: Download the dataset from <a href="https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022">Link</a>.</li>
<li class="has-line-data" data-line-start="27" data-line-end="29"><strong>Raabin-WBC Dataset</strong>: Download the dataset from <a href="https://raabindata.com/">Link</a>.</li>
</ul>
<p class="has-line-data" data-line-start="29" data-line-end="32">Convert the <a href="https://github.com/intelligentMachines-ITU/LowCostMalariaDetection_CVPR_2022">malaria data</a> into YOLOFormat format using  <code>preprocessing/full_m5_To_yolo.py</code><br>
Convert the <a href="https://raabindata.com/">raabin data</a> into YOLOFormat format using  <code>preprocessing/orig_raabin2yolo.py</code>.<br>
Datasets are stored in the <code>./datasets</code> folder like following structure:</p>
<pre><code class="has-line-data" data-line-start="33" data-line-end="52">- datasets/
 - m5-malaria/
  - images/
    - <span class="hljs-built_in">test</span>/
    - train/
      - <span class="hljs-number">1000</span>x/
         Malaria_CM1_21Jun2021101548_0001_127.<span class="hljs-number">9</span>_10.<span class="hljs-number">9</span>_1000x.png
         .
         .
    - val/
  - labels/
    - <span class="hljs-built_in">test</span>/
    - train/
      - <span class="hljs-number">1000</span>x/
         Malaria_CM1_21Jun2021101548_0001_127.<span class="hljs-number">9</span>_10.<span class="hljs-number">9</span>_1000x.txt
         .
         .
   - val/
</code></pre>
<ul>
<li class="has-line-data" data-line-start="52" data-line-end="53">Data Generation with Class Balancing cut Paste - Augmentation</li>
</ul>
<pre><code class="has-line-data" data-line-start="54" data-line-end="56">
</code></pre>
<h3 class="code-line" data-line-start="56" data-line-end="57"><a id="Training_56"></a>Training</h3>
<p class="has-line-data" data-line-start="57" data-line-end="58">Add paths to your dataset in <code>data/m5.yaml</code></p>
<ul>
<li class="has-line-data" data-line-start="58" data-line-end="59">Train model for only Domian Allignment</li>
</ul>
<pre><code class="has-line-data" data-line-start="60" data-line-end="62">train_contrat.py --cfg ./models/yolov5x.yaml \ --hyp ./data/hyp.finetune.yaml \ --epoch 100 --batch 4 --data ./data/m5.yaml
</code></pre>
<ul>
<li class="has-line-data" data-line-start="62" data-line-end="63">For Complete Training</li>
</ul>
<pre><code class="has-line-data" data-line-start="64" data-line-end="66">train.py --cfg ./models/yolov5x.yaml \ --hyp ./data/hyp.finetune.yaml \ --epoch 100 --batch 4 --data ./data/m5.yaml
</code></pre>
<h3 class="code-line" data-line-start="66" data-line-end="67"><a id="Evaluation_66"></a>Evaluation</h3>
<pre><code class="has-line-data" data-line-start="68" data-line-end="70">test.py --weights ./runs/train/exp2/weights  \ --batch-size 2 --task test --data ./data/m5.yaml
</code></pre>
<h2 class="code-line" data-line-start="71" data-line-end="72"><a id="Citation_71"></a>Citation</h2>
<p class="has-line-data" data-line-start="73" data-line-end="74">If you found I2DA FSDAOD useful in your research, please consider starring ⭐ us on GitHub and citing 📚 us in your research!</p>
</div>
</div>
</div>
</div>
</div>

</body></html>
