# ChatENV: An Interactive Vision-Language Model for Sensor-Guided Environmental Monitoring and Scenario Simulation 🌍

This is the code base for the ChatENV project. It includes finetuning the Qwen 2.5 VL model to describe images, and comment on changes between image pairs, in addition to asking hypothetical "what-if" questions regarding weather information and sensor data. Additionally, the repo includes steps to finetune video-based llava models for better temporal analysis, using Video-LLaVA and LLaVA-NeXT-Video.

#### [Hosam Elgendy](https://scholar.google.com/citations?user=6RA4_m8AAAAJ&hl=en&oi=ao), [Ahmed Sharshar](https://scholar.google.com/citations?user=GC8A9k0AAAAJ&hl=en), [Ahmed Aboeitta](https://scholar.google.com/citations?user=sEZTgaYAAAAJ&hl=en&oi=ao) and [Mohsen Guizani](https://scholar.google.com/citations?user=RigrYkcAAAAJ&hl=en&oi=ao)
#### Mohamed bin Zayed University of AI (MBZUAI)

---
<p align='center'>
<img src="assets/qwen_model.jpg" height="400">
</p>

---

## Contents
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
- [Directoy Setup](#direcotry-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

---

## Environment Setup

1. Clone this repository:
    ```shell
    git clone https://github.com/HosamGen/ChatENV.git
    cd ChatENV
    ```
> [!NOTE]
> This repo consists of the steps to finetune the Qwen 2.5 Model (named ChatENV), in addition to the setup to finetune Video-LLaVA and LLaVA-NeXT-Video (named videollava). The setup is split based on which model(s) is used.

2. Install the necessary dependencies:
    ```shell
    conda create -n chatenv python=3.10
    conda activate chatenv
    pip install -r chatenv_requirements.txt
    ```
2. [Optional] Setup for video models.
   ```shell
    conda create -n videollava python=3.10
    conda activate videollava
    pip install -r llava_requirements.txt
    ```

---
## Dataset
### ChatENV Custom Dataset

1. fMoW Images:

Please refer to the [fMoW dataset](https://github.com/fMoW/dataset?tab=readme-ov-file) for the original remote sensing dataset. Image files are needed for the ChatENV (Qwen based) model finetuning. We provide the cleaned annotations/QAs in the [Annotations]() section below.

2. Dataset for finetuning the Qwen model in the Three-Turn Setting, Sheet with Questions and Answers for the What-If finetuning, and Sheets with emissions data can be downloaded from: [ChatENV](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hosam_elgendy_mbzuai_ac_ae/ElUQBEmS821KsHf9WkisV4wBYNWru3K-gb2Lp7XNYsBrXQ?e=xVA7NS).

3. [OPTIONAL] For the video models, the annotationsare too large, and can be downloaded via [Videollava-Annotations](https://mbzuaiac-my.sharepoint.com/:f:/g/personal/hosam_elgendy_mbzuai_ac_ae/ErvrSn_bdfJOkr8VOoF2oaIBRRmDECYP6_SFnBS_NAR6dw?e=RpEijA).
4. [OPTIONAL] For the video models, the videos of combined images can be downloaded as a zip file through: [Videollava-Videos](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/hosam_elgendy_mbzuai_ac_ae/EcRuKZwN2y5AlNU3PTc36goBNfhlOdxtcWcZ35ZhiYFDXA?e=OIL0S2). These videos can be unzipped as follows:

```shell
unzip chatenv_videos.zip

|    ├── chatenv_train_videos/
|    ├── chatenv_val_videos/
```

## Direcotry Setup
The setup for the ChatENV directory should look like this:
```
ChatENV
├── annotations/
|    ├── chatgpt_train_annotations.json
|    ├── gemini_train_annotations.json
|    |   .....
├── chatenv_train_videos/
|    ├── xxx.mp4
|    |   .....
├── chatenv_val_videos/
|    ├── xx.mp4
|    |   .....
├── llavanext_eval.py
├── llavanext_finetune.py
├── qwen/
├── videollava_finetune.py
├── videollava_test.py
...
```
The llavanext/videollava files are for the video based models, whereas `qwen/` contains the scripts for finetuing the qwen model.

The `qwen/` directory should look like:

```
ChatENV
├── qwen/
|    ├──chatgpt_eval_data
|    ├──chatgpt_train_data
|    ├──gemini_train_data
|    ├──gemini_eval_data
|    ├──...
|    ├──run_qwen_vl.py
|    ├──emissions_chatgpt.csv
|    ├──emissions_gemini.csv
|    ├──emissions_merge.csv

```
## Training

For the Qwen model finetuning, a single script `run_qwen_vl.py` does finetuning, zero-shot evaluation and fine-tuning evaluation. Simply use:

```shell
python run_qwen_vl.py --XX --YY
```

As for the video models, run the `videollava_finetune.py` or `llavanext_finetune.py` scripts, change the flags based on your configuration.

Flags:
```shell
--lora
--qlora
--4bit
--model
--batch_size
..
```

For Video-LLaVA:
```shell
python videollava_finetune.py
```

For LLaVA-NeXT:
```shell
python llavanext_finetune.py
```

Modify parameters such as:
```shell
MAX_LENGTH = 256
USE_LORA = False
USE_QLORA = True 
USE_8BIT = False 
PRUNE = False 
prune_amount = 0.05 
MODEL_TYPE = "sample" #for 10k sample dataset
# MODEL_TYPE = "full" #for the full 100k dataset
batch_size = 2

#lora parameters
lora_r = 64
lora_alpha = 128
```

## Evaluation

For the Qwen model finetuning, the same script `run_qwen_vl.py` is used:
Zero-Shot Evaluation only:
```shell
FLAAAAG```

Finetuning Evaluation only:
```shell
FLAAAAG
```

Comapring Zero-Shot vs Finetuning:
```shell
FLAAAAG
```


To evaluate the fine-tuned models on the test dataset, use the following commands:

For Video-LLaVA:
```shell
python videollava_test.py
```

For LLaVA-NeXT:
```shell
python llavanext_eval.py
```

> [!IMPORTANT]
> The MODEL_PATH must be changed during evaluation based on the model that was finetuned.

These commands will run the evaluation on the specified test dataset and generate performance metrics, including ROUGE, BLEU, and BERT scores. The results will help assess the model's performance in detecting temporal changes in remote sensing data.

## Results

We evaluated the performance of GeoLLaVA across various metrics, including ROUGE, BLEU, and BERT scores. The fine-tuned model demonstrated significant improvements in capturing and describing temporal changes in geographical landscapes.

To calculate the scores after evaluating the models, please check the steps in the [Results.ipynb](https://github.com/HosamGen/GeoLLaVA/blob/main/Results.ipynb) notebook.

## Video-LLaVA Results

| Model                | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | BERT  |
|----------------------|---------|---------|---------|-------|-------|
| **Base**             | 0.211   | 0.041   | 0.122   | 0.039 | 0.456 |
| **10K LoRA**         | 0.563   | 0.214   | 0.313   | 0.243 | 0.849 |
| **100K LoRA**        | **0.576**   | **0.226**   | **0.325**   | **0.250** | **0.863** |
| **10K QLoRA**        | 0.565   | 0.212   | 0.310   | 0.243 | 0.845 |
| **100K QLoRA**       | 0.571   | 0.220   | 0.316   | **0.250** | 0.854 |
| **10K Pruning 5%**   | 0.031   | 0.007   | 0.024   | 0.010 | 0.265 |
| **100K Pruning 5%**  | 0.125   | 0.034   | 0.110   | 0.043 | 0.359 |

## LLaVA-NeXT Results

| Model                | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | BERT  |
|----------------------|---------|---------|---------|-------|-------|
| **Base**             | 0.197   | 0.037   | 0.113   | 0.042 | 0.404 |
| **10K LoRA**         | 0.554   | 0.198   | 0.300   | 0.232 | 0.856 |
| **100K LoRA**        | **0.562**   | 0.199   | 0.300   | **0.239** | **0.864** |
| **10K QLoRA**        | 0.543   | 0.193   | 0.283   | 0.213 | 0.836 |
| **100K QLoRA**       | 0.561   | **0.202**   | **0.302**   | 0.229 | 0.858 |
| **10K Pruning 5%**   | 0.532   | 0.178   | 0.278   | 0.209 | 0.829 |
| **100K Pruning 5%**  | 0.541   | 0.183   | 0.284   | 0.210 | 0.840 |

**Final Model (100K LoRA)** | **0.556** | **0.202** | **0.290** | **0.227** | **0.850** |



These metrics illustrate how well the models performed in describing temporal changes in remote sensing data, with fine-tuning techniques like LoRA and QLoRA leading to notable improvements.

## Acknowledgement
+ [Qwen-2.5VL](https://github.com/QwenLM/Qwen2.5-VL) Original repo for the Qwen2.5-VL model.
+ [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/) Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. We have used Video-LLaVA as one of the models for finetuning.
+ [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) LLaVA-NeXT: Open Large Multimodal Models. The video model was used as the second model.
+ [fMoW RGB Dataset](https://github.com/fMoW/dataset) Original fMoW dataset repo.

## Citation
please cite using this BibTeX:
```bibtex

```


