# VideoWorld: Exploring Knowledge Learning from Unlabeled Videos
> #### Zhongwei Ren, Yunchao Wei<sup>&dagger;</sup>, Xun Guo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin<sup>&dagger;</sup><sup>&ddagger;</sup>
> <sup>&dagger;</sup>Correspondence, <sup>&ddagger;</sup>Project Lead

> Beijing Jiaotong University, University of Science and Technology of China, ByteDance Seed

<font size=7><div align='center' > <a href='https://arxiv.org/pdf/2501.09781'>**Paper**</a> | <a href="https://maverickren.github.io/VideoWorld.github.io">**Project Page**</a> | [**Installation**](#Install) | [**Training**](#training) | [**Inference**](#inference)  | <a href="https://huggingface.co/datasets/maverickrzw/VideoGo-Bench">**Video-GoBench**</a></div></font>

<img width="1000" alt="image" src='figs/figure1.png'>


## :fire: News
* **[2025.1]** We release the code and dataset.

# Highlight

👉 We explore, for the first time, whether video generation models can learn knowledge and observe two key findings: i) merely observing videos suffices to learn complex tasks, and ii) compact representations of visual changes greatly enhance knowledge learning.

👉 We propose VideoWorld, leveraging a latent dynamics model to represent multi-step visual changes, boosting both efficiency and effectiveness of knowledge acquisition.

👉 We construct Video-GoBench, a large-scale video-based Go dataset for training and evaluation, facilitating future research on knowledge learning from pure videos.

# Introduction
This work explores whether a deep generative model can learn complex knowledge solely from visual input, in contrast to the prevalent focus on text-based models like large language models (LLMs). We develop \emph{VideoWorld}, an autoregressive video generation model trained on unlabeled video data, and test its knowledge acquisition abilities in video-based Go and robotic control tasks. Our experiments reveal two key findings: (1) video-only training provides sufficient information for learning knowledge, including rules, reasoning and planning capabilities, and (2) the representation of visual changes is crucial for
knowledge learning. To improve both the efficiency and efficacy of knowledge learning, we introduce the Latent Dynamics Model (LDM) as a key component of VideoWorld. Remarkably, VideoWorld reaches a 5-dan professional level in the Video-GoBench with just a 300-million-parameter model, without relying on search algorithms or reward mechanisms typical in reinforcement learning. In robotic tasks, VideoWorld effectively learns diverse control operations and generalizes across environments, approaching the performance of oracle models in CALVIN and RLBench. This study opens new avenues for knowledge acquisition from visual data, with all code, data, and models to be open-sourced for further research.

# Video
[![IMAGE ALT TEXT](./figs/ytb_new.png)](https://www.youtube.com/watch?v=y_TT4dtIPXA "VideoWorld Demo")

# Architecture

<img width="1000" alt="image" src='figs/architecture.png'>
Overview of the proposed VideoWorld model architecture. (Left) Overall architecture. (Right) The proposed latent dynamics model (LDM). First, LDM compresses the visual changes from each frame to its subsequent H frames into compact and informative latent codes. Then, an auto-regressive transformer seamlessly integrates the output of LDM with the next token prediction paradigm.


# Installation

### Setup Environment
```
conda create -n videoworld python=3.10 -y
conda activate videoworld
pip install --upgrade pip  

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```
### Install VideoWorld
```
git clone https://github.com/bytedance/VideoWorld.git
cd VideoWorld

bash install.sh
```


# Inference
### Go Battle
VideoWorld relies on the Katago Go engine. We provide scripts to facilitate battles against our model; install Katago to engage in these matches.
```
cd VideoWorld # This VideoWorld is located in a subdirectory.
bash install_katago.sh 
```
or follow the official installation instructions: https://github.com/lightvector/KataGo

We provide a version of the weights for playing against humans in https://huggingface.co/maverickrzw/VideoWorld-GoBattle. 
Use the script to start a match:
```
# Please place the weight in the path: ./VideoWorld/work_dirs/go_battle.pth
cd VideoWorld # This VideoWorld is located in a subdirectory.
bash ./tools/battle_vs_human.sh
```
### Robotics
Download CALVIN dataset follow the official instructions and organize it as follows:
```
├── VideoWorld
│   ├── VideoWorld
│   │   └── data
│   └──       └── calvin
```

Testing requires the CALVIN environment configuration. We have automated the installation of CALVIN in the install.sh script. If any issues arise, please refer to the official installation instructions: https://github.com/mees/calvin
```
cd VideoWorld # This VideoWorld is located in a subdirectory.
# Since we only tested the tasks of opening drawers, pushing 
# blocks, and switching lights, the original CALVIN test task 
# construction script needs to be replaced.
rm -r ../calvin/calvin_models/calvin_agent/evaluation/multistep_sequences.py
cp ./tools/calvin_utils/multistep_sequences.py ../calvin/calvin_models/calvin_agent/evaluation/
bash ./tools/calvin_test.sh
```

# Training
Our training consists of two stages: LDM training and autoregressive transformer training. We use the CALVIN robotic environment as an example to demonstrate how to initiate the training.
### Stage 1: LDM Training
Download CALVIN dataset follow the official instructions and organize it as follows:
```
├── VideoWorld
│   ├── LDM
│   │   └── data
│   └──       └── calvin
```
Use the script ./LDM/tools/calvin_ldm_train.sh to initiate LDM training. Upon completion, the latent codes on the training set will be automatically saved to ./LDM/work_dirs/calvin_ldm_results.pth, and the UMAP visualization of the latent codes will also be generated.
```
cd LDM 
bash ./tools/calvin_ldm_train.sh
```
### Stage 2: Next Token Prediction
We write the file path of the latent codes generated by LDM into the configuration, and then initiate the autoregressive transformer training. Please store data in the same path as inference
```
cd VideoWorld 
bash ./tools/calvin_train.sh
```









