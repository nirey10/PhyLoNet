# PhyLoNet
This is the source code of our paper:

PhyLoNet: Physically-Constrained Long Term Video Prediction

ACCV 2022

Nir Ben Zikri & Andrei Sharf

## Abstract
Motions in videos are often governed by physical and biological laws such as gravity, collisions, flocking, etc. Accounting for such natural properties is an appealing way to improve realism in future frame video prediction. Nevertheless, the definition and computation of intricate physical and biological properties in motion videos is challenging.
In this work we introduce PhyLoNet, a PhyDNet extension which learns long term future frame prediction and manipulation. Similar to PhyDNet, our network consists of a two-branch deep architecture which explicitly disentangles physical dynamics from complementary information. It uses a recurrent physical cell (PhyCell) for performing physically-constrained prediction in latent space.
In contrast to PhyDNet, PhyLoNet introduces a modified encoder-decoder architecture together with a novel relative flow loss. This enables a longer term future frame prediction from a small input sequence with higher accuracy and quality.
We have carried extensive experiments, showing the ability
of PhyLoNet to outperform PhyDNet on various challenging natural motion datasets such as ball collisions, flocking and pool game. Ablation studies highlight the importance of our new components. Finally, we show application of PhyLoNet to video manipulation and editing by a novel class label modification architecture.

![PhyDNet](https://user-images.githubusercontent.com/15626875/192331986-9ba98ddc-88a5-48fb-a553-f75ccee41a5d.png)
![testNoInput](https://user-images.githubusercontent.com/15626875/192332814-cb755856-56e2-493b-88fe-27e0bdcdfdc7.png)
![trajectory_combined](https://user-images.githubusercontent.com/15626875/192332944-863c68ef-c6a2-4d8d-a7dc-c5a28e4783dd.png)

## Run
This code has been tested on pytorch 1.5.1 and Cuda 9.1. The GPU used for training is RTX 2080.

## Datasets:
Our dataset compose of train and validation folders where each sample consists of 30 video frames.
Our full datasets might be available in the future.
