# MAC-Net: Model Aided Nonlocal Neural Network for Hyperspectral Image Denoising
## Fengchao Xiong; Jun Zhou; Qinling Zhao; Jianfeng Lu; Yuntao Qian

[Link to paper](https://ieeexplore.ieee.org/abstract/document/9631264/)

# Abstract

Hyperspectral image (HSI) denoising is an ill-posed inverse problem. The underlying physical model is always important to tackle this problem, which is unfortunately ignored by most of the current deep learning (DL)-based methods, producing poor denoising performance. To address this issue, this paper introduces an end-to-end model aided nonlocal neural network (MAC-Net) which simultaneously takes the spectral low-rank model and spatial deep prior into account for HSI noise reduction. Specifically, motivated by the success of the spectral low-rank model in depicting the strong spectral correlations and the nonlocal similarity prior in capturing spatial long-range dependencies, we first build a spectral low-rank model and then integrate a nonlocal U-Net into the model. In this way, we obtain a hybrid model-based and DL-based HSI denoising method where the spatial local and nonlocal multi-scale and spectral low-rank structures are effectively exploited. After that, we cast the optimization and denoising procedure of the hybrid method as a forward process of a neural network and introduce a set of learnable modules to yield our MAC-Net. Compared with traditional model-based methods, our MAC-Net overcomes the difficulties of accurate modeling thanks to the strong learning and representation ability of DL. Unlike most “black-box” DL-based methods, the spectral low-rank model is beneficial to increase the generalization ability of the network and decrease the requirement of training samples. Experimental results on the natural and remote sensing HSIs show that MAC-Net achieves state-of-the-art performance over both model-based and DL-based methods.

# Requirements

We tested the implementation in Python 3.7.




# Datasets

* The ICVL dataset can be downloaded from [Link to Dataset](http://icvl.cs.bgu.ac.il/hyperspectral/).
 
* The 100 HSIs used in our training can be found in [ICVL_train.txt](https://github.com/bearshng/mac-net/blob/master/ICVL_train.txt).


* The 50 HSIs used for testing can be found in [ICVL_test.txt](https://github.com/bearshng/mac-net/blob/master/ICVL_test_gauss.txt).

* The sample WDC testing HSI can be accssed via [Google Drive](https://drive.google.com/drive/folders/1XI2S-AVCsx1jNyO4-XvQnfY8n7sXj1mW?usp=sharing).


# Test

## Test with known noise level

`python test.py  --test_path '---' --channels 16 --num_half_layer 5 --blind 0 --noise_level 15  --gt_path '---'  --gpus 0 --verbose 1`

## Test with unknown noise level

`python test.py  --test_path '---' --channels 16 --num_half_layer 5 --blind 1 --noise_level 0  --gt_path '---'  --gpus 0 --verbose 1`

## Test with real-world remote sensing HSI
`python test.py  --test_path '---' --channels 16 --num_half_layer 5 --blind 1 --noise_level 0  --gt_path '---'  --gpus 0 --verbose 1 --save 1 --rs_real 1`


### Important arguments

* `test_path `: your data path
* `channels `: the dimensiona of feature extractor
* `blind `: blind denoising
* `noise_level `: the maximum noise level
* `gt_path `: path to ground truth image
* `gpus `: device id
* `save` : save results 
* `rs_real `: realworld remote sensing HSI




# Train

## train your own model

`python train.py --noise_level 15 --lr 5e-3 --patch_size 64 --train_path 'your_path'  --test_path  'your_path' --log_dir './log' --out_dir './trained_model'  --verbose 1  --validation_every 300  --gpus 0    --num_epochs 300 --bandwise 1 --train_batch 16 --num_half_layer 5`


### Important arguments

* `noise_level `:  maximum noise level
* `bandwise `: add noise with a range of sigma
* `lr`: learning rate
* `patch_size`: patch size
*`train_path`: path to training set
* `test_path `: path to validation set
* `log_dir `: path to logs
* `out_dir `: path to resulted models
* `num_epochs `: maximum epochs
* `train_batch `: batch size

## Bibtex

`@ARTICLE{9631264,  author={Xiong, Fengchao and Zhou, Jun and Zhao, Qinling and Lu, Jianfeng and Qian, Yuntao},  journal={IEEE Transactions on Geoscience and Remote Sensing},   title={MAC-Net: Model Aided Nonlocal Neural Network for Hyperspectral Image Denoising},   year={2021},  volume={},  number={},  pages={1-1},  doi={10.1109/TGRS.2021.3131878}}`

## Update

Because of the difference between MATLAB and Python in SSIM index calculation, the value produced by Python is a little higher than that by MATLAB. In MATLAB,  the  SSIM of MACNet in the ICVL dataset are respectively:

| Sigma | SSIM  |
|:----------|:----------|
| [0-15]    | 0.9945    |
| [0-55]   | 0.9802    |
| [0-95]    | 0.9560    |
| [blind]    | 0.9700     |

We reimplemented the `ssim` function  to keep consisent with MATLAB. 
## Contact Information:

Fengchao  Xiong: fcxiong@njust.edu.cn

School of Computer Science and Engineering

Nanjing University of Science and Technology

