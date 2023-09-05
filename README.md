# Self-supervised Character-to-Character Distillation for Text Recognition （ICCV23）
This is the code of "Self-supervised Character-to-Character Distillation for Text Recognition". 
For more details, please refer to our [arxiv](https://arxiv.org/abs/2211.00288).

[中文解读](https://zhuanlan.zhihu.com/p/644350078)

## Pipeline 
<center>
<img src=graph/pipeline.png width="600px">
</center>

## Model architecture
![examples](graph/network.png)

## Environments
```bash
# 3090 Ubuntu 16.04 Cuda 11
conda create -n CCD python==3.7
source activate CCD
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install tensorboard==1.15.0
pip install tensorboardX==2.2
# The following optional dependencies are necessary
pip install yaml opencv-python Pillow LMDB nltk six natsort scipy sklearn scikit-image matplotlib editdistance tqdm
pip install fastai==1.0.60 imgaug==0.4.0
```

## Pretrain
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train.py --config ./Dino/configs/CCD_pretrain_ViT_xxx.yaml
```
## Finetune
```bash
#update model.pretrain_checkpoint in CCD_vision_model_xxx.yaml
CUDA_VISIBLE_DEVICES=0,1 python train_finetune.py --config ./Dino/configs/CCD_vision_model_xxx.yaml
```

## Data (please refer to [DiG](https://github.com/ayumiymk/DiG))
```
    data_lmdb
    ├── charset_36.txt
    ├── Mask
    ├── TextSeg
    ├── Super_Resolution
    ├── training
    │   ├── label
    │   │   └── synth
    │   │       ├── MJ
    │   │       │   ├── MJ_train
    │   │       │   ├── MJ_valid
    │   │       │   └── MJ_test
    │   │       └── ST
    │   │── URD
    │   │    └── OCR-CC
    │   ├── ARD
    │   │   ├── Openimages
    │   │   │   ├── train_1
    │   │   │   ├── train_2
    │   │   │   ├── train_5
    │   │   │   ├── train_f
    │   │   │   └── validation
    │   │   └── TextOCR 
    ├── validation
    │   ├── 1.SVT
    │   ├── 2.IIIT
    │   ├── 3.IC13
    │   ├── 4.IC15
    │   ├── 5.COCO
    │   ├── 6.RCTW17
    │   ├── 7.Uber
    │   ├── 8.ArT
    │   ├── 9.LSVT
    │   ├── 10.MLT19
    │   └── 11.ReCTS
    └── evaluation
        └── benchmark
            ├── SVT
            ├── IIIT5k_3000
            ├── IC13_1015
            ├── IC15_2077
            ├── SVTP
            ├── CUTE80
            ├── COCOText
            ├── CTW
            ├── TotalText
            ├── HOST
            ├── WOST
            ├── MPSC
            └── WordArt
```

## Mask preparation
  - optional, kmeans results of Synth and URD
  - if you don't want to generate mask, you can generate mask results online. 
     please rewrite [code1](https://github.com/TongkunGuan/CCD/blob/main/Dino/dataset/dataset.py#L133) and [code2](https://github.com/TongkunGuan/CCD/blob/main/Dino/dataset/datasetsupervised_kmeans.py#L48)
```bash
cd ./mask_create
run generate_mask.py #parallelly process mask --> lmdb file
run merge.py #merge multiple lmdb files into single file
```

## Visualization
<div style="align: center">
<img src=graph/order.png width="800px">
<img src=graph/SM_1.png width="800px">
<img src=graph/SM_3.png width="800px">
<img src=graph/SM_2.png width="800px">
</div>

### TODO
- [ ] Release weights


## Citation
```bash
If you find our method useful for your reserach, please cite

@misc{guan2023selfsupervised,
      title={Self-supervised Character-to-Character Distillation for Text Recognition}, 
      author={Tongkun Guan and Wei Shen and Xue Yang and Qi Feng and Zekun Jiang and Xiaokang Yang},
      year={2023},
      eprint={2211.00288},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## License
```bash
- This code are only free for academic research purposes and licensed under the 2-clause BSD License - see the LICENSE file for details.
```
