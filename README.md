# Self-supervised Character-to-Character Distillation for Text Recognition （ICCV23）
This is the code of "Self-supervised Character-to-Character Distillation for Text Recognition". 
For more details, please refer to our [arxiv](https://arxiv.org/abs/2211.00288).
Code will be released.

## Pipeline 
<center>
<img src=graph/pipeline.png width="600px">
</center>

## Model architecture
![examples](graph/network.png)

## Environments
```bash
# 3090 Ubuntu 16.04 Cuda 11
conda create -n SIGA python==3.7.0
source activate SIGA
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install tensorboard==2.11.2
pip install tensorboardX==2.2
pip install opencv-python
pip install Pillow LMDB nltk six natsort scipy
# if you meet bug about setuptools
# pip uninstall setuptools
# pip install setuptools==58.0.4
```
## Data
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

## Highlights
- **Dataset link:**
  - [Synth](https://github.com/FangShancheng/ABINet/README.md)
  - [evaluation](https://github.com/FangShancheng/ABINet/README.md)

## Visualization
<div style="align: center">
<img src=graph/order.png width="800px">
<img src=graph/SM_1.png width="800px">
<img src=graph/SM_3.png width="800px">
<img src=graph/SM_2.png width="800px">
</div>

### TODO
- [ ] Release data
- [ ] Release code


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
