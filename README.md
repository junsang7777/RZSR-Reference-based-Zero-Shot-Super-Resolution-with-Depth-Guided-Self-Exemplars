# RZSR: Reference-based Zero-Shot Super-Resolution with Depth Guided Self-Exemplars
This is official pytorch implementation of the paper "RZSR: Reference-based Zero-Shot Super-Resolution with Depth Guided Self-Exemplars"
## [Arxiv](https://arxiv.org/abs/2208.11313) | [IEEE Transactions on Multimedia](https://ieeexplore.ieee.org/document/9868165)

## Dependencies
* Python=3.7
* PyTorch
* opencv-python
* scikit-image(skimage)

## Code
Clone this repository into any place you want.

```python
    git clone https://github.com/junsang7777/RZSR.git 
    cd RZSR
```

## Demo
You can test our SR algorithm with your images. Place your image in "set" folder. (img - RGB, dep - Depth, ker - Kernel)

If you don't have kernel files, Change "kernels = None" in 'main.py' script

You can get the depth & kernel of the images from the repository as follows: Depth : [monodepth2](https://github.com/nianticlabs/monodepth2) & [Adabins](https://github.com/shariqfarooq123/AdaBins) Kernel : [KernelGAN](https://github.com/sefibk/KernelGAN)

```python
    python main.py
```


---
Set dir : img (Random gaussian blurred img) , dep (Adabins depth estimation result) , ker ( KernelGAN estimation result)

hyper-parmeter : hitogram_threshold & Number of BINs & configs.py etc...
![table5](https://user-images.githubusercontent.com/37012124/217458860-f75817ab-123a-47ae-a9a7-76de1047743a.png)

### Framework
![framework](https://user-images.githubusercontent.com/37012124/187117833-38ad62e7-cd89-4166-a2b0-6767082b1016.png)

---
### Historical Comaparision

![NIMA](https://user-images.githubusercontent.com/37012124/154197367-abb6d02a-88a2-4c98-ba94-b12e688462d8.png)

---
### Related Work
[“Zero-Shot” Super-Resolution using Deep Internal Learning](https://openaccess.thecvf.com/content_cvpr_2018/papers/Shocher_Zero-Shot_Super-Resolution_Using_CVPR_2018_paper.pdf) | [git](https://github.com/assafshocher/ZSSR)

[Robust Reference-based Super-Resolution
with Similarity-Aware Deformable Convolution](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shim_Robust_Reference-Based_Super-Resolution_With_Similarity-Aware_Deformable_Convolution_CVPR_2020_paper.pdf)

[Image Super-Resolution with Cross-Scale Non-Local Attention
and Exhaustive Self-Exemplars Mining](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mei_Image_Super-Resolution_With_Cross-Scale_Non-Local_Attention_and_Exhaustive_Self-Exemplars_Mining_CVPR_2020_paper.pdf) | [git](https://github.com/SHI-Labs/Cross-Scale-Non-Local-Attention)

### Acknowledgment
The code is based on [pytorch-ZSSR](https://github.com/HarukiYqM/pytorch-ZSSR)

### Citation (Early access)
```
@ARTICLE{9868165,
    author={Yoo, Jun-Sang and Kim, Dong-Wook and Lu, Yucheng and Jung, Seung-Won},
    journal={IEEE Transactions on Multimedia}, 
    title={RZSR: Reference-based Zero-Shot Super-Resolution with Depth Guided Self-Exemplars}, 
    year={2022},
    volume={},
    number={},
    pages={1-13},
    doi={10.1109/TMM.2022.3202018}}
```
