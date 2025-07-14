# TNT-GS (ACM MM 2025)
TNT-GS: Truncated and Tailored Gaussian Splatting.

## Visable Demos

You can explore our visual demonstrations at the link below:

[**TNT-GS Webpage**](https://googolplexgoodenough.github.io/TNT-GS-webpage/)

More examples will be added soon—stay tuned!

## Code


This repository includes submodules. To clone it properly, please use one of the following commands:
```shell
# SSH
git clone git@github.com:GoogolplexGoodenough/TNT-GS.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/GoogolplexGoodenough/TNT-GS --recursive
```

And install the torch and torchvision packages from the official [Pytorch website](https://pytorch.org/). Make sure that the CUDA version matches your local NVCC version to ensure proper GPU support, especially for compiling CUDA-based submodules.
Once PyTorch is installed, use the following command to install the remaining dependencies:
```shell
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff_TNT_rasterization
```

## Running

To launch the optimizer, use the following command:

```shell
python train.py -s <path to COLMAP or NeRF Synthetic dataset> -m <output path> --ouput_size <desired model size>
```

The --output_size argument specifies the target model size (MB). (Default: 50). Generally we set it to be ~50% of SOTA model size for compactness in the paper.

The command-line arguments for train.py are largely consistent with those of 2D Gaussian Splatting. [2DGS](https://github.com/hbb1/2d-gaussian-splatting).


## DTU Dataset Demo

We provide a demo script for the DTU dataset. To run it, use:

```shell
python auto_run_DTU.py
```

Please modify the dataset path before running the script. The DTU dataset we use can be downloaded from this [Google Drive Link](https://drive.google.com/drive/folders/1SJFgt8qhQomHX55Q4xSvYE2C6-8tFll9), which is shared by the [2DGS repository](https://github.com/hbb1/2d-gaussian-splatting).


## More results

We provide the CUDA code based on [GSSurfels](https://github.com/turandai/gaussian_surfels.git). And the code based on [2DGS](https://github.com/hbb1/2d-gaussian-splatting) with more results ans scripts is coming soon.