<div align="center">

# LaRa: Latents and Rays for <br> Multi-Camera Bird’s-Eye-View <br> Semantic Segmentation



<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2206.13294-B31B1B.svg)](https://arxiv.org/abs/2206.13294)

</div>

## Description

PyTorch implementation for LaRa: [arxiv.2206.13294](https://arxiv.org/abs/2206.13294)

## ⚙ Setup <a name="setup"></a>

### Environment

First, clone the project
```bash
git clone https://github.com/F-Barto/LaRa.git
cd LaRa
```
Then, create the [conda](https://docs.conda.io/en/latest/miniconda.html) environment,
install dependencies, activate env, and install project.
```bash
conda env create -n LaRa -f requirements.yaml
conda activate LaRa
pip install -e .
```

### Paths configuration

Change the paths present in the `.env` file to configure the saving dir and the path to your dataset.

## 🏋️ Training <a name="training"></a>

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=LaRa_inCamrays_outCoord
```

You can override any parameter from the command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64 model=tiny_LaRa experiment=LaRa_inCamrays_outFourier
```

## Credits

This project used or adapted code from:
* https://github.com/nv-tlabs/lift-splat-shoot
* https://github.com/TRI-ML/packnet-sfm
* https://github.com/krasserm/perceiver-io

Please consider giving them a star or citing their work if you use them.