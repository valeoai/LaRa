<div align="center">

# LaRa: Latents and Rays for <br> Multi-Camera Bird’s-Eye-View <br> Semantic Segmentation



<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.2206.13294-B31B1B.svg)](https://arxiv.org/abs/2206.13294)
[![Conference](http://img.shields.io/badge/CoRL-2022-4b44ce.svg)](https://openreview.net/forum?id=abd_D-iVjk0)


This is the reference PyTorch implementation for training and testing depth prediction models using the method described 
in our paper [**LaRa: Latents and Rays for Multi-Camera Bird’s-Eye-View Semantic Segmentation**
](https://openreview.net/forum?id=abd_D-iVjk0)

</div>

If you find our work useful, please consider citing:
```bibtex
@inproceedings{
    bartoccioni2022lara,
    title={LaRa: Latents and Rays for Multi-Camera Bird{\textquoteright}s-Eye-View Semantic Segmentation},
    author={Florent Bartoccioni and Eloi Zablocki and Andrei Bursuc and Patrick Perez and Matthieu Cord and Karteek Alahari},
    booktitle={6th Annual Conference on Robot Learning},
    year={2022},
    url={https://openreview.net/forum?id=abd_D-iVjk0}
}
```

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

> **Note**
> A smaller and faster version of LaRa is available with `model=LaRaUP`.
> BEV features are first predicted at 25x25 resolution and then upsampled to 200x200.

> **Note**
> Results also improve with integrating plucker coordinates as geometric embedding (in addition to cam origin and ray direction)
> We recommand using `experiment=LaRa_inCamplucker_outCoord`


Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=LaRa_inCamrays_outCoord
```

You can override any parameter from the command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64 model=LaRaUP experiment=LaRa_inCamrays_outFourier
```

The ground-truth BEV map will be color-coded when logged to Tensorboard:
- RED visibility of whole object is between and 0 and 40% (visibility=1)
- GREEN visibility of whole object is between and 40 and 60% (visibility=2)
- CYAN visibility of whole object is between and 60 and 80% (visibility=3)
- YELLOW visibility of whole object is between and 80 and 100% (visibility=4)

## 🎖️ Acknowledgements

This project used or adapted code from:
* https://github.com/nv-tlabs/lift-splat-shoot
* https://github.com/krasserm/perceiver-io

In particular, to structure our code we used:
https://github.com/ashleve/lightning-hydra-template

Please consider giving them a star or citing their work if you use them.