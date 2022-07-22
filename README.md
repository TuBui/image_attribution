# RepMix: Representation Mixing for Robust Attribution of Synthesized Images
![Python 3.8](https://img.shields.io/badge/Python-3.8-green) ![Pytorch 1.8.1](https://img.shields.io/badge/Pytorch-1.8.1-green) ![Licence CC-BY-4.0](https://img.shields.io/badge/license-CC--4.0--BY-blueviolet)

This repo contains official code and datasets for the ECCV 2022 paper ["RepMix: Representation Mixing for Robust Attribution of Synthesized Images"](https://arxiv.org/abs/2207.02063).

## Dependencies

We experimented with the following main libraries (other versions may still work):
```
pytorch == 1.8.1
torchvision == 0.9.1
imagenet-C (see below)
opencv-python >= 4.2.0
Pillow == 8.3.1
pytorch-lightning == 1.4.6
...

```
The full list of dependencies can be found at [requirements.txt](dependencies/requirements.txt).

To install imagenet-C:
```
git clone https://github.com/hendrycks/robustness.git && cd robustness/ImageNet-C/imagenet_c/ && pip install -e .
```

We also provide a [Dockerfile](dependencies/Dockerfile) so that you can build a docker image yourself. Alternatively you can download our pre-built docker image at:

```bash
docker pull tuvbui/ganprov:v1
```

## The Attribution88 benchmark
The full dataset can be downloaded [here](https://kahlan.cvssp.org/data/Flickr25K/tubui/eccv22_repmix/Attribution88.tar.gz) (30GB). It consists of 12000x8x11=1056000 images of 11 semantics and 8 sources (real + 7 GANs). The train, validation and test splits are also included in the tar file. 

We also release the processed test set [here](https://kahlan.cvssp.org/data/Flickr25K/tubui/eccv22_repmix/Attribution88_test.tar.gz) (5.4GB).

## Train and evaluate
To train the RepMix model:
```
python train.py -d /path/to/attribution88/directory -tl /path/to/train/split/train.csv -vl /path/to/validation/split/val.csv -o /output/directory
```

To test a model:
```
python test.py -d /path/to/attribution88/test/directory -l /path/to/test/split/test.csv -w /path/to/model/checkpoint/last.ckpt
```

## Reference
```
@InProceedings{bui2022repmix,
  title = {RepMix: Representation Mixing for Robust Attribution of Synthesized Images},
  author = {Bui, Tu and Yu, Ning and Collomosse, John},
  booktitle = {Proc. ECCV},
  year = {2022}
}
```
