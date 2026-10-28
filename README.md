# Butterfly wing classification with transfer learning

Author: Forest Bremer. Commits under the name Woods26 are also his.

Research code from the USDA-ARS Pacific Basin Agricultural Research Center in Hilo, 2019, written by
Forest Bremer for Julian Dupuis's *Polygonia* species-delimitation work. The species were already separated by genomic
data. The questions were whether pretrained ImageNet CNNs could tell them apart from lab photographs of
the wings, and whether the activations behind a classification showed human-readable wing features
that could serve as characters for field guides and identification keys. Grad-CAM was the visualization
used for the second question. By this method, that was not helpful. Devanshi Bhimjiyani was the
student intern on the project with Forest.

## What it does

- **Transfer learning on a small dataset.** Starts from ImageNet-pretrained torchvision models
  (ResNet-18, AlexNet, VGG-11 with batch norm, SqueezeNet, DenseNet-121, and later VGG runs). Each is trained two ways: with the
  backbone frozen and only the final layer retrained, and fully fine-tuned.
- **Balanced classes and held-out specimens.** The dataset was small and uneven, so classes were
  regrouped. Whole specimens were held out for validation and test, and only then was the training set
  oversampled to balance the classes, with random-crop and flip augmentation.
- **A sweep over data and models.** `train.py` trains one dataset and a list of models per run; the
  dataset, the model list and `feature_extract` are set at the top of the file and were edited between
  runs. The runs in `logs/` cover wing view (dorsal, ventral, both), grouping (3 or 6 classes),
  architecture, and frozen against fine-tuned. Each run logs to TensorBoard and saves per-epoch
  weights with train and validation accuracy in the file name. The checked-in `train.py` is the last
  configuration used, not the whole sweep.
- **Grad-CAM.** `test_model.py` writes a visualization of the activations that lead to a classification.

## Results

On the set that combines both wing views, for three species (`polygonia_224_3cat_oversample-20`):
- 504 training images, after oversampling; 60 validation images (two per specimen, 10 specimens per
  species); 60 test images.
- No specimen appears in more than one split.
- At the last logged epoch, fine-tuned DenseNet, ResNet and VGG each classified 59 of 60 validation
  images correctly. Each of those three runs also logged an earlier epoch at 60 of 60. Frozen-backbone
  runs on this set finished between 46 and 53 of 60.
- `train.py` saved weights for each epoch that matched or beat the best validation accuracy so far.
  Those weight files were not in the recovered download. Test-set scores were not logged.

The full table, including the 6-class, dorsal-only, and ventral-only runs, is in
[RESULTS.md](RESULTS.md). Those three image folders were not in the download, so their rows stay
as the logged fraction.

Grad-CAM visualizes the activations that lead to a classification. It was the attempt to see whether
the models were detecting human-readable wing features. By this method, that was not helpful. The two
images under `gradcam/` in the photograph zip are those visualizations.

## Layout

| File | Purpose |
| --- | --- |
| `transferLearner.py` | Model setup and the training loop, with TensorBoard logging and per-epoch accuracy history |
| `train.py` | Training script; dataset and model list edited per run |
| `test_model.py` | Evaluation on held-out images, with Grad-CAM output |
| `gradcam.py` | Grad-CAM implementation (see Credits) |
| `logs/` | TensorBoard event files for the runs: `tensorboard --logdir logs` |
| `RESULTS.md` | Validation accuracy read from those logs |
| `rebuild_imagefolders.py` | Rebuilds the ImageFolder directories from the photograph release |

## Data

The wing photographs are in the release
[image-sets-2019](https://github.com/fbremer/transferLearning/releases/tag/image-sets-2019),
in `polygonia-image-sets.zip`. The scripts expect torchvision `ImageFolder` layout:
`data/<dataset>/train/<class>/*.jpg` and `data/<dataset>/val/<class>/*.jpg`, plus a `test/` split
for `test_model.py`.

Each distinct photograph is stored once. The split tables name every training file,
including the repeated copies.

- `photographs/224/`, 506 photographs at 224×224, dorsal and ventral.
- `photographs/dorsal_1024/`, 253 dorsal photographs at 1024×768, one per specimen, in
  species folders. The form stays in the filename.
- `splits/polygonia_224_3cat_oversample-20.tsv`, faunus, progne, and satyrus. 504 training
  rows (344 photographs and 160 repeated copies), 60 validation, 60 test.
- `splits/dorsal_1024_4class.tsv`, faunus, gracilis, progne, and satyrus, train and val,
  plus four filename aliases.
- `gradcam/`, the two Grad-CAM visualizations.

Gracilis is in the 224-pixel photographs, as gracilisG and gracilisZ, and in the dorsal set
as its own class. It is not in the three-species split.

Rebuild the directories the scripts expect:

    python rebuild_imagefolders.py /path/to/polygonia-image-sets

That writes `data/polygonia_224_3cat_oversample-20/` and `data/dorsal_1024_4class/`.

Not recovered: the 6-class folder, the dorsal-only and ventral-only training folders, and the
per-epoch `.pth` files named in `train.py`.

## Requirements

Written in 2019 against PyTorch 1.x and torchvision of that era: `torch`, `torchvision`,
`tensorboardX`, `tensorboard`, `numpy`, `Pillow`, `matplotlib`, `colorcet`. Current releases may need small
updates (for example, the `pretrained=True` model arguments).

Run `python train.py` (writes `logs/` and `models/`), then `python test_model.py` after editing the
model and data paths at its top (writes `heatmaps/`). View the curves with `tensorboard --logdir logs`.

Known issues, left as written in 2019: in `test_model.py` the test glob repeats `test/`, so it needs
the path fixed before it finds images, and the per-epoch history in `transferLearner.py` stores
accuracy in its loss fields. The TensorBoard loss curves are correct. The `vgg16` branch constructs
VGG-19, and the `vgg16_bn` branch constructs VGG-19 with batch norm. Runs logged under those names
are those models.

## Credits

- The training loop and model setup follow the PyTorch tutorial
  [Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
  by Nathan Inkawhich.
- `gradcam.py` is adapted from Utku Ozbulak's
  [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (MIT License).
- The license notices for both are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
