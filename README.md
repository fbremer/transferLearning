# Butterfly wing classification with transfer learning

Author: Forest Bremer. Commits under the name Woods26 are also his.

Research code from the USDA-ARS Pacific Basin Agricultural Research Center in Hilo, 2019, written by
Forest Bremer for Julian Dupuis's *Polygonia* species-delimitation work. The species were already separated by genomic
data. The questions were whether pretrained ImageNet CNNs could tell them apart from lab photographs of
the wings, and which wing regions carry the difference, as candidate characters for field guides and
identification keys. Grad-CAM was the tool for the second question.

## What it does

- **Transfer learning on a small dataset.** Starts from ImageNet-pretrained torchvision models
  (ResNet-18, AlexNet, VGG-11 with batch norm, SqueezeNet, DenseNet-121, and later VGG-16 variants). Each is trained two ways: with the
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
- **Grad-CAM.** `test_model.py` runs saved models on held-out test images and writes heatmaps of the
  wing regions each prediction relied on.

## Results

On the set that combines both wing views, for three species (`polygonia_224_3cat_oversample-20`):
- 504 training images, after oversampling; 60 validation images (two per specimen, 10 specimens per
  species); 60 test images.
- No specimen appears in more than one split.
- At the final epoch, fine-tuned DenseNet, ResNet and VGG each classified 59 of 60 validation images
  correctly (98%). The frozen-backbone versions reached 77 to 88%.
- Validation accuracy also picked the best epoch, so treat these figures as optimistic. Test-set results
  weren't logged.

Using Grad-CAM to turn what the models attended to into human-readable identification features was
more exploratory, and no conclusions were written up.

## Layout

| File | Purpose |
| --- | --- |
| `transferLearner.py` | Model setup and the training loop, with TensorBoard logging and per-epoch accuracy history |
| `train.py` | Training script; dataset and model list edited per run |
| `test_model.py` | Evaluation on held-out images, with Grad-CAM output |
| `gradcam.py` | Grad-CAM implementation (see Credits) |
| `logs/` | TensorBoard event files for the runs: `tensorboard --logdir logs` |

## Data

The wing photographs are not included. The scripts expect torchvision `ImageFolder` layout:
`data/<dataset>/train/<class>/*.jpg` and `data/<dataset>/val/<class>/*.jpg`, plus a `test/` split
for `test_model.py`.

## Requirements

Written in 2019 against PyTorch 1.x and torchvision of that era: `torch`, `torchvision`,
`tensorboardX`, `tensorboard`, `numpy`, `Pillow`, `matplotlib`, `colorcet`. Current releases may need small
updates (for example, the `pretrained=True` model arguments).

Run `python train.py` (writes `logs/` and `models/`), then `python test_model.py` after editing the
model and data paths at its top (writes `heatmaps/`). View the curves with `tensorboard --logdir logs`.

Known issues, left as written in 2019: in `test_model.py` the test glob repeats `test/`, so it needs
the path fixed before it finds images, and the per-epoch history in `transferLearner.py` stores
accuracy in its loss fields. The TensorBoard loss curves are correct.

## Credits

- The training loop and model setup follow the PyTorch tutorial
  [Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
  by Nathan Inkawhich.
- `gradcam.py` is adapted from Utku Ozbulak's
  [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (MIT License).
- The license notices for both are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
