# Butterfly wing classification with transfer learning

Research code from the USDA-ARS Pacific Basin Agricultural Research Center in Hilo, 2019, written for
Julian Dupuis's *Polygonia* species-delimitation work. The species were already separated by genomic
data. The questions were whether pretrained ImageNet CNNs could tell them apart from lab photographs of
the wings, and which wing regions carry the difference, as candidate characters for field guides and
identification keys. Grad-CAM was the tool for the second question.

## What it does

- **Transfer learning on a small dataset.** Starts from ImageNet-pretrained torchvision models
  (ResNet-18, AlexNet, VGG-11 and VGG-19, SqueezeNet, DenseNet-121). Each is trained two ways: with the
  backbone frozen and only the final layer retrained, and fully fine-tuned.
- **Balanced classes and held-out specimens.** The dataset was small and uneven, so classes were
  regrouped. Whole specimens were held out for validation and test, and only then was the training set
  oversampled to balance the classes, with random-crop and flip augmentation.
- **A sweep over data and models.** `train.py` runs every combination of wing view (dorsal, ventral,
  both), grouping (3 or 6 classes) and architecture. It logs each run to TensorBoard and saves
  per-epoch weights with train and validation accuracy in the file name.
- **Grad-CAM.** `test_model.py` runs saved models on held-out test images and writes heatmaps of the
  wing regions each prediction relied on.

## Results

On the combined dorsal and ventral set for three species (`polygonia_224_3cat_oversample-20`):
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
| `transferLearner.py` | Model setup and the training loop, with TensorBoard logging and per-epoch history |
| `train.py` | The experiment sweep (datasets by architectures) |
| `test_model.py` | Evaluation on held-out images, with Grad-CAM output |
| `gradcam.py` | Grad-CAM implementation (see Credits) |
| `logs/` | TensorBoard event files for the runs: `tensorboard --logdir logs` |

## Data

The wing photographs are not included. The scripts expect torchvision `ImageFolder` layout:
`data/<dataset>/train/<class>/*.jpg` and `data/<dataset>/val/<class>/*.jpg`, plus a `test/` split
for `test_model.py`.

## Requirements

Written in 2019 against PyTorch 1.x and torchvision of that era: `torch`, `torchvision`,
`tensorboardX`, `numpy`, `Pillow`, `matplotlib`, `colorcet`. Current releases may need small
updates (for example, the `pretrained=True` model arguments).

## Credits

- The training loop and model setup follow the PyTorch tutorial
  [Finetuning Torchvision Models](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html)
  by Nathan Inkawhich.
- `gradcam.py` is adapted from Utku Ozbulak's
  [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) (MIT License).
