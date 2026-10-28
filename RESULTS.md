# Validation accuracy from the 2019 TensorBoard logs

Each number is `logs/val_acc` in `logs/`. The step numbers are the epoch indexes written at the time. Most runs logged epochs 1 through 14. A run whose directory name ends in `_finetune` updated the whole network. The others trained the new classifier on a frozen ImageNet backbone. A `_001` suffix is a second directory with that setting. `train.py` wrote a weight file for each epoch whose validation accuracy matched or beat the best so far. Those files were not in the Drive download. Test-set scores were not logged.

Only the three-species split, `polygonia_224_3cat_oversample-20`, is in the image-set release. Its validation split is 60 images, so those rows are also written as a count. The other three dataset folders were not in the download. Their rows stay as the logged fraction.

Runs whose name is `vgg16` or `vgg16_bn` were built by those branches in `transferLearner.py`. The `vgg16` branch constructs VGG-19, and the `vgg16_bn` branch constructs VGG-19 with batch norm.

## Both wing views, 3 species, oversample-20

`polygonia_224_3cat_oversample-20`

| Model | Training | Repeat | Epochs logged | Last epoch | Best epoch |
| --- | --- | --- | --- | --- | --- |
| alexnet | frozen |  | 1-14 (14) | 53 of 60 | 53 of 60 at epoch 11 |
| alexnet | fine-tuned |  | 1-14 (14) | 50 of 60 | 58 of 60 at epoch 11 |
| densenet | frozen |  | 1-14 (14) | 46 of 60 | 51 of 60 at epoch 13 |
| densenet | fine-tuned |  | 1-14 (14) | 59 of 60 | 60 of 60 at epoch 9 |
| resnet | frozen |  | 1-14 (14) | 52 of 60 | 52 of 60 at epoch 14 |
| resnet | fine-tuned |  | 1-14 (14) | 59 of 60 | 60 of 60 at epoch 6 |
| squeezenet | frozen |  | 1-14 (14) | 51 of 60 | 56 of 60 at epoch 11 |
| squeezenet | fine-tuned |  | 1-14 (14) | 51 of 60 | 55 of 60 at epoch 6 |
| vgg | frozen |  | 1-14 (14) | 53 of 60 | 54 of 60 at epoch 11 |
| vgg | fine-tuned |  | 1-14 (14) | 59 of 60 | 60 of 60 at epoch 4 |

## Both wing views, 6 classes, oversample-10

`polygonia_224_6cat_oversample-10`

| Model | Training | Repeat | Epochs logged | Last epoch | Best epoch |
| --- | --- | --- | --- | --- | --- |
| alexnet | frozen |  | 1-14 (14) | 85.00% | 86.67% at epoch 6 |
| alexnet | fine-tuned |  | 1-14 (14) | 93.33% | 95.00% at epoch 9 |
| densenet | frozen |  | 1-14 (14) | 91.67% | 95.00% at epoch 10 |
| densenet | fine-tuned |  | 1-14 (14) | 98.33% | 98.33% at epoch 6 |
| resnet | frozen |  | 1-14 (14) | 91.67% | 93.33% at epoch 12 |
| resnet | fine-tuned |  | 1-14 (14) | 100.00% | 100.00% at epoch 6 |
| squeezenet | frozen |  | 1-14 (14) | 88.33% | 88.33% at epoch 12 |
| squeezenet | fine-tuned |  | 1-14 (14) | 63.33% | 91.67% at epoch 13 |
| vgg | frozen |  | 1-14 (14) | 91.67% | 91.67% at epoch 5 |
| vgg16 | frozen |  | 1-14 (14) | 80.00% | 83.33% at epoch 8 |
| vgg16_bn | frozen |  | 1-12 (12) | 81.67% | 85.00% at epoch 4 |
| vgg16 | fine-tuned |  | 1-3 (3) | 90.00% | 96.67% at epoch 2 |
| vgg | fine-tuned |  | 1-14 (14) | 95.00% | 100.00% at epoch 6 |

## Dorsal only, 3 species, oversample-10

`polygonia_dorsal_224_3cat_oversample-10`

| Model | Training | Repeat | Epochs logged | Last epoch | Best epoch |
| --- | --- | --- | --- | --- | --- |
| alexnet | frozen |  | 1-14 (14) | 83.33% | 90.00% at epoch 11 |
| alexnet | frozen | second | 1-14 (14) | 90.00% | 90.00% at epoch 1 |
| alexnet | fine-tuned |  | 1-14 (14) | 86.67% | 100.00% at epoch 6 |
| alexnet | fine-tuned | second | 1-14 (14) | 93.33% | 100.00% at epoch 11 |
| densenet | frozen |  | 1-14 (14) | 90.00% | 93.33% at epoch 11 |
| densenet | fine-tuned |  | 1-14 (14) | 96.67% | 100.00% at epoch 9 |
| resnet | frozen |  | 1-14 (14) | 83.33% | 90.00% at epoch 6 |
| resnet | fine-tuned |  | 1-14 (14) | 96.67% | 100.00% at epoch 1 |
| squeezenet | frozen |  | 1-14 (14) | 93.33% | 93.33% at epoch 7 |
| squeezenet | fine-tuned |  | 1-14 (14) | 40.00% | 83.33% at epoch 11 |
| squeezenet | fine-tuned | second | 1-14 (14) | 96.67% | 96.67% at epoch 14 |
| vgg | frozen |  | 1-14 (14) | 90.00% | 90.00% at epoch 1 |
| vgg | fine-tuned |  | 1-14 (14) | 100.00% | 100.00% at epoch 5 |

## Ventral only, 3 species, oversample-10

`polygonia_ventral_224_3cat_oversample-10`

| Model | Training | Repeat | Epochs logged | Last epoch | Best epoch |
| --- | --- | --- | --- | --- | --- |
| alexnet | frozen |  | 1-14 (14) | 86.67% | 90.00% at epoch 8 |
| alexnet | frozen | second | 1-14 (14) | 96.67% | 96.67% at epoch 14 |
| alexnet | fine-tuned |  | 1-14 (14) | 96.67% | 96.67% at epoch 9 |
| alexnet | fine-tuned | second | 1-14 (14) | 93.33% | 100.00% at epoch 8 |
| densenet | frozen |  | 1-14 (14) | 93.33% | 100.00% at epoch 11 |
| densenet | fine-tuned |  | 1-14 (14) | 100.00% | 100.00% at epoch 2 |
| resnet | frozen |  | 1-14 (14) | 83.33% | 90.00% at epoch 5 |
| resnet | fine-tuned |  | 1-14 (14) | 96.67% | 100.00% at epoch 3 |
| squeezenet | frozen |  | 1-14 (14) | 93.33% | 96.67% at epoch 8 |
| squeezenet | fine-tuned |  | 1-14 (14) | 63.33% | 96.67% at epoch 10 |
| squeezenet | fine-tuned | second | 1-14 (14) | 33.33% | 90.00% at epoch 2 |
| vgg | frozen |  | 1-14 (14) | 96.67% | 96.67% at epoch 14 |
| vgg | fine-tuned |  | 1-14 (14) | 100.00% | 100.00% at epoch 10 |
