#! /usr/bin/env python

from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


from transferLearner import train_model, initialize_model, device

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
# data_dir = "./data/hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "squeezenet"

# Number of classes in the dataset
num_classes = 3

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

if __name__ == "__main__":
    for data_dir in ["data/polygonia_224_3cat_oversample-20"]:

        # for model_name in ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]:
        for model_name in ["alexnet"]:

            # Initialize the model for this run
            model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # Print the model we just instantiated
            print(model_ft)

            # Data augmentation and normalization for training
            # Just normalization for validation
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomResizedCrop(input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.CenterCrop(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }

            print("Initializing Datasets and Dataloaders...")

            # Create training and validation datasets
            image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                              for x in ['train', 'val']}
            # Create training and validation dataloaders
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                           for x in ['train', 'val']}
            class_names = image_datasets['train'].classes


            # Send the model to GPU
            model_ft = model_ft.to(device)

            # Gather the parameters to be optimized/updated in this run. If we are
            #  finetuning we will be updating all parameters. However, if we are
            #  doing feature extract method, we will only update the parameters
            #  that we have just initialized, i.e. the parameters with requires_grad
            #  is True.
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in model_ft.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)
                        print("\t", name)
            else:
                params_to_update = model_ft.parameters()
                for name, param in model_ft.named_parameters():
                    if param.requires_grad:
                        print("\t", name)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            log_dir = "runs/{}_{}_{}{}".format(os.path.basename(data_dir),
                                               model_name,
                                               num_epochs,
                                               "_finetune" if feature_extract is False else "")

            # Train and evaluate
            model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs,
                                         is_inception=(model_name == "inception"), log_dir=log_dir)

            torch.save(model_ft.state_dict(),
                       "{}_{}_{}_{:4f}{}.pth".format(os.path.basename(data_dir),
                                                   model_name,
                                                   num_epochs,
                                                   sorted(hist)[-1],
                                                   "_finetune" if feature_extract is False else ""))
