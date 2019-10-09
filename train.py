#! /usr/bin/env python

from __future__ import division
from __future__ import print_function

import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from transferLearner import train_model, initialize_model, device, get_unique_dir

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
# data_dir = "./data/hymenoptera_data"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "squeezenet"

# Number of classes in the dataset
# num_classes = 6

# Batch size for training (change depending on how much memory you have)
batch_size = 8

# Number of epochs to train for
num_epochs = 15

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

if __name__ == "__main__":
    for data_dir in ["data/polygonia_224_6cat_oversample-10"]:

        # Number of classes in the dataset
        num_classes = len(glob(data_dir + "/train/*"))

        # for model_name in ["resnet", "alexnet", "vgg", "squeezenet", "densenet"]:
        for model_name in ["vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]:

            # Initialize the model for this run
            model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

            # Print the model we just instantiated
            print(model)

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
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                          shuffle=True, num_workers=4)
                           for x in ['train', 'val']}
            class_names = image_datasets['train'].classes

            # Send the model to GPU
            model = model.to(device)

            # Gather the parameters to be optimized/updated in this run. If we are
            #  finetuning we will be updating all parameters. However, if we are
            #  doing feature extract method, we will only update the parameters
            #  that we have just initialized, i.e. the parameters with requires_grad
            #  is True.
            print("Params to learn:")
            if feature_extract:
                params_to_update = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        params_to_update.append(param)
                        print("\t", name)
            else:
                params_to_update = model.parameters()
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print("\t", name)

            # Observe that all parameters are being optimized
            optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

            # Setup the loss fxn
            criterion = nn.CrossEntropyLoss()

            run_id = "{}_{}{}".format(os.path.basename(data_dir),
                                      model_name,
                                      "_finetune" if feature_extract is False else "")

            log_dir = get_unique_dir(os.path.join("logs", run_id))

            run_id_inc = os.path.basename(log_dir)
            model_dir = os.path.join("models", run_id_inc)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Train and evaluate
            best_epochs, training_hist = train_model(model, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs,
                                                     is_inception=(model_name == "inception"), log_dir=log_dir)



            for epoch in best_epochs:
                model_wts = training_hist[epoch]["model_wts"]
                train_acc = training_hist[epoch]["train_acc"]
                val_acc = training_hist[epoch]["val_acc"]

                torch.save(model_wts, os.path.join(model_dir, "{}_ep{}_trn{:.2f}_val{:.2f}.pth".format(run_id,
                                                                                                     epoch,
                                                                                                     train_acc,
                                                                                                     val_acc)))
