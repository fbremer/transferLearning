from __future__ import division
from __future__ import print_function

import copy
import os
import re
import time
from glob import glob

import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter
from torchvision import models

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_log = 5


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, log_dir="logs"):
    since = time.time()

    hist = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # logging
            writer = SummaryWriter(log_dir)

            if phase == 'val':
                writer.add_scalar('logs/val_loss', epoch_loss, epoch)
                writer.add_scalar('logs/val_acc', epoch_acc, epoch)
            else:
                writer.add_scalar('logs/train_loss', epoch_loss, epoch)
                writer.add_scalar('logs/train_acc', epoch_acc, epoch)

            writer.close()

            # save history
            if phase == 'train':
                hist.append({"epoch": epoch,
                             "train_loss": epoch_acc,
                             "train_acc": epoch_acc,
                             })

            if phase == 'val':
                hist[epoch].update({"val_loss": epoch_acc,
                                    "val_acc": epoch_acc,
                                    })

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_epoch.append(epoch)
                hist[epoch].update({"model_wts": copy.deepcopy(model.state_dict())})

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(hist[best_epoch]["model_wts"])
    return best_epoch, hist


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def get_unique_dir(path, width=3):
    # if it doesn't exist, create
    if not os.path.isdir(path):
        # log.debug("Creating new directory - {}".format(path))
        os.makedirs(path)
        return path

    # if it's empty, use
    if not os.listdir(path):
        # log.debug("Using empty directory - {}".format(path))
        return path

    # otherwise, increment the highest number folder in the series

    def get_trailing_number(search_text):
        serch_obj = re.search(r"([0-9]+)$", search_text)
        if not serch_obj:
            return 0
        else:
            return int(serch_obj.group(1))

    dirs = glob(path + "*")
    next_num = sorted([get_trailing_number(d) for d in dirs])[-1] + 1
    new_path = "{0}_{1:0>{2}}".format(path, next_num, width)

    # log.debug("Creating new incremented directory - {}".format(new_path))
    os.makedirs(new_path)
    return new_path
