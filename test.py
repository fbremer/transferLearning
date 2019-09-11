import glob
import os

import PIL
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision import models

from helper import device
from gradcam import GradCam, save_class_activation_images

data_dir = "data/polygonia_224_3cat_oversample-16"
batch_size = 8
num_classes = 3
input_size = 224
model_name = "alexnet"
feature_extract = True

# Initialize the model for this run
# pretrained_model = models.AlexNet(num_classes=num_classes)

pretrained_model = models.alexnet(pretrained=True)
num_ftrs = pretrained_model.classifier[6].in_features
pretrained_model.classifier[6] = torch.nn.Linear(num_ftrs, num_classes)

pretrained_model.load_state_dict(torch.load('polygonia_224_3cat_oversample-20_alexnet_15_0.933333_finetune.pth'))
pretrained_model.eval()

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Create training and validation datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['test']}
# Create training and validation dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False, num_workers=4)
               for x in ['test']}
class_names = image_datasets['test'].classes





def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = preprocess(image)
    # image.unsqueeze_(0)
    return image #torch.autograd.Variable(image, requires_grad=True)


def only_crop_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size)
    ])
    image = preprocess(image)
    return image


def predict2(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Implement the code to predict the class from an image file
    img = PIL.Image.open(image_path)
    img = process_image(img)

    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img)

    model.eval()
    inputs = torch.autograd.Variable(img).to(device)
    logits = model.forward(inputs)

    ps = torch.nn.functional.softmax(logits, dim=1)
    topk = ps.cpu().topk(topk)

    return (e.data.numpy().squeeze().tolist() for e in topk)

for image_path in glob.glob("data/polygonia_224_3cat_oversample-20/test/*/*"):

    # image_path = 'data/polygonia_224_3cat_oversample-20/test/progne/progne_UASM370053_dorsal.jpg'
    img = PIL.Image.open(image_path)

    # print(pretrained_model)
    idx_to_class = {v: k for k, v in image_datasets['test'].class_to_idx.items()}

    probs, label = predict2(image_path, pretrained_model.to(device))
    print(os.path.basename(image_path))
    print(probs)
    # print(*label, sep=", ")
    print([idx_to_class[idx] for idx in label], sep=", ")
    print("-"*60 + "\n")


    # gradcam
    original_image = PIL.Image.open(image_path).convert('RGB')
    prep_img = process_image(original_image).unsqueeze(0)
    crop_image = only_crop_image(original_image)


    target_class = label[0]
    file_name_to_export = "butterfly_gradcam"

    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(crop_image, cam,
                                 "{}_{}_{:.2f}".format(os.path.splitext(os.path.basename(image_path))[0],
                                                       idx_to_class[label[0]],
                                                       probs[0]))
