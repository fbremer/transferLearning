import glob
import os

import PIL
import numpy as np
import torch
from torchvision import transforms
import colorcet as cc

from gradcam import GradCam, save_class_activation_images
from transferLearner import device, initialize_model


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

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
    return image  # torch.autograd.Variable(image, requires_grad=True)


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


# regex = r"([0-9]+)$"
# serch_obj = re.search(regex, model_dir)
# if not serch_obj:
#     pass
# else:
#     pass
#     # return int(serch_obj.group(1))

# table_header = ("model_name", "target_layer", "data_dir", "model_dir")
# table_data = [("alexnet", 11, "data/polygonia_dorsal_224_3cat_oversample-10/test", "models/polygonia_dorsal_224_3cat_oversample-10_alexnet_finetune/polygonia_dorsal_224_3cat_oversample-10_alexnet_finetune_ep6_trn0.94_val1.00.pth"),
#               ()]


model_dir = ("models/polygonia_dorsal_224_3cat_oversample-10_alexnet_finetune/"
             "polygonia_dorsal_224_3cat_oversample-10_alexnet_finetune_ep6_trn0.94_val1.00.pth")

data_dir = "data/polygonia_dorsal_224_3cat_oversample-10/test"  # could be parsed from model dir?
model_name = "alexnet"  # can be parsed from model_dir?
target_layer = 35  # depends on model_name

batch_size = 8

# classes in the dataset
classes, class_to_idx = find_classes(data_dir)
idx_to_class = {v: k for k, v in class_to_idx.items()}

# load model
pretrained_model, input_size = initialize_model(model_name, len(classes), feature_extract=False, use_pretrained=True)

# load weights
pretrained_model.load_state_dict(
    torch.load(model_dir))
pretrained_model.eval()

# classify and gradcam
for image_path in glob.glob(os.path.join(data_dir, "test/*/*")):
    # image_path = 'data/polygonia_224_3cat_oversample-20/test/progne/progne_UASM370053_dorsal.jpg'
    img = PIL.Image.open(image_path)

    # print(pretrained_model)

    probs, label = predict2(image_path, pretrained_model.to(device))
    print(os.path.basename(image_path))
    print(probs)
    # print(*label, sep=", ")
    print([idx_to_class[idx] for idx in label], sep=", ")
    print("-" * 60 + "\n")

    # gradcam
    original_image = PIL.Image.open(image_path).convert('RGB')
    prep_img = process_image(original_image).unsqueeze(0)
    crop_image = only_crop_image(original_image)

    target_class = label[0]
    file_name_to_export = "butterfly_gradcam"

    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=target_layer)
    # Generate cam mask
    cam = grad_cam.generate_cam(prep_img, target_class)



    # Save mask
    save_class_activation_images(crop_image, cam,
                                 os.path.join('heatmaps', os.path.splitext(os.path.basename(model_dir))[0]),
                                 "{}_{}_{:.2f}".format(os.path.splitext(os.path.basename(image_path))[0],
                                                       idx_to_class[label[0]],
                                                       probs[0]))
