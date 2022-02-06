import os
import pickle

import csv
import cv2
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

path_to_model = 'models/whole_pretrained_500ep.pth'

with open(f"{os.getcwd()}/datasets/animals10_test", 'r') as f:
    reader = csv.reader(f, delimiter=',')
    images = list(reader)

net = torch.load(path_to_model)
net.eval()

with open(f"{os.getcwd()}/datasets/animals10_utils", 'rb') as f:
    data = pickle.load(f)
classes = data
# classes ={0: 'cow', 1: 'cat', 2: 'chicken', 3: 'butterfly', 4: 'dog', 5: 'elephant', 6: 'horse', 7: 'sheep', 8: 'spider', 9: 'squirrel'}

# hook the feature extractor
features_blobs = []


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


net._modules.get('layer4').register_forward_hook(hook_feature)

# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-2].data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def main():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    path_to_data = f"{os.getcwd()}/datasets/Animals-10/"
    path_to_write = f"{os.getcwd()}/CAM/Animals-10/"

    for image_file, _ in images:
        print(f'Predictions for {image_file}')
        # load test image
        img_pil = Image.open(path_to_data + image_file)
        if len(img_pil.getextrema()) == 4:
            continue
        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
        logit = net(img_variable)

        # load the imagenet category list
        # with open(LABELS_file) as f:
        #     classes = json.load(f)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        # output the prediction
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

        # generate class activation mapping for the top1 prediction
        CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

        # render the CAM and output
        print(f'Output {image_file} for the top1 prediction: {classes[idx[0]]}')
        img = cv2.imread(path_to_data + image_file)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(path_to_write + image_file, result)


if __name__ == '__main__':
    main()
