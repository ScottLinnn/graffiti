import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import os
import sys
import scipy
import scipy.io
import scipy.misc

from PIL import Image

sys.path.append(os.path.join("dnn_models", "d2net"))
from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale


def resize_image(image, target_size):
    # Convert the NumPy array to a PIL Image object
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    # Resize the image
    resized_pil_image = pil_image.resize(target_size, Image.ANTIALIAS)
    # Convert the PIL Image object back to a NumPy array
    resized_image = np.array(resized_pil_image).astype("float") / 255
    return resized_image


def extract_features(
    image_list_file,
    model_file="models/d2_tf.pth",
    preprocessing="caffe",
    max_edge=800,
    max_sum_edges=1400,
    multiscale=False,
    use_relu=True,
):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Creating CNN model
    model = D2Net(model_file=model_file, use_relu=use_relu, use_cuda=use_cuda)

    # Process the file
    with open(image_list_file, "r") as f:
        lines = f.readlines()

    feature_data = {}
    for line in tqdm(lines, total=len(lines)):
        path = line.strip()

        image = imageio.imread(path)
        if len(image.shape) == 2:
            image = image[:, :, np.newaxis]
            image = np.repeat(image, 3, -1)

        # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.

        # Get the resizing factor for max_edge constraint
        resized_image = image
        if max(resized_image.shape) > max_edge:
            factor_max_edge = max_edge / max(resized_image.shape)
            new_size_max_edge = (
                int(resized_image.shape[1] * factor_max_edge),
                int(resized_image.shape[0] * factor_max_edge),
            )
            resized_image = resize_image(resized_image, new_size_max_edge)

        # Get the resizing factor for max_sum_edges constraint
        if sum(resized_image.shape[:2]) > max_sum_edges:
            factor_max_sum_edges = max_sum_edges / sum(resized_image.shape[:2])
            new_size_max_sum_edges = (
                int(resized_image.shape[1] * factor_max_sum_edges),
                int(resized_image.shape[0] * factor_max_sum_edges),
            )
            resized_image = resize_image(resized_image, new_size_max_sum_edges)

        # resized_image = image
        # if max(resized_image.shape) > max_edge:
        #     resized_image = scipy.misc.imresize(
        #         resized_image, max_edge / max(resized_image.shape)
        #     ).astype("float")
        # if sum(resized_image.shape[:2]) > max_sum_edges:
        #     resized_image = scipy.misc.imresize(
        #         resized_image, max_sum_edges / sum(resized_image.shape[:2])
        #     ).astype("float")

        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(resized_image, preprocessing=preprocessing)
        with torch.no_grad():
            if multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device,
                    ),
                    model,
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device,
                    ),
                    model,
                    scales=[1],
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i
        keypoints[:, 1] *= fact_j
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]]

        # Store the feature data
        feature_data[path] = {
            "keypoints": keypoints,
            "scores": scores,
            "descriptors": descriptors,
        }
    return feature_data
