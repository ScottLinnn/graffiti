import argparse

import numpy as np

import imageio

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from .dnn_models.d2net.lib.model_test import D2Net
from .dnn_models.d2net.lib.utils import preprocess_image
from .dnn_models.d2net.lib.pyramid import process_multiscale


def extract_features(
    image_list_file,
    model_file="models/d2_tf.pth",
    preprocessing="caffe",
    max_edge=1600,
    max_sum_edges=2800,
    multiscale=False,
    use_relu=True,
):
    # CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # # Argument parsing
    # parser = argparse.ArgumentParser(description='Feature extraction script')

    # parser.add_argument(
    #     '--image_list_file', type=str, required=True,
    #     help='path to a file containing a list of images to process'
    # )

    # parser.add_argument(
    #     '--preprocessing', type=str, default='caffe',
    #     help='image preprocessing (caffe or torch)'
    # )
    # parser.add_argument(
    #     '--model_file', type=str, default='models/d2_tf.pth',
    #     help='path to the full model'
    # )

    # parser.add_argument(
    #     '--max_edge', type=int, default=1600,
    #     help='maximum image size at network input'
    # )
    # parser.add_argument(
    #     '--max_sum_edges', type=int, default=2800,
    #     help='maximum sum of image sizes at network input'
    # )

    # parser.add_argument(
    #     '--output_extension', type=str, default='.d2-net',
    #     help='extension for the output'
    # )
    # parser.add_argument(
    #     '--output_type', type=str, default='npz',
    #     help='output file type (npz or mat)'
    # )

    # parser.add_argument(
    #     '--multiscale', dest='multiscale', action='store_true',
    #     help='extract multiscale features'
    # )
    # parser.set_defaults(multiscale=False)

    # parser.add_argument(
    #     '--no-relu', dest='use_relu', action='store_false',
    #     help='remove ReLU after the dense feature extraction module'
    # )
    # parser.set_defaults(use_relu=True)

    # args = parser.parse_args()

    # print(args)

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
        resized_image = image
        if max(resized_image.shape) > max_edge:
            resized_image = scipy.misc.imresize(
                resized_image, max_edge / max(resized_image.shape)
            ).astype("float")
        if sum(resized_image.shape[:2]) > max_sum_edges:
            resized_image = scipy.misc.imresize(
                resized_image, max_sum_edges / sum(resized_image.shape[:2])
            ).astype("float")

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
