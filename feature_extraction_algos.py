import os
import cv2
import keras.applications

import config
from extract_d2net import (
    extract_features,
)  # Assuming the above code is saved in feature_extractor.py

class FeatureExtractionAlgorithm:
    def get_features(self, image_list_file: str):
        pass

    @staticmethod
    def provides_keypoints():
        pass

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        pass

    @staticmethod
    def get_kp_desc_pair(feature_data):
        pass

###############################################################################

def get_image_list_from_file(image_list_file):
    image_path_list = []
    with open(image_list_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue
            assert os.path.exists(line), f"{line} not found"
            if os.path.isdir(line):
                # If the line is a directory, collect all image files within it
                directory_path = line
                image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
                is_image_file = \
                    lambda filename : filename.lower().endswith(image_extensions)
                image_files = [
                    os.path.join(directory_path, filename)
                    for filename in os.listdir(directory_path)
                    if is_image_file(filename)
                ]
                image_path_list.extend(image_files)
            else:
                # Otherwise, assume it's a direct image file path
                image_path_list.append(line)
    return image_path_list

###############################################################################

class SURF(FeatureExtractionAlgorithm):
    def __init__(self):
        self.surf = cv2.xfeatures2d.SURF_create()
    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            file_to_kp_desc_map[image_file] = \
                self.surf.detectAndCompute(img, None)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return feature_data

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

class D2Net(FeatureExtractionAlgorithm):
    def __init__(self, model_file):
        self.model_file = model_file

    def get_features(self, image_list_file: str):
        image_path_list = get_image_list_from_file(image_list_file)
        temp_file_name = "temp_file_list"
        with open(temp_file_name, 'w') as file:
            for image_path in image_path_list:
                file.write(image_path + '\n')

        features = extract_features(temp_file_name, self.model_file)
        os.remove(temp_file_name)
        return features

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return (feature_data["keypoints"], feature_data["descriptors"])

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

class ORB(FeatureExtractionAlgorithm):
    def __init__(self):
        self.orb = cv2.ORB_create()

    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            file_to_kp_desc_map[image_file] = \
                self.orb.detectAndCompute(img, None)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return feature_data

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

class VGG16(FeatureExtractionAlgorithm):
    def __init__(self):
        # Include all layers except the final classification layer.
        self.vgg16 = vgg16.VGG16(weights='imagenet', include_top=False)
        for model_layer in self.vgg16.layers:
            model_layer.trainable = False
        self.vgg16.summary()

    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            file_to_kp_desc_map[image_file] = self.vgg16.predict(img)

        return file_to_kp_desc_map

    @staticmethod
    def provides_keypoints():
        return False

###############################################################################

class BRIEF(FeatureExtractionAlgorithm):
    def __init__(self):
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.star = cv2.xfeatures2d.StarDetector_create()

    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            kp = self.star.detect(img, None)
            file_to_kp_desc_map[image_file] = \
                self.brief.compute(img, kp)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return feature_data

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

class SIFT(FeatureExtractionAlgorithm):
    def __init__(self):
        self.sift = cv2.SIFT_create()
    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            file_to_kp_desc_map[image_file] = \
                self.sift.detectAndCompute(img, None)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return feature_data

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

class KAZE(FeatureExtractionAlgorithm):
    def __init__(self):
        self.kaze = cv2.KAZE_create()
    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            file_to_kp_desc_map[image_file] = \
                self.kaze.detectAndCompute(img, None)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return feature_data

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

class AKAZE(FeatureExtractionAlgorithm):
    def __init__(self):
        self.akaze = cv2.AKAZE_create()
    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}

        image_path_list = get_image_list_from_file(image_list_file)

        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            file_to_kp_desc_map[image_file] = \
                self.akaze.detectAndCompute(img, None)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

    @staticmethod
    def get_kp_desc_pair(feature_data):
        return feature_data

    @staticmethod
    def provides_keypoints():
        return True

###############################################################################

