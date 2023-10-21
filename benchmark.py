import argparse
import cv2
import numpy as np
from scipy import stats
import logging
from enum import Enum
import os
import torch
import tensorflow as tf

import config
import matching_algos

###############################################################################

class BenchmarkType(Enum):
    FALSE_POS = 1
    FALSE_NEG = 2

###############################################################################

# Compute false positives for images of different things
class benchmarker:
    def __init__(self, benchmark_type, image_list_file, matching_alg):
        self.flann = cv2.FlannBasedMatcher(config.INDEX_PARAMS, config.SEARCH_PARAMS)
        self.image_list_file = image_list_file
        self.matching_alg = matching_alg
        self.benchmark_type = benchmark_type

    # Finds the median bin of a histogram
    def hist_median(self, hist):
        total_samples = hist.sum()
        half_samples = total_samples // 2
        s = 0
        for i in range(len(hist)):  # changed xrange to range
            s += hist[i]
            if s > half_samples:
                return i

###############################################################################

    def extract_good_matches(self, matches):
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < (config.DISTANCE_THRESH * n.distance):
                good.append(m)
        return good

###############################################################################

    # Returns a match score between two images
    def compute_match_score(self, query_data, train_data):
        query_kp, query_des, query_hist, query_img = query_data
        train_kp, train_des, train_hist, train_img = train_data

        score = 0

        matches = \
            self.extract_good_matches(
                self.flann.knnMatch(query_des, train_des, k=2))

        # Filter out high intensity pixel values
        train_hist[245:] = train_hist[244]
        query_hist[245:] = query_hist[244]

        # Filter out low intensity pixel values
        train_hist[:10] = train_hist[10]
        query_hist[:10] = query_hist[10]

        # Shift histograms based on median bin to match score
        train_hist_median = self.hist_median(train_hist)
        if train_hist_median is None:
            train_hist_median = 128
        query_hist_median = self.hist_median(query_hist)
        if query_hist_median is None:
            query_hist_median = 128
        if query_hist_median > train_hist_median:
            n_shift = query_hist_median - train_hist_median
            hist_new = train_hist.copy()
            hist_new[:] = 0
            hist_new[n_shift:255] = train_hist[: 255 - n_shift]
            train_hist = hist_new
        else:
            n_shift = train_hist_median - query_hist_median
            hist_new = query_hist.copy()
            hist_new[:] = 0
            hist_new[n_shift:255] = query_hist[: 255 - n_shift]
            query_hist = hist_new

        # Find histogram correlation
        hist_correlation = \
            cv2.compareHist(train_hist, query_hist, cv2.HISTCMP_CORREL) * 100

        # Find Mann-Whitney U Test score
        hist_mwn = (
            stats.mannwhitneyu(
                query_hist.flatten(),
                train_hist.flatten(),
                use_continuity=True,
                alternative="two-sided",
            ).pvalue
            * 100
        )

        # Find DCT correlation
        imf = np.float32(query_img) / 255.0  # Float conversion/scale
        dst = cv2.dct(imf)  # Calculate the dct
        img1 = dst

        imf = np.float32(train_img) / 255.0  # Float conversion/scale
        dst = cv2.dct(imf)  # Calculate the dct
        img2 = dst

        dct_diff = img1 - img2
        dct_correl = (
            cv2.compareHist(img1.flatten(), img2.flatten(), cv2.HISTCMP_CORREL) * 100
        )

        logging.debug(f"NUMBER OF GOOD MATCHES: {len(matches)}")
        logging.debug(f"HISTOGRAM CORRELATION: {hist_correlation}")
        logging.debug(f"MWN CORRELATION: {hist_mwn}")
        logging.debug(f"DCT CORRELATION: {dct_correl}")

        # Calculate match threshold based on the number of keypoints detected in the database image and the query image
        train_threshold = 0.1 * len(train_kp)
        query_threshold = 0.1 * len(query_kp)
        threshold = max(train_threshold, query_threshold)

        logging.debug(f"THRESHOLD: {threshold}")

        # Reject match if number of detected matches is less than the threshold
        if len(matches) < threshold:
            return None, None
        else:
            score += len(matches)

        # calculate the relative displacement between two group of key points
        shift_xs = []
        shift_ys = []
        for m in matches:
            k_q = query_kp[m.queryIdx]
            k_t = train_kp[m.trainIdx]

            k_q_coord = self.matching_alg.get_keypoint_coordinates(k_q);
            k_t_coord = self.matching_alg.get_keypoint_coordinates(k_t);
            shift_xs.append(k_q_coord[0] - k_t_coord[0])
            shift_ys.append(k_q_coord[1] - k_t_coord[1])

        shift_x1 = sum(shift_xs) / len(shift_xs)
        shift_y1 = sum(shift_ys) / len(shift_ys)
        shift_x2 = np.median(np.array(shift_xs))
        shift_y2 = np.median(np.array(shift_ys))
        shift_x = (shift_x1 + shift_x2) / 2
        shift_y = (shift_y1 + shift_y2) / 2

        hist_test_passes = 0
        if hist_correlation > config.CORREL_TH:
            hist_test_passes += 1
        if dct_correl > config.DCT_TH:
            hist_test_passes += 1
        if hist_mwn > config.MWN_TH:
            hist_test_passes += 1

        # Reject match if less than 2 hist tests pass
        if hist_test_passes >= 2:
            score += hist_correlation + dct_correl + hist_mwn
        else:
            return None, None

        logging.info(f"SCORE IS: {score}")
        return score, (shift_x, shift_y)

###############################################################################

    @staticmethod
    def calculate_histogram(image):
        """
        Calculate a histogram of a single-channel (greyscale) image using 256
        bins, one for each possible value of a pixel. This histogram provides
        information about the distribution of pixel densities in the image.
        """
        return cv2.calcHist([image], [0], None, [256], [0, 256])

###############################################################################

    @staticmethod
    def images_are_of_same_object(filepath1, filepath2):
        different_things_dirpath = \
            os.path.abspath('test_images/different_things')
        same_things_dirpath = \
            os.path.abspath('test_images/same_things')

        filepaths = [filepath1, filepath2]

        file_in_dir = \
            lambda (filepath, dirpath) :
                os.path.commonpath(filepath, dirpath) == dirpath

        if not file_in_dir(filepath1, different_things_dirpath) and
           not file_in_dir(filepath1, same_things_dirpath):
               raise Exception("Image not placed correctly: " + filepath1)

        if not file_in_dir(filepath2, different_things_dirpath) and
           not file_in_dir(filepath2, same_things_dirpath):
               raise Exception("Image not placed correctly: " + filepath2)

        images_from_same_dir = \
            os.path.dirname(filepath1) == os.path.dirname(filepath2)

        images_in_same_things_dir = [
            file_in_dir(filepath, same_things_dirpath)
            for filepath in filepaths
        ]

        if [True, True] == images_in_same_things_dir and
            images_from_same_dir(filepath1, filepath2):
            return True

        return False


###############################################################################

    def perform_matching(self):
        # Extract features
        logging.info("Beginning feature extraction")
        feature_data = self.matching_alg.get_features(self.image_list_file)
        logging.info("Feature extraction is complete")
        logging.info(f"Feature data is of type {type(feature_data)}")
        total_num = len(list(feature_data.keys()))
        false_positives = 0
        false_negatives = 0
        best_fit = None
        best_score = 0
        best_shift = None

        # For every image, compute the number of other images it matches with.
        for query_image_path, query_data in feature_data.items():
            # Load the image in grayscale.
            query_img = cv2.imread(query_image_path, 0)
            query_img = cv2.resize(query_img,
                                   (config.IM_HEIGHT, config.IM_WIDTH))

            if self.matching_alg.provides_keypoints():
                query_hist = self.calculate_histogram(query_img)
                (query_kp, query_des) = \
                    self.matching_alg.get_kp_desc_pair(query_data)

            for target_image_path, target_data in feature_data.items():
                if query_image_path == target_image_path:
                    continue  # Skip matching with itself
                logging.info(
                    f"Now comparing {query_image_path} with {target_image_path}")

                target_img = cv2.imread(target_image_path, 0)
                target_img = cv2.resize(target_img, (config.IM_HEIGHT, config.IM_WIDTH))

                if self.matching_alg.provides_keypoints():
                    target_hist = self.calculate_histogram(target_img)
                    (target_kp, target_des) = \
                        self.matching_alg.get_kp_desc_pair(target_data)

                    # Convert descriptors to type float32, which is required by FLANN
                    query_des = query_des.astype(np.float32)
                    target_des = target_des.astype(np.float32)

                    score, shift = self.compute_match_score(
                        (query_kp, query_des, query_hist, query_img),
                        (target_kp, target_des, target_hist, target_img),
                    )
                else:
                    euclidean_dist = \
                        scipy.partial.distance.euclidean(query_data, target_data)
                    score = euclidean_dist


                if score is None and
                    images_are_of_same_object(query_image_path, target_image_path):
                    false_negatives += 1

                if score is not None and
                    not images_are_of_same_object(query_image_path, target_image_path):
                    false_positives += 1

                if score is not None and score > best_score:
                    best_score = score
                    best_shift = shift
                    best_fit = (query_image_path, target_image_path)

            logging.info(f"BEST FIT IS: {best_fit}")

        logging.info(f"False positive is: {false_positives/total_num/100}%")
        logging.info(f"False negative is: {false_negatives/total_num/100}%")

###############################################################################

def main():
    logging.basicConfig(level=logging.INFO)

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

    print(tf.sysconfig.get_build_info())
    print(tf.config.list_physical_devices('GPU'))

    parser = \
        argparse.ArgumentParser(
            description="Benchmarking tool for feature matching algorithms")
    parser.add_argument(
        "--type", choices=["false_pos", "false_neg"], required=True,
        help="Specify the type of benchmark to run")

    matching_algos_list = ["surf", "d2net", "orb", "vgg16"]
    parser.add_argument(
        "--matching_algo", choices=matching_algos_list, required=True,
        help="Specify the matching algorithm to use for the benchmark")
    args = parser.parse_args()

    if args.type == "false_pos":
        benchmark_type = BenchmarkType.FALSE_POS
        image_list_file = "fp_image_list.txt"
        logging.info("Performing false positive benchamark")
    elif args.type == "false_neg":
        benchmark_type = BenchmarkType.FALSE_NEG
        image_list_file = "fn_image_list.txt"
        logging.info("Performing false negative benchamark")

    if args.matching_algo == "surf":
        matching_algo = matching_algos.SURF()
    elif args.matching_algo == "d2net":
        model_file = "models/d2_tf.pth"
        matching_algo = matching_algos.D2Net(model_file)
    elif args.matching_algo == "orb":
        matching_algo = matching_algos.ORB()
    elif args.matching_algo == "vgg16":
        matching_algo = matching_algos.VGG16()

    b = benchmarker(benchmark_type, image_list_file, matching_algo)
    b.perform_matching()

if __name__ == "__main__":
    main()
