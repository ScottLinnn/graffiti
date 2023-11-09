import argparse
import cv2
import numpy as np
from scipy import stats
import logging
from enum import Enum
import os
import torch
import tensorflow as tf
import math
import sys
import time

import config
import feature_extraction_algos

print_matches = False

###############################################################################

# Compute false positives for images of different things
class benchmarker:
    def __init__(self, image_list_file, feature_extraction_algo):
        self.flann = cv2.FlannBasedMatcher(config.INDEX_PARAMS, config.SEARCH_PARAMS)
        self.image_list_file = image_list_file
        self.feature_extraction_algo = feature_extraction_algo
        self.get_image_data()

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

    def get_image_data(self):
        self.image_data = dict()
        image_path_list = \
            feature_extraction_algos.get_image_list_from_file(self.image_list_file)
        for image_file in image_path_list:
            img = cv2.imread(image_file, 0)
            img = cv2.resize(img, (config.IM_WIDTH, config.IM_HEIGHT))
            self.image_data[image_file] = img

###############################################################################

    def extract_good_matches(self, matches):
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < (config.DISTANCE_THRESH * n.distance):
                good.append(m)
        return good

###############################################################################

    # Returns a match score between two images
    def compute_match_score(self, query_data, train_data, query_image, train_image):
        query_kp, query_des, query_hist, query_img = query_data
        train_kp, train_des, train_hist, train_img = train_data

        score = 0

        if min(len(query_kp), len(train_kp), 2) < 2:
            return None, None, { 'matchThresholdMet': False }
        matches = \
            self.extract_good_matches(
                self.flann.knnMatch(query_des, train_des, k=2))

        sorted_matches = sorted(matches, key = lambda x:x.distance)
        # print(type(query_kp))
        # print(type(train_kp))
        # print(query_kp)
        # print(train_kp)
        # print([tuple(row[:2].tolist()) for row in query_kp])

        if print_matches:
            if isinstance(query_kp[0], cv2.KeyPoint):
                img = cv2.drawMatches(
                        query_image, query_kp, train_image, train_kp, sorted_matches[:50],
                     None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            else:
                q_k = [row[:2].tolist() for row in query_kp]
                t_k = [row[:2].tolist() for row in train_kp]
                img = cv2.drawMatches(
                        query_image, [cv2.KeyPoint(x, y, 5) for (x,y) in q_k], train_image, [cv2.KeyPoint(x, y, 5) for (x,y) in t_k], sorted_matches[:50],
                    None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Matches", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # print([(x,y) for (x,y) in q_k])

        # img = cv2.drawKeypoints(query_image, query_kp, 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # img = cv2.drawKeypoints(query_image, [cv2.KeyPoint(x,y,5) for (x,y) in q_k], 0, (0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow("Matches", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite('/app/tmp/orb_rushm_img1.jpg', img)

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

        def DCT(img):
            imf = np.float32(img) / 255.0  # Float conversion/scale
            dst = cv2.dct(imf)  # Calculate the dct
            return dst

        # Find DCT correlation
        img1 = DCT(query_img)
        img2 = DCT(train_img)
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
            return None, None, { 'matchThresholdMet': False }
        else:
            score += len(matches)

        # calculate the relative displacement between two group of key points
        shift_xs = []
        shift_ys = []
        for m in matches:
            k_q = query_kp[m.queryIdx]
            k_t = train_kp[m.trainIdx]

            k_q_coord = self.feature_extraction_algo.get_keypoint_coordinates(k_q);
            k_t_coord = self.feature_extraction_algo.get_keypoint_coordinates(k_t);
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
            logging.debug(f"Did not meet hist test, {hist_test_passes}/3")
            return None, None, { 'histTestPassed': False, 'matchThresholdMet': True }

        logging.debug(f"SCORE IS: {score}")
        return score, (shift_x, shift_y), dict()

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
        filepath1 = os.path.abspath(filepath1)
        filepath2 = os.path.abspath(filepath2)

        filepaths = [filepath1, filepath2]

        file_in_dir = \
            (lambda filepath, dirpath:
                os.path.commonpath([filepath, dirpath]) == dirpath)

        if (not file_in_dir(filepath1, different_things_dirpath) and
            not file_in_dir(filepath1, same_things_dirpath)):
               raise Exception("Image not placed correctly: " + filepath1)

        if (not file_in_dir(filepath2, different_things_dirpath) and
            not file_in_dir(filepath2, same_things_dirpath)):
               raise Exception("Image not placed correctly: " + filepath2)

        images_from_same_dir = \
            os.path.dirname(filepath1) == os.path.dirname(filepath2)

        images_in_same_things_dir = [
            file_in_dir(filepath, same_things_dirpath)
            for filepath in filepaths
        ]

        if [True, True] == images_in_same_things_dir and images_from_same_dir:
            logging.debug(f"Images of same object: {filepath1}, {filepath2}")
            return True

        return False


###############################################################################

    def perform_matching(self):
        # Extract features
        logging.info("Beginning feature extraction")
        feature_data = self.feature_extraction_algo.get_features(self.image_list_file)
        logging.info("Feature extraction is complete")
        logging.info(f"Feature data is of type {type(feature_data)}")

        # histogram_data = dict()
        # for query_image_path, _ in feature_data.items():
        #     histogram_data[query_image_path] = self.calculate_histogram(query_img)

        num_images = len(list(feature_data.keys()))
        total_num = math.comb(num_images, 2)
        logging.info(f"{num_images=} images were specified for a total of "
                     f"{total_num=} matches")

        false_positives = 0
        false_negatives = 0
        # The number of matches reported by the algorithm that were correct.
        true_positives = 0
        # The number of matches that are actually true matches.
        ground_truth_positives = 0
        best_fit = None
        best_score = 0
        best_shift = None
        num_match_pairs = 0

        false_neg_match_threshold_unmet = 0
        false_neg_hist_test_failed = 0

        # For every image, compute the number of other images it matches with.
        for query_image_path, query_data in feature_data.items():
            query_img = self.image_data[query_image_path]

            if self.feature_extraction_algo.provides_keypoints():
                query_hist = self.calculate_histogram(query_img)
                (query_kp, query_des) = \
                    self.feature_extraction_algo.get_kp_desc_pair(query_data)

            for target_image_path, target_data in feature_data.items():
                if query_image_path == target_image_path:
                    continue  # Skip matching with itself
                if query_image_path > target_image_path:
                    continue
                num_match_pairs += 1
                logging.debug(
                    f"Now comparing {query_image_path} with {target_image_path}")

                target_img = self.image_data[target_image_path]

                if self.feature_extraction_algo.provides_keypoints():
                    target_hist = self.calculate_histogram(target_img)
                    (target_kp, target_des) = \
                        self.feature_extraction_algo.get_kp_desc_pair(target_data)

                    if query_des is not None and target_des is not None:
                        # Convert descriptors to type float32, which is required by FLANN
                        query_des = query_des.astype(np.float32)
                        target_des = target_des.astype(np.float32)

                        score, shift, reason = self.compute_match_score(
                            (query_kp, query_des, query_hist, query_img),
                            (target_kp, target_des, target_hist, target_img),
                            query_img, target_img,
                        )
                    else:
                        score = None
                else:
                    euclidean_dist = \
                        scipy.partial.distance.euclidean(query_data, target_data)
                    score = euclidean_dist


                if score is not None and score > best_score:
                    best_score = score
                    best_shift = shift
                    best_fit = target_image_path

                if self.images_are_of_same_object(query_image_path, target_image_path):
                    ground_truth_positives += 1
                    if score is None:
                        logging.info(f"False negative: {query_image_path} {target_image_path}")
                        false_negatives += 1
                        if not reason['matchThresholdMet']:
                            false_neg_match_threshold_unmet += 1
                        elif not reason['histTestPassed']:
                            false_neg_hist_test_failed += 1
                    else:
                        true_positives += 1
                elif score is not None:
                    logging.info(f"False positive: {query_image_path} {target_image_path}")
                    false_positives += 1

            logging.debug(f"BEST FIT IS: {query_image_path} {best_fit}")

        logging.info(f"{false_positives=}, {false_negatives=}, {total_num=}")
        logging.info(f"False positive rate is: {(false_positives/total_num)*100}%")
        logging.info(f"False negative rate is: {(false_negatives/total_num)*100}%")
        # Of all the instances the model predicted as positive, how many were actually correct?
        logging.info(f"Precision: {(true_positives/(true_positives + false_positives))*100}")
        # Of all the actual positive instances, how many did the model correctly predict as positive?
        logging.info(f"Recall: {(true_positives/(true_positives + false_negatives))*100}")
        logging.info(f"Found {num_match_pairs} match pairs")
        logging.info(f"{false_neg_match_threshold_unmet=}, {false_neg_hist_test_failed=}")
        logging.info(f"{true_positives=}, {false_positives=}, {false_negatives=}")
        logging.info(f"{ground_truth_positives=}, positive find rate: {(true_positives/ground_truth_positives)*100}%")

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

    feature_extraction_algos_list = ["surf", "d2net", "orb", "vgg16", "brief", "sift", "kaze", "akaze"]
    parser.add_argument(
        "--feature_extraction_algo", choices=feature_extraction_algos_list, required=True,
        help="Specify the matching algorithm to use for the benchmark")
    args = parser.parse_args()

    image_list_file = "image_list.txt"

    if args.feature_extraction_algo == "surf":
        feature_extraction_algo = feature_extraction_algos.SURF()
    elif args.feature_extraction_algo == "d2net":
        model_file = "models/d2_tf.pth"
        feature_extraction_algo = feature_extraction_algos.D2Net(model_file)
    elif args.feature_extraction_algo == "orb":
        feature_extraction_algo = feature_extraction_algos.ORB()
    elif args.feature_extraction_algo == "vgg16":
        feature_extraction_algo = feature_extraction_algos.VGG16()
    elif args.feature_extraction_algo == "brief":
        feature_extraction_algo = feature_extraction_algos.BRIEF()
    elif args.feature_extraction_algo == "sift":
        feature_extraction_algo = feature_extraction_algos.SIFT()
    elif args.feature_extraction_algo == "kaze":
        feature_extraction_algo = feature_extraction_algos.KAZE()
    elif args.feature_extraction_algo == "akaze":
        feature_extraction_algo = feature_extraction_algos.AKAZE()

    b = benchmarker(image_list_file, feature_extraction_algo)
    start_time = time.time()
    b.perform_matching()
    logging.info(f"Benchmark took {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()
