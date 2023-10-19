import cv2
import numpy as np
import config
from scipy import stats
import logging
from extract_d2net import (
    extract_features,
)  # Assuming the above code is saved in feature_extractor.py

class FeatureMatchingAlgorithm:
    def get_features(self, image_list_file: str):
        pass

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        pass

class SURF(FeatureMatchingAlgorithm):
    def get_features(self, image_list_file: str):
        file_to_kp_desc_map = {}
        surf = cv2.xfeatures2d.SURF_create()

        image_path_list = []
        with open(image_list_file, 'r') as file:
            image_path_list = [line.strip() for line in file.readlines()]

        for image_file in image_list_file:
            img = cv2.imread(query_image_path, 0)
            file_to_kp_desc_map[image_file] = surf.detectAndCompute(img, None)

        return file_to_kp_desc_map

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint.pt

class D2Net(FeatureMatchingAlgorithm):
    def __init__(self, model_file):
        self.model_file = model_file

    def get_features(self, image_list_file: str):
        return extract_features(image_list_file, self.model_file)

    @staticmethod
    def get_keypoint_coordinates(keypoint):
        return keypoint


# Compute false positives for images of different things
class fp_benchmarker:
    def __init__(self, image_list_file, matching_alg):
        self.flann = cv2.FlannBasedMatcher(config.INDEX_PARAMS, config.SEARCH_PARAMS)
        self.image_list_file = image_list_file
        self.matching_alg = matching_alg

    # Finds the median bin of a histogram
    def hist_median(self, hist):
        total_samples = hist.sum()
        half_samples = total_samples // 2
        s = 0
        for i in range(len(hist)):  # changed xrange to range
            s += hist[i]
            if s > half_samples:
                return i

    def extract_good_matches(self, matches):
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < (config.DISTANCE_THRESH * n.distance):
                good.append(m)
        return good

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

        # logging.DEBUG(f"NUMBER OF GOOD MATCHES: {len(matches)}")
        # logging.DEBUG(f"HISTOGRAM CORRELATION: {hist_correlation}")
        # logging.DEBUG(f"MWN CORRELATION: {hist_mwn}")
        # logging.DEBUG(f"DCT CORRELATION: {dct_correl}")

        # Calculate match threshold based on the number of keypoints detected in the database image and the query image
        train_threshold = 0.1 * len(train_kp)
        query_threshold = 0.1 * len(query_kp)
        threshold = max(train_threshold, query_threshold)

        # logging.DEBUG(f"THRESHOLD: {threshold}")

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

    def calculate_histogram(image):
        """
        Calculate a histogram of a single-channel (greyscale) image using 256
        bins, one for each possible value of a pixel. This histogram provides
        information about the distribution of pixel densities in the image.
        """
        return cv2.calcHist([image], [0], None, [256], [0, 256])

    def perform_matching(self):
        # Extract features
        logging.info("Beginning feature extraction")
        feature_data = self.matching_alg.get_features(self.image_list_file)
        logging.info("Feature extraction is complete")
        total_num = len(list(feature_data.keys()))
        positive_num = 0
        negative_num = 0
        best_fit = None
        best_score = 0
        best_shift = None

        # For every image, compute the number of other images it matches with.
        for query_image_path, query_data in feature_data.items():
            # Load the image in grayscale.
            query_img = cv2.imread(query_image_path, 0)
            query_img = cv2.resize(query_img, (config.IM_HEIGHT, config.IM_WIDTH))
            query_hist = calculate_histogram(query_img)
            query_des = query_data["descriptors"]
            query_kp = query_data["keypoints"]

            for target_image_path, target_data in feature_data.items():
                if query_image_path == target_image_path:
                    continue  # Skip matching with itself

                target_img = cv2.imread(target_image_path, 0)
                target_img = cv2.resize(target_img, (config.IM_HEIGHT, config.IM_WIDTH))
                target_hist = calculate_histogram(target_img)
                target_des = target_data["descriptors"]
                target_kp = target_data["keypoints"]

                # Convert descriptors to type float32, which is required by FLANN
                query_des = query_des.astype(np.float32)
                target_des = target_des.astype(np.float32)

                score, shift = self.compute_match_score(
                    (query_kp, query_des, query_hist, query_img),
                    (target_kp, target_des, target_hist, target_img),
                )

                if score is None:
                    negative_num += 1
                else:
                    positive_num += 1

                if score is not None and score > best_score:
                    best_score = score
                    best_shift = shift
                    best_fit = (query_image_path, target_image_path)

        logging.info(f"BEST FIT IS: {best_fit}")
        logging.info(f"False positive is: {(float)(positive_num/total_num/100)}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    image_list_file = "fp_image_list.txt"
    model_file = "models/d2_tf.pth"
    logging.info("Performing false positive benchamark")

    # d2net_alg = D2Net(model_file)
    surf_alg = SURF()

    benchmarker = fp_benchmarker(image_list_file, surf_alg)
    benchmarker.perform_matching()
