import cv2
import numpy as np
from extract_d2net import (
    extract_features,
)  # Assuming the above code is saved in feature_extractor.py


def perform_matching(image_list_file, model_file):
    # Extract features
    feature_data = extract_features(image_list_file, model_file)

    # Create FLANN object
    flann = cv2.FlannBasedMatcher_create()

    # Assume you want to match features of the first image with all others
    query_image_path = list(feature_data.keys())[0]
    query_descriptors = feature_data[query_image_path]["descriptors"]

    for target_image_path, target_data in feature_data.items():
        if query_image_path == target_image_path:
            continue  # Skip matching with itself

        target_descriptors = target_data["descriptors"]

        # Convert descriptors to type float32, which is required by FLANN
        query_descriptors = query_descriptors.astype(np.float32)
        target_descriptors = target_descriptors.astype(np.float32)

        # Perform matching
        matches = flann.knnMatch(query_descriptors, target_descriptors, k=2)

        # ... process matches ...


if __name__ == "__main__":
    perform_matching("image_list.txt", "models/d2_tf.pth")
