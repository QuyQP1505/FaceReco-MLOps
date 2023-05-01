import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog


def extract_features(img, resize_shape=(120, 120)):
    
    # Convert the image to grayscale.
    img = cv2.resize(img, resize_shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the LBP feature vector for the grayscale image.
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    hist_lbp,_ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist_lbp = hist_lbp.astype("float") / np.sum(hist_lbp)
    
    # Compute the HOG feature vector for the grayscale image.
    orientations = 9
    pixels_per_cell = (8, 8)
    cells_per_block = (3, 3)
    features_hog,_ = hog(gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, block_norm="L2-Hys", visualize=True)
    
    # Concatenate the LBP and HOG feature vectors into a single feature vector.
    features = np.concatenate((hist_lbp, features_hog))
    
    return features