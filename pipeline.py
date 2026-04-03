import cv2
import numpy as np

def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = np.uint8(255 * (gray/255)**0.5)
    enhanced = cv2.equalizeHist(gamma)
    return enhanced