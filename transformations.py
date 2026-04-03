import cv2
import numpy as np

def transformations(gray):
    rows, cols = gray.shape
    results = {}

    angles = [30,45,60,90,120,150,180]
    for a in angles:
        M = cv2.getRotationMatrix2D((cols/2, rows/2), a, 1)
        results[f"rotate_{a}"] = cv2.warpAffine(gray, M, (cols, rows))

    # Translation
    M = np.float32([[1,0,50],[0,1,30]])
    results["translated"] = cv2.warpAffine(gray, M, (cols, rows))

    # Shear
    M = np.float32([[1,0.5,0],[0,1,0]])
    results["sheared"] = cv2.warpAffine(gray, M, (cols, rows))

    return results