import cv2
import numpy as np

def sampling(gray):
    scales = [0.25, 0.5, 1, 1.5, 2]
    results = {}

    for s in scales:
        resized = cv2.resize(gray, None, fx=s, fy=s)
        results[f"scale_{s}"] = resized

    return results

def quantization(gray):
    def quantize(img, bits):
        levels = 2**bits
        return (img // (256//levels)) * (256//levels)

    return {
        "8bit": quantize(gray, 8),
        "4bit": quantize(gray, 4),
        "2bit": quantize(gray, 2)
    }