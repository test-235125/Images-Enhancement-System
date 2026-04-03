import numpy as np

def intensity(gray):
    gray = gray.astype(float)

    negative = 255 - gray

    # SAFE LOG TRANSFORM
    log = 255 * np.log1p(gray) / np.log(256)
    log = np.uint8(np.clip(log, 0, 255))

    gamma05 = np.uint8(255 * (gray/255)**0.5)
    gamma15 = np.uint8(255 * (gray/255)**1.5)

    return {
        "negative": negative.astype(np.uint8),
        "log": log,
        "gamma05": gamma05,
        "gamma15": gamma15
    }