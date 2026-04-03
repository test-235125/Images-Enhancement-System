import cv2
import matplotlib.pyplot as plt
import os

def histogram(gray):

    # CREATE FOLDER IF NOT EXISTS ✅
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.hist(gray.ravel(), 256)
    plt.title("Original Histogram")
    plt.savefig("results/original_hist.png")
    plt.show()

    eq = cv2.equalizeHist(gray)

    plt.figure()
    plt.hist(eq.ravel(), 256)
    plt.title("Equalized Histogram")
    plt.savefig("results/equalized_hist.png")
    plt.show()

    return eq