import cv2

def image_acquisition(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Resolution:", gray.shape)
    print("Data Type:", gray.dtype)
    print("Matrix:\n", gray[:5, :5])

    return img, gray