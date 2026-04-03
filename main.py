import customtkinter as ctk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1100x650")
app.title("Smart Image Enhancement System")

img = None
gray = None

# ---------------- LOAD IMAGE ----------------
def load_image():
    global img, gray
    path = ctk.filedialog.askopenfilename()
    if not path:
        return
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    show_image(img)

# ---------------- DISPLAY ----------------
def show_image(image):
    image = cv2.resize(image, (400, 400))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    img_tk = ctk.CTkImage(image, size=(400, 400))
    image_label.configure(image=img_tk)

# ---------------- SAMPLING ----------------
def sampling():
    scales = [0.25, 0.5, 1, 1.5, 2]
    plt.figure("Sampling")
    for i, s in enumerate(scales):
        resized = cv2.resize(gray, None, fx=s, fy=s)
        plt.subplot(2,3,i+1)
        plt.imshow(resized, cmap='gray')
        plt.title(f"{s}x")
        plt.axis('off')
    plt.show()

# ---------------- QUANTIZATION ----------------
def quantization():
    def q(bits):
        levels = 2**bits
        return (gray // (256//levels)) * (256//levels)

    plt.figure("Quantization")
    for i, b in enumerate([8,4,2]):
        plt.subplot(1,3,i+1)
        plt.imshow(q(b), cmap='gray')
        plt.title(f"{b}-bit")
        plt.axis('off')
    plt.show()

# ---------------- TRANSFORMATIONS ----------------
def transformations():
    angles = [30,45,60,90,120,150,180]
    plt.figure("Transformations")
    for i, a in enumerate(angles):
        M = cv2.getRotationMatrix2D((gray.shape[1]/2, gray.shape[0]/2), a, 1)
        rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]))
        plt.subplot(3,3,i+1)
        plt.imshow(rot, cmap='gray')
        plt.title(f"{a}°")
        plt.axis('off')
    plt.show()

# ---------------- INTENSITY ----------------
def intensity():
    neg = 255 - gray
    log = np.uint8(255*np.log1p(gray)/np.log(256))
    g1 = np.uint8(255*(gray/255)**0.5)
    g2 = np.uint8(255*(gray/255)**1.5)

    imgs = [neg, log, g1, g2]
    titles = ["Negative","Log","Gamma 0.5","Gamma 1.5"]

    plt.figure("Intensity")
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# ---------------- HISTOGRAM ----------------
def histogram():
    eq = cv2.equalizeHist(gray)

    plt.figure("Histogram")
    plt.subplot(2,2,1), plt.imshow(gray, cmap='gray'), plt.title("Original")
    plt.subplot(2,2,2), plt.hist(gray.ravel(),256), plt.title("Hist")

    plt.subplot(2,2,3), plt.imshow(eq, cmap='gray'), plt.title("Equalized")
    plt.subplot(2,2,4), plt.hist(eq.ravel(),256), plt.title("Equalized Hist")
    plt.show()

# ---------------- PIPELINE ----------------
def enhance():
    gamma = np.uint8(255*(gray/255)**0.5)
    enhanced = cv2.equalizeHist(gamma)
    show_image(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))

# ---------------- LAYOUT ----------------

# Sidebar
sidebar = ctk.CTkFrame(app, width=200)
sidebar.pack(side="left", fill="y")

ctk.CTkLabel(sidebar, text="MENU", font=("Arial", 20)).pack(pady=20)

ctk.CTkButton(sidebar, text="Load Image", command=load_image).pack(pady=10)
ctk.CTkButton(sidebar, text="Sampling", command=sampling).pack(pady=10)
ctk.CTkButton(sidebar, text="Quantization", command=quantization).pack(pady=10)
ctk.CTkButton(sidebar, text="Transformations", command=transformations).pack(pady=10)
ctk.CTkButton(sidebar, text="Intensity", command=intensity).pack(pady=10)
ctk.CTkButton(sidebar, text="Histogram", command=histogram).pack(pady=10)
ctk.CTkButton(sidebar, text="Enhance", command=enhance).pack(pady=20)

# Main Display
main_frame = ctk.CTkFrame(app)
main_frame.pack(side="right", expand=True, fill="both")

image_label = ctk.CTkLabel(main_frame, text="Load Image", width=400, height=400)
image_label.pack(pady=50)

app.mainloop()