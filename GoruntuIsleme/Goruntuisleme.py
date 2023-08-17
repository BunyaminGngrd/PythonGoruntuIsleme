import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from scipy import ndimage

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.kernel_size = tk.Scale(self, from_=1, to=20, orient=tk.HORIZONTAL, label='Kernel Size')
        self.kernel_size.set(3)
        self.kernel_size.pack()

        self.sigma = tk.Scale(self, from_=0, to=10, orient=tk.HORIZONTAL, resolution=0.1, label='Sigma')
        self.sigma.pack()

        self.run_button = tk.Button(self)
        self.run_button["text"] = "Run"
        self.run_button["command"] = self.run
        self.run_button.pack()

    def run(self):
        file_path = filedialog.askopenfilename()
        img = np.array(Image.open(file_path).convert('L'))

        kernel_size = self.kernel_size.get()
        sigma = self.sigma.get()

        # Gaussian filtering
        img_gauss = ndimage.gaussian_filter(img, sigma=(sigma, sigma), order=0, truncate=kernel_size)

        # Median filtering
        img_median = ndimage.median_filter(img, size=kernel_size)

        # Uniform filtering
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        img_uniform = ndimage.convolve(img, kernel)

        # Sobel filtering
        dx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        dy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        img_dx = ndimage.convolve(img, dx_kernel)
        img_dy = ndimage.convolve(img, dy_kernel)
        img_sobel = np.sqrt(img_dx**2 + img_dy**2)

        # Display images
        fig, axes = plt.subplots(1, 4, figsize=(10, 5))
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original')
        axes[1].imshow(img_gauss, cmap='gray')
        axes[1].set_title('Gaussian')
        axes[2].imshow(img_median, cmap='gray')
        axes[2].set_title('Median')
        axes[3].imshow(img_sobel, cmap='gray')
        axes[3].set_title('Sobel')

        # Save figure
        plt.savefig('smoothed_image.png')
        plt.show()

root = tk.Tk()
app = Application(master=root)
app.mainloop()
