import numpy as np
from os import path
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_erosion, binary_closing, label

class RetinalFeatureExtractor:
    """
    This class helps extract features from a retinal image for the purpose of later
    using these feature points for 2D registration.
    """

    @staticmethod
    def segment_blood_vessel(image: np.ndarray):

        # worked according to this method: https://iopscience.iop.org/article/10.1088/1742-6596/1376/1/012023/pdf

        # crop bottom label

        image = image[:image.shape[0] - 100, :, :]

        # Get green channel
        segmented_image = image[:, :, 1]

        # take out cancer
        segmented_image[450:1000, 500:1200] = 255

        # get complement of green channel

        segmented_image = np.max(segmented_image) - segmented_image
        plt.imshow(segmented_image, cmap='gray')
        plt.show()
        # use contrast limited adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(8, 8))
        segmented_image = clahe.apply(segmented_image)

        # Otsu thresholding
        segmented_image = median_filter(segmented_image, 7)
        ret3,segmented_image = cv2.threshold(segmented_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        # Morphological operations
        segmented_image = binary_opening(segmented_image, disk(3), iterations=2)
        segmented_image = binary_opening(segmented_image, disk(1), iterations=5)
        segmented_image = binary_erosion(segmented_image, disk(3), iterations=2)
        segmented_image = median_filter(segmented_image, footprint=disk(3))
        segmented_image, components = label(segmented_image)
        segmented_image = segmented_image == np.argmax(np.bincount(segmented_image.flat)[1:]) + 1
        plt.imshow(segmented_image, cmap='gray')
        plt.show()

    @staticmethod
    def find_retina_features(image):
        pass


if __name__ == '__main__':
    dataset_path = "/home/edan/Desktop/HighRad/Exercises/data/Targil2_data_2018-20230315T115832Z-001/Targil2_data_2018"
    bl = "BL01.tif"
    bl_path = path.join(dataset_path, bl)
    image = Image.open(bl_path)
    im_arr = np.array(image)
    RetinalFeatureExtractor.segment_blood_vessel(im_arr)