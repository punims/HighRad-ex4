import numpy as np
from PIL import Image
from RetinalFeatureExtractor import RetinalFeatureExtractor
from RigidTransformFinder import RigidTransformFinder


class RetinaRegistration:
    """
    This class takes as input two retina images, bl and fu and
    applies a registration between both images using RetinalFeatureExtractor
    to get the points of interest and RigidTransformFinder to find the actual transformation
    """

    @staticmethod
    def feature_registration(bl_path: str, fu_path: str) -> np.ndarray:
        """
        registers bl and fu using feature finding:
        1. Finds features in both images using SIFT
        2. Matches features from both images using brute force
        3. Picks best matching points for a registration using Ransac
        4. Applies the transformation and returns the registered images
        Returns
        -------

        """
        bl_im = Image.open(bl_path)
        bl_arr = np.array(bl_im)

        fu_im = Image.open(fu_path)
        fu_arr = np.array((fu_im))

        # Extract key points
        bl_keypoints = RetinalFeatureExtractor.find_retina_features(bl_arr)
        fu_keypoints = RetinalFeatureExtractor.find_retina_features(fu_arr)

        # Get a match of the key points
        bl_keypoints, fu_keypoints = RetinaRegistration.match_keypoints(bl_keypoints, fu_keypoints)

        # register and plot
        registrator = RigidTransformFinder(bl_path, fu_path)
        transformation, indices = registrator.calc_robust_point_based_reg(bl_keypoints, fu_keypoints)
        registrator.plot_with_outliers(bl_keypoints, fu_keypoints, indices)


    @staticmethod
    def match_keypoints(k1: list, k2: list) -> tuple[list, list]:
        """
        given key points k1 and k2 finds the best matches between both images.
        Parameters
        ----------
        k1
        k2

        Returns
        -------

        """
        pass

    @staticmethod
    def cross_correlation_registration(bl_path: str, fu_path: str):
        """
        registers bl and fu using cross correlation:
        1. extract segmentations of blood vessels of both bl and fu
        2. rotate one segmentation slightly
        3. finds the max cross-correlation
        4. repeats 1-3 and take the rotation that gives the global maxima.
        Parameters
        ----------
        bl_path
        fu_path

        Returns
        -------

        """
        pass

