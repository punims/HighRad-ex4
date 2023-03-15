from typing import Tuple
from os.path import join
from utils import getPoints
import numpy as np
from PIL import Image
from numpy import ndarray
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import cv2


class RigidTransformFinder:
    """
    This class is responsible for the rigid transform registration of two 2d images.
    """

    def __init__(self, bl_im_path: str, fu_im_path):
        """
        Constructor
        Parameters
        ----------
        bl_im_path: baseline image path
        fu_im_path: follow-up image path
        """
        self.fu_im_path = fu_im_path
        self.bl_im_path = bl_im_path

    def read_images(self) -> tuple[ndarray, ndarray]:
        """
        returns bl and fu images if they exist
        Returns
        -------

        """

        bl = Image.open(self.bl_im_path)
        bl = np.array(bl)

        fu = Image.open(self.fu_im_path)
        fu = np.array(fu)

        return bl, fu

    def plot_points_on_images(self) -> None:
        """
        reads the images and plots points from getPoints onto them.
        getPoints returns the points for bl and then fu.
        Returns
        -------

        """
        bl, fu = self.read_images()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(bl)
        ax1.set_title('baseline')
        ax2.imshow(fu)
        ax2.set_title('follow-up')

        points1, points2 = getPoints('no_outliers')

        for i in range(len(points1)):
            ax1.scatter(points1[i][0], points1[i][1])
            ax1.annotate(i, (points1[i][0], points1[i][1]), color='red',
                         fontsize=12)

        for i in range(len(points2)):
            ax2.scatter(points2[i][0], points2[i][1])
            ax2.annotate(i, (points2[i][0], points2[i][1]), color='red',
                         fontsize=12)

        plt.show()

    @staticmethod
    def calc_point_based_reg(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Given two nx2 arrays bl and fu, each row being a 2d point that corresponds to the
        point in the other matrix, find the transformation that minimizes the least squared distance
        between the two using SVD to find the rotation and translation and then returns the 3x3 rigid
        transformation matrix.
        The transformation matrix moves p onto q
        Parameters
        ----------
        bl_points
        fu_points

        Returns
        -------

        """

        # compute centroids and centered vectors
        p_centroid = np.mean(p, axis=0)
        q_centroid = np.mean(q, axis=0)

        p_centered = p - p_centroid
        q_centered = q - q_centroid

        # compute rotation using svd of covariance matrix
        cov_matrix = p_centered.T @ q_centered
        u, s, vh = np.linalg.svd(cov_matrix)
        det_v_ut = np.linalg.det(vh.T @ u.T)
        n = cov_matrix.shape[0]
        middle_rotation_component = np.eye(n)
        middle_rotation_component[-1, -1] = det_v_ut
        R = vh.T @ middle_rotation_component @ u.T

        # compute translation using rotation and centroids
        t = q_centroid - R.dot(p_centroid)

        # compute and return 3x3 rigid matrix
        # R | t
        # 0 | 1
        transformation_matrix = np.hstack((R, t[:, np.newaxis]))
        transformation_matrix = np.vstack((transformation_matrix, [0, 0, 1]))

        return R, t, transformation_matrix


    @staticmethod
    def calc_dist(bl_points: np.ndarray, fu_points: np.ndarray, rigid_matrix: np.ndarray) -> None:
        """
        Calculates the Root Mean Squared Error between the points bl,1 and rigid_matrix * fu,1
        Note that the rigid matrix is a 3x3 matrix so that we need to use homogenous coordinates
        Parameters
        ----------
        bl_points nx2 points matrix
        fu_points nx2 points matrix
        rigid_matrix 3x3 rigid transformation matrix

        Returns
        -------

        """

        # homogenous coordinates
        bl_homo = np.vstack((bl_points.T, np.ones((bl_points.shape[0], 1)).T))
        fu_homo = np.vstack((fu_points.T, np.ones((bl_points.shape[0], 1)).T))

        transformed_fu = rigid_matrix @ fu_homo
        rmse = mean_squared_error(bl_homo, transformed_fu)
        print(rmse)

    def register(self, bl_points: np.ndarray, fu_points: np.ndarray) -> None:
        """
        Registers fu onto bl
        Parameters
        ----------
        fu_points : follow up points
        bl_points : baseline points

        Returns
        -------

        """

        bl, fu = self.read_images()
        _, _, rigid_transformation = RigidTransformFinder.calc_point_based_reg(fu_points, bl_points)
        rows, cols, _ = bl.shape
        transformed_fu = cv2.warpAffine(fu, rigid_transformation[:2, :], (cols, rows))
        bl = Image.fromarray(bl).convert("RGBA")
        transformed_fu = Image.fromarray(transformed_fu).convert("RGBA")

        new_img = Image.blend(bl, transformed_fu, 0.5)
        new_img.save("new.png", "PNG")


def run():
    dataset_path = "/home/edan/Desktop/HighRad/Exercises/data/Targil2_data_2018-20230315T115832Z-001/Targil2_data_2018"
    bl = "BL01.tif"
    fu = "FU01.tif"

    bl = join(dataset_path, bl)
    fu = join(dataset_path, fu)
    registrator = RigidTransformFinder(bl, fu)
    # registrator.plot_points_on_images()

    bl_points, fu_points = getPoints('no_outliers')
    # R, t, rigid_transformation = registrator.calc_point_based_reg(fu_points, bl_points)
    # registrator.calc_dist(bl_points, fu_points, rigid_transformation)
    #
    registrator.register(bl_points, fu_points)




if __name__ == '__main__':
    run()