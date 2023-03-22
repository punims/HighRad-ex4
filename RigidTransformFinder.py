from os.path import join
from utils import getPoints, ransac
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

    def plot_points_on_images(self, points1, points2) -> None:
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

        for i in range(len(points1)):
            ax1.scatter(points1[i][0], points1[i][1])
            ax1.annotate(i, (points1[i][0], points1[i][1]), color='red',
                         fontsize=12)

        for i in range(len(points2)):
            ax2.scatter(points2[i][0], points2[i][1])
            ax2.annotate(i, (points2[i][0], points2[i][1]), color='red',
                         fontsize=12)

        plt.show()

    def plot_with_outliers(self, points1, points2, indices):
        """
        Plots 2 images side by side, marking all indices within indices as inliers with one color
        and another color for all other indices.
        Parameters
        ----------
        points1
        points2
        indices

        Returns
        -------

        """

        bl, fu = self.read_images()
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(bl)
        ax1.set_title('baseline')
        ax2.imshow(fu)
        ax2.set_title('follow-up')
        inlier_color = 'red'
        outlier_color = 'blue'

        for i in range(len(points1)):

            if i in indices:
                color = inlier_color
            else:
                color = outlier_color
            ax1.scatter(points1[i][0], points1[i][1], color=color)
            ax2.scatter(points2[i][0], points2[i][1], color=color)
            ax1.annotate(i, (points1[i][0], points1[i][1]), color='black',
                         fontsize=12)
            ax2.annotate(i, (points2[i][0], points2[i][1]), color='black',
                         fontsize=12)

        plt.show()


    @staticmethod
    def calc_point_based_reg(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Given two nx2 arrays bl and fu, each row being a 2d point that corresponds to the
        point in the other matrix, find the transformation that minimizes the least squared distance
        between the two using SVD to find the rotation and translation and then returns the 3x3 rigid
        transformation matrix.
        The transformation matrix moves p onto q, the returned matrix is the inverse which moves q onto p


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

        return np.linalg.inv(transformation_matrix).T


    @staticmethod
    def calc_dist(bl_points: np.ndarray, fu_points: np.ndarray, rigid_matrix: np.ndarray) -> float:
        """
        Calculates the Root Mean Squared Error between the points bl,1 and fu,1 * rigid_matrix
        Note that the rigid matrix is a 3x3 matrix so that we need to use homogenous coordinates
        Parameters and prints it. returns the n length vector of the residual distances
        ----------
        bl_points nx2 points matrix
        fu_points nx2 points matrix
        rigid_matrix 3x3 rigid transformation matrix

        Returns residual distances of rmse
        -------

        """

        # homogenous coordinates
        bl_homo = np.hstack((bl_points, np.ones((bl_points.shape[0], 1))))
        fu_homo = np.hstack((fu_points, np.ones((bl_points.shape[0], 1))))

        transformed_fu = fu_homo @ rigid_matrix
        rmse = mean_squared_error(bl_homo, transformed_fu)
        return np.sqrt(np.sum(np.square(transformed_fu - bl_homo), axis=1))

    def register(self, bl_points: np.ndarray, fu_points: np.ndarray, transform: np.ndarray = None) -> None:
        """
        Registers fu onto bl
        Parameters
        ----------
        fu_points : follow up points
        bl_points : baseline points

        Returns
        -------

        """

        # Warp image
        bl, fu = self.read_images()
        if transform is not None:
            transform = RigidTransformFinder.calc_point_based_reg(bl_points, fu_points)
        rows, cols, _ = bl.shape
        transformed_fu = cv2.warpAffine(fu, transform[:, :2].T, (cols, rows))

        # Create overlay of fl onto bl and save
        bl = Image.fromarray(bl).convert("RGBA")
        transformed_fu = Image.fromarray(transformed_fu).convert("RGBA")
        new_img = Image.blend(bl, transformed_fu, 0.5)
        new_img.save("new.png", "PNG")


    def calc_robust_point_based_reg(self, bl_with_outliers: np.ndarray, fu_with_outliers: np.ndarray) -> tuple[None, None]:
        """
        Given two sets of points that might contain outliers from the bl and fu images
        uses ransac to find the best transformation using these sets of points.

        Parameters
        ----------
        self
        bl_with_outliers
        fu_with_outliers

        Returns
        -------

        """
        return ransac(bl_with_outliers, fu_with_outliers, RigidTransformFinder.calc_point_based_reg, RigidTransformFinder.calc_dist,
                      minPtNum=40, iterNum=1000, thDist=10, thInlrRatio=0.1)

def run():
    dataset_path = "/home/edan/Desktop/HighRad/Exercises/data/Targil2_data_2018-20230315T115832Z-001/Targil2_data_2018"
    bl = "BL01.tif"
    fu = "FU01.tif"

    bl = join(dataset_path, bl)
    fu = join(dataset_path, fu)
    registrator = RigidTransformFinder(bl, fu)
    # registrator.plot_points_on_images()

    # bl_points, fu_points = getPoints('no_outliers')
    # rigid_transformation = registrator.calc_point_based_reg(bl_points, fu_points)
    # registrator.calc_dist(bl_points, fu_points, rigid_transformation)

    # registrator.register(bl_points, fu_points)

    bl_w_outliers, fu_w_outliers = getPoints('with_outliers')
    transformation, indices = registrator.calc_robust_point_based_reg(bl_w_outliers, fu_w_outliers)
    registrator.plot_with_outliers(bl_w_outliers, fu_w_outliers, indices)





if __name__ == '__main__':
    run()
