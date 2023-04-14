"""
version: 0.0.1
*********************************

Dependencies:
- numpy
- opencv

*********************************


References

- Numpy Documentation: https://numpy.org/doc/

- OpenCV Documentation: https://docs.opencv.org/4.x/

*********************************



"""
import cv2
import numpy as np


class PatternMaker:
    def __init__(self, alpha=0.5, gamma=0, display=False):
        """
        This piece of code is a python implementation of the sliding window algorithm
        to superimpose a pattern over an image. It contains method functions for overlaying
        an mesh image/filter/pattern over the original image using weighted sum and for making
        patterns by passing a patch across the sliding window.

        """
        self.alpha = alpha
        self.gamma = gamma
        self.display = display

    def superimpose(self, original_image, mesh_image):
        """
        Overlay mesh image over the original image using weighted sum
        Read More:
        1.) https://www.etutorialspoint.com/index.php/319-python-opencv-overlaying-or-blending-two-images

        :param original_image: opencv source image array
        :param mesh_image: opencv mesh image array
        :return: final image after the weighted sum
        """
        overlay_pattern = cv2.addWeighted(
            mesh_image,
            self.alpha,
            original_image,
            1 - self.alpha,
            self.gamma,
        )
        if self.display:
            cv2.imshow("Final Image", overlay_pattern)
            cv2.waitKey(0)
        return overlay_pattern

    def make_patterns(self, image, mesh_img, window_size=200):
        """
        Sliding Window Algorithm
        ***************************************
        The function implements the sliding window algorithm to divide the original image into patches of size
        'window_size'. The algorithm iteratively moves a patch across the image in a left-to-right, top-to-bottom fashion,
        superimposing the mesh image on each patch of the original image using the 'superimpose()' method of the PatternMaker
        class. Within the function, the original image is first padded with a constant value of 255(white) along the borders to
        avoid edge cases while moving the sliding window. The function then initializes the starting coordinates of the patch, sets
        the direction of movement to right, and enters a loop that runs until the entire image has been covered by the patch.
        At each iteration of the loop, the function checks the direction of movement and updates the coordinates if the patch accordingly.
        It then calculates the coordinates of the bottom-right corner of the patch, computes the width and height of the patch, and checks
        whether the patch is fully contained within the boundaries of the original image.

        If the patch is fully contained within the boundaries of the original image, the 'superimpose()' method is called to overlay the mesh image
        on the patch and the resulting image is stored in a temporary variable 'img'. The contents of the original image are then updated with the contents
        of the temporary variable 'img', and the loop continues.
        ****************************************

        :param image: numpy array, original image
        :type image:  numpy array(int/float)
        :param mesh_image: numpy array, mesh image
        :type mesh_image: numpy array(int/float)
        :param window_size: height and width of patch, default value is 200
        :type window_size: int
        """
        h, w, _ = image.shape
        initial_coords = [0, 0]
        top = bottom = np.int32(window_size - 1)
        left = right = np.int32(window_size - 1)
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        h, w, _ = image.shape
        img = image.copy()
        direction = "right"
        while True:
            if direction == "right":
                initial_coords[0] += window_size
            elif direction == "left":
                initial_coords[0] -= window_size
            bottom_right_corner = (initial_coords[0] + window_size, initial_coords[1] + window_size)
            hy = abs(initial_coords[1] - bottom_right_corner[1])
            wx = abs(initial_coords[1] - bottom_right_corner[1])
            if (
                hy > 0
                and wx > 0
                and initial_coords[0] + wx < w
                and initial_coords[1] + hy < h
                and initial_coords[0] + wx > 0
                and initial_coords[1] + hy > 0
            ):
                image[
                    initial_coords[1] : initial_coords[1] + hy, initial_coords[0] : initial_coords[0] + wx, :
                ] = self.superimpose(
                    img[initial_coords[1] : initial_coords[1] + hy, initial_coords[0] : initial_coords[0] + wx, :],
                    mesh_img,
                )
            img = image.copy()
            if bottom_right_corner[0] >= w:
                direction = "left"
                initial_coords[1] += window_size
            elif bottom_right_corner[0] <= 0:
                direction = "right"
                initial_coords[1] += window_size
            if bottom_right_corner[0] >= w and bottom_right_corner[1] >= h:
                break
        return image
