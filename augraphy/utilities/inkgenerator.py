import random

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.morphology import thin
from sklearn.datasets import make_blobs

from augraphy.augmentations.brightness import Brightness
from augraphy.augmentations.lib import add_noise
from augraphy.augmentations.lib import binary_threshold
from augraphy.augmentations.lib import rotate_image
from augraphy.augmentations.lib import sobel


class InkGenerator:
    """Core object to generate different inks effect.

    :param ink_type: Types of ink, select from "pencil", "pen", "marker" or "highlighter".
    :type ink_type: string
    :param ink_draw_method: Content of ink generation, select from "lines" or "text".
    :param ink_draw_method: string
    :param ink_draw_iterations: Tuple of ints determining the drawing iterations
    :param ink_draw_iterations: int
    :param ink_location: Tuple of ints determining location of ink drawings.
            Or use "random: for random line location.
    :type ink_location: tuple or string
    :param ink_background: Background of ink generation.
    :param ink_background: numpy array
    :param ink_background_size: Tuple of ints (height, width) or (height, width, channels)
        determining the size of new background for ink generation.
        A new background will be created only if ink_background is not provided.
    :param ink_background_size: tuple
    :param ink_background_color: Tuple of ints (BGR) determining the color of background.
    :type ink_background_color: tuple
    :param ink_color: Tuple of ints (BGR) determining the color of ink.
    :type ink_color: tuple
    :param ink_min_brightness: Flag to enable min brightness in the generated ink.
    :type ink_min_brightness: int
    :param ink_min_brightness_value_range: Pair of ints determining the range for min brightness value in the generated ink.
    :type ink_min_brightness_value_range: tuple
    :param ink_draw_size_range: Pair of floats determining the range for
           the size of the ink drawing.
    :type ink_draw_size_range: tuple
    :param ink_thickness_range: Pair of floats determining the range for the thickness of the generated ink.
    :type scribbles_thickness_range: tuple
    :param ink_brightness_change: A list of value change for the brightness of the ink.
           If more than one value is provided, the final value will be randomly selected.
    :type ink_brightness_change: list
    :param ink_skeletonize: Flag to enable skeletonization in the generated drawings.
    :type ink_skeletonize: int
    :param ink_skeletonize_iterations_range: Pair of ints determining the number of iterations in skeletonization process.
    :type ink_skeletonize_iterations_range: int
    :param ink_text: Text value of ink generation, valid only if ink_draw_method is "text".
    :param ink_text: string
    :param ink_text_font: List contain paths to font types. Valid only if ink content is "text".
    :type ink_text_font: list
    :param ink_text_rotate_range: Tuple of ints to determine rotation angle of "text" based drawings.
    :type ink_text_rotate_range: tuple
    :param ink_lines_coordinates: A list contain coordinates of line.
    :type ink_lines_coordinates: list
    :param ink_lines_stroke_count_range: Pair of floats determining the range for
           the number of created lines.
    :type ink_lines_stroke_count_range: tuple
    """

    def __init__(
        self,
        ink_type,
        ink_draw_method,
        ink_draw_iterations,
        ink_location,
        ink_background,
        ink_background_size,
        ink_background_color,
        ink_color,
        ink_min_brightness,
        ink_min_brightness_value_range,
        ink_draw_size_range,
        ink_thickness_range,
        ink_brightness_change,
        ink_skeletonize,
        ink_skeletonize_iterations_range,
        ink_text,
        ink_text_font,
        ink_text_rotate_range,
        ink_lines_coordinates,
        ink_lines_stroke_count_range,
    ):
        self.ink_type = ink_type
        self.ink_draw_method = ink_draw_method
        self.ink_draw_iterations = ink_draw_iterations
        self.ink_location = ink_location
        self.ink_background = ink_background
        self.ink_background_size = ink_background_size
        self.ink_background_color = ink_background_color
        self.ink_color = ink_color
        self.ink_min_brightness = ink_min_brightness
        self.ink_min_brightness_value_range = ink_min_brightness_value_range
        self.ink_draw_size_range = ink_draw_size_range
        self.ink_thickness_range = ink_thickness_range
        self.ink_brightness_change = ink_brightness_change
        self.ink_skeletonize = ink_skeletonize
        self.ink_skeletonize_iterations_range = ink_skeletonize_iterations_range
        self.ink_text = ink_text
        self.ink_text_font = ink_text_font
        self.ink_text_rotate_range = ink_text_rotate_range
        self.ink_lines_coordinates = ink_lines_coordinates
        self.ink_lines_stroke_count_range = ink_lines_stroke_count_range

    def apply_brightness(self, image):
        """Brighten image based on the minimum brightness value by using Brightness augmentation.

        :param image: The image to be brighten.
        :type image: numpy.array (numpy.uint8)
        """

        # get location of intensity < min intensity
        min_intensity = random.randint(self.ink_min_brightness_value_range[0], self.ink_min_brightness_value_range[1])
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        y_location, x_location = np.where(image_hsv[:, :, 2] < min_intensity)

        # if there's location where intensity < min intensity, apply brightness
        if len(y_location) > 0:
            image_min_intensity = min(image_hsv[:, :, 2][y_location, x_location])
            if image_min_intensity > 0 and image_min_intensity < min_intensity:
                brighten_ratio = abs(image_min_intensity - min_intensity) / image_min_intensity
                brighten_min = 1 + brighten_ratio
                brighten_max = 1 + brighten_ratio + 0.5
                brightness = Brightness(brightness_range=(brighten_min, brighten_max))
                image = brightness(image)

        return image

    def apply_pencil_effect(self, ink_image, ink_background):
        """Apply foreground image with pencil effect to background image.

        :param ink_image: Image with pencil drawings.
        :type ink_image: numpy.array (numpy.uint8)
        :param image: The background image.
        :type image: numpy.array (numpy.uint8)
        """
        # create pencil effect by changing some pixels into white background
        gray_image = cv2.cvtColor(ink_image, cv2.COLOR_BGR2GRAY)
        noise_mask = add_noise(gray_image, (1.0, 1.0), (0, 2), 4)
        for i in range(ink_image.shape[2]):
            ink_image[:, :, i][noise_mask == 0] = 255
        ink_image = cv2.GaussianBlur(ink_image, (3, 3), 0)

        # add some brightness in value channel of hsv image
        hsv = cv2.cvtColor(ink_image, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:, :, 2] += random.choice(self.ink_brightness_change)
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

        hsv = np.array(hsv, dtype=np.uint8)
        ink_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return cv2.multiply(ink_image, ink_background, scale=1 / 255)

    def apply_pen_effect(self, ink_image, ink_background):
        """Apply foreground image with pen effect to background image.

        :param ink_image: Image with pen drawings.
        :type ink_image: numpy.array (numpy.uint8)
        :param image: The background image.
        :type image: numpy.array (numpy.uint8)
        """
        return cv2.multiply(ink_image, ink_background, scale=1 / 255)

    def apply_marker_effect(self, ink_image, ink_background):
        """Apply foreground image with marker effect to background image.

        :param ink_image: Image with marker drawings.
        :type ink_image: numpy.array (numpy.uint8)
        :param image: The background image.
        :type image: numpy.array (numpy.uint8)
        """

        gray_image = cv2.cvtColor(ink_image, cv2.COLOR_BGR2GRAY)
        ink_image = gray_image.copy()

        # kernel size depends on thickness
        kernel_size = random.randint(self.ink_thickness_range[0], self.ink_thickness_range[1])

        # minimum size of marker effect
        min_size = 2
        # kernel for jagged effect
        kernel_jagged = np.zeros((min_size + kernel_size, min_size + kernel_size), dtype="uint8")
        kernel_jagged[int((min_size + kernel_size) / 2), :] = 1
        kernel_jagged[:, int((min_size + kernel_size) / 2)] = 1

        # erode image and get jagged edge
        random_image = np.random.randint(0, 255, size=ink_image.shape[:2], dtype="uint8")
        image_jagged = np.full(ink_image.shape, fill_value=255, dtype="uint8")
        indices = np.logical_and(gray_image < 255, random_image < 128)
        image_jagged[indices] = 0
        image_jagged = cv2.erode(image_jagged, kernel_jagged, iterations=2)

        # erode ink image , but with lesser iteration
        ink_image = cv2.erode(ink_image, kernel_jagged, iterations=1)

        # add white noise in the borders
        image_white_noise = np.full(ink_image.shape, fill_value=255, dtype="uint8")
        kernel_white_noise = np.ones((max(min_size, kernel_size), max(min_size, kernel_size)), dtype="uint8")
        random_image2 = np.random.randint(0, 255, size=ink_image.shape[:2], dtype="uint8")
        indices_white_noise = np.logical_and(gray_image < 255, random_image2 < 16)
        image_white_noise[indices_white_noise] = 0
        image_white_noise = cv2.erode(image_white_noise, kernel_white_noise, iterations=2)
        image_jagged[image_white_noise < 255] = 255

        # merge jagged and eroded ink image
        ink_image[ink_image < 255] = 0
        ink_image[ink_image > 0] = 1
        ink_image = np.multiply(image_jagged, ink_image)

        # add white noise at the center of the ink
        # this value control white noise at the center, the higher the value, the lower the white noise at the center
        center_white_noise_threshold = random.randint(200, 225)
        image_sobel = sobel(gray_image)
        image_sobel[random_image2 < center_white_noise_threshold] = 0
        ink_image[image_sobel > 0] = 255

        # update color
        ink_image_gray = ink_image
        ink_image = cv2.cvtColor(ink_image, cv2.COLOR_GRAY2BGR)
        ink_image[ink_image_gray < 255] = self.ink_color

        # Last step, blur image for better effect
        ink_image = cv2.GaussianBlur(ink_image, (3, 3), 0)

        return cv2.multiply(ink_image, ink_background, scale=1 / 255)

    def generate_edges(self, image, repeat=0, randomize=0):
        """Generate edges based on image gradients.

        :param image: Image with lines ot text drawings.
        :type image: numpy.array (numpy.uint8)
        :param repeat: Number of recursive edge generation iterations.
        :type repeat: int, optional
        :param randomize: Number of recursive edge generation iterations.
        :type randomize: int, optional

        """
        gx, gy = np.gradient(image, edge_order=1)

        if randomize:
            image_edges = (gx * random.choice([1, -1])) + (gy * random.choice([1, -1]))
            image_edges[image_edges < 0] = 0
        else:
            image_edges = abs(gx) + abs(gy)
        image_edges[image_edges > 0] = 1

        if repeat:
            image_edges = self.generate_edges(image_edges, repeat=repeat - 1)

        return image_edges

    def generate_noise_clusters(self, image, n_clusters=(200, 200), n_samples=(300, 300), std_range=(5, 10)):
        """Generate clusters of noise with make_blobs.

        :param image: Image with lines ot text drawings.
        :type image: numpy.array (numpy.uint8)
        :param n_clusters: Tuple of ints determining number of noise clusters.
        :type n_clusters: tuple, optional
        :param n_samples: Tuple of ints determining number of noise sample in each cluster.
        :type n_samples: tuple, optional
        :param std_range: Tuple of ints determining spreadness of noise clusters.
        :type std_range: int, optional

        """

        # generate random clusters, samples and std
        samples = [
            random.randint(n_samples[0], n_samples[1]) for _ in range(random.randint(n_clusters[0], n_clusters[1]))
        ]
        std = random.randint(std_range[0], std_range[1])

        # generate clusters of blobs
        generated_points, point_group = make_blobs(
            n_samples=samples,
            center_box=(0, max(image.shape)),
            cluster_std=std,
            n_features=2,
        )

        # index of invalid points (beyond image borders)
        ind_delete = np.logical_or.reduce(
            (
                generated_points[:, 0] < 0,
                generated_points[:, 1] < 0,
                generated_points[:, 0] > image.shape[1] - 1,
                generated_points[:, 1] > image.shape[0] - 1,
            ),
        )
        # delete invalid points
        generated_points_x = np.delete(generated_points[:, 0], ind_delete.reshape(ind_delete.shape[0]), axis=0).astype(
            "int",
        )
        generated_points_y = np.delete(generated_points[:, 1], ind_delete.reshape(ind_delete.shape[0]), axis=0).astype(
            "int",
        )

        # add noise into image
        image_noise = np.zeros_like(image, dtype="uint8")
        image_noise[generated_points_y, generated_points_x] = 1

        return image_noise

    def apply_highlighter_effect(self, ink_image, ink_background):
        """Apply foreground image with highlighter effect to background image.

        :param ink_image: Image with highlighter drawings.
        :type ink_image: numpy.array (numpy.uint8)
        :param ink_background: The background image.
        :type ink_background: numpy.array (numpy.uint8)
        """

        gray_image = cv2.cvtColor(ink_image, cv2.COLOR_BGR2GRAY)
        ink_image = gray_image.copy()

        # kernel size depends on thickness
        kernel_size = random.randint(self.ink_thickness_range[0], self.ink_thickness_range[1])

        # minimum size of highligther effect
        min_size = 3
        # kernel with diagonal lines to produce highlighter effect
        kernel_diagonal = np.zeros((min_size + kernel_size, min_size + kernel_size), dtype="uint8")
        for i in range(min_size + kernel_size):
            kernel_diagonal[i, i] = 1
        if min_size + kernel_size > 2:
            mid_location = int((min_size + kernel_size) / 2)
            kernel_diagonal[mid_location, mid_location] = 0

        # dilate image to make it thicker
        ink_image = cv2.erode(ink_image, kernel_diagonal, iterations=2)

        # effect at the core center of ink
        image_edges_center = self.generate_edges(gray_image, repeat=2, randomize=1)
        # effect at the borders
        image_edges_border = self.generate_edges(ink_image, repeat=2, randomize=1)
        # effect with single or multiple lines
        image_edges_lines = self.generate_edges(gray_image, repeat=0, randomize=1)

        n_clusters = (200 + kernel_size * 10, 250 + kernel_size * 10)
        n_samples = (200 + kernel_size * 5, 250 + kernel_size * 5)
        std_range = (3 + np.ceil(kernel_size / 5), 7 + np.ceil(kernel_size / 5))
        image_noise_center = self.generate_noise_clusters(gray_image, n_clusters, n_samples, std_range)
        image_noise_border = self.generate_noise_clusters(gray_image, n_clusters, n_samples, std_range)

        # white line as noise
        if self.ink_draw_method == "lines":
            indices = np.logical_and(image_edges_lines > 0, (image_noise_border + image_noise_center) > 0)
            ink_image[indices] = 255

        # effect at center line
        indices = np.logical_and(image_edges_center > 0, image_noise_center > 0)
        ink_image[indices] = 255

        # effect at borders
        indices = np.logical_and(image_edges_border > 0, image_noise_border > 0)
        ink_image[indices] = 255

        # convert back image to BGR and add color
        ink_image_gray = ink_image
        ink_image = cv2.cvtColor(ink_image, cv2.COLOR_GRAY2BGR)
        ink_image[ink_image_gray < 255] = self.ink_color

        # Last step, blur image for better effect
        ink_image = cv2.GaussianBlur(ink_image, (3, 3), 0)

        return cv2.multiply(ink_image, ink_background, scale=1 / 255)

    def generate_lines(self, ink_background):
        """Generated lines drawing in background image.

        :param ink_backgrounde: The background image.
        :type ink_background: numpy.array (numpy.uint8)
        """

        # ink background is the max size
        max_height, max_width = ink_background.shape[:2]

        ink_draw_iterations = random.randint(
            self.ink_draw_iterations[0],
            self.ink_draw_iterations[1],
        )

        # background across all iterations
        combined_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

        for i in range(ink_draw_iterations):

            # background of lines image
            lines_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

            # each stroke count
            ink_lines_stroke_count = random.randint(
                self.ink_lines_stroke_count_range[0],
                self.ink_lines_stroke_count_range[1],
            )

            if self.ink_lines_coordinates == "random":
                # get size of foreground
                size = random.randint(max(self.ink_draw_size_range[0], 30), max(40, self.ink_draw_size_range[1]))
                xsize = ysize = min([size, max_height, max_width])
            else:
                # if coordinates are provided, all lines will be drew at one time
                ink_lines_stroke_count = 1

                xpoint_min = max_width
                ypoint_min = max_height
                xpoint_max = 0
                ypoint_max = 0
                for points in self.ink_lines_coordinates:
                    xpoints = points[:, 0]
                    ypoints = points[:, 1]
                    xpoint_min = min(xpoint_min, min(xpoints))
                    ypoint_min = min(ypoint_min, min(ypoints))
                    xpoint_max = max(xpoint_max, max(xpoints))
                    ypoint_max = max(ypoint_max, max(ypoints))

                # add offset, to prevent cut off of thicken drawing at the edges of image
                offset = 50
                xsize = xpoint_max - xpoint_min + (offset * 2)
                ysize = ypoint_max - ypoint_min + (offset * 2)

                # reset coordinates so that it starts at min coordinates
                ink_lines_coordinates = []
                for points in self.ink_lines_coordinates:
                    points_new = []
                    xpoints = points[:, 0]
                    ypoints = points[:, 1]
                    for xpoint, ypoint in zip(xpoints, ypoints):
                        points_new.append([xpoint - xpoint_min + offset, ypoint - ypoint_min + offset])
                    ink_lines_coordinates.append(np.array(points_new))

                # fixed ink location if lines coordinates are provided
                self.ink_location = (xpoint_min, ypoint_min)

            if self.ink_location == "random":
                # random paste location
                xstart = random.randint(0, max(1, max_width - xsize - 1))
                ystart = random.randint(0, max(1, max_height - ysize - 1))
            else:
                xstart, ystart = self.ink_location
                xstart = max(0, xstart - offset)
                ystart = max(0, ystart - offset)
                if xstart < 0:
                    xstart = 0
                elif xstart + xsize >= max_width:
                    xsize = max_width - xstart
                if ystart < 0:
                    ystart = 0
                elif ystart + ysize >= max_height:
                    ysize = max_height - ystart

            # create each stroke of lines
            for i in range(ink_lines_stroke_count):

                # generate lines thickness
                ink_thickness = random.randint(
                    self.ink_thickness_range[0],
                    self.ink_thickness_range[1],
                )

                # min thickness of highlighter is 2
                if self.ink_type == "highlighter":
                    ink_thickness = max(2, ink_thickness)

                # foreground of line image
                line_image = np.full((ysize, xsize, 3), fill_value=255, dtype="uint8")

                if self.ink_lines_coordinates == "random":
                    x = np.array(
                        [
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                            random.randint(5, xsize - 25),
                        ],
                    )
                    y = np.array(
                        [
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                            random.randint(5, ysize - 25),
                        ],
                    )

                    start_stop = [
                        random.randint(5, ysize // 2),
                        random.randint(ysize // 2, ysize - 5),
                    ]

                    # Initilaize y axis
                    lspace = np.linspace(min(start_stop), max(start_stop))

                    # calculate the coefficients.
                    z = np.polyfit(x, y, 2)

                    # calculate x axis
                    line_fitx = z[0] * lspace**2 + z[1] * lspace + z[2]
                    verts = np.array(list(zip(line_fitx.astype(int), lspace.astype(int))))
                    ink_lines_coordinates = [verts]

                # get a patch of background
                line_background = lines_background[ystart : ystart + ysize, xstart : xstart + xsize]

                # draw lines
                cv2.polylines(
                    line_image,
                    ink_lines_coordinates,
                    False,
                    self.ink_color,
                    thickness=ink_thickness,
                )

                # apply line image with ink effect to background
                line_background = self.apply_ink_effect(line_image, line_background)

                # reassign background patch to background
                lines_background[ystart : ystart + ysize, xstart : xstart + xsize] = line_background

            # reduce transparency for highligther
            if self.ink_type == "highlighter":
                lines_background = cv2.addWeighted(
                    lines_background,
                    0.5,
                    np.full_like(lines_background, fill_value=255, dtype="uint8"),
                    0.5,
                    0,
                )

            # combine backgrounds in each iteration
            combined_background = cv2.multiply(lines_background, combined_background, scale=1 / 255)

        # skeletonize image (optional)
        if self.ink_skeletonize:
            binary_image = cv2.cvtColor(255 - combined_background, cv2.COLOR_BGR2GRAY)
            binary_image[binary_image > 0] = 1
            max_iter = random.randint(
                self.ink_skeletonize_iterations_range[0],
                self.ink_skeletonize_iterations_range[1],
            )
            thin_mask = thin(binary_image, max_iter=max_iter) * 1
            for i in range(3):
                combined_background[:, :, i][thin_mask == 0] = 255

        # brighten image to reach minimum brightness (optional)
        if self.ink_min_brightness:
            combined_background = self.apply_brightness(combined_background)

        # merge image with lines with ink background
        image_output = cv2.multiply(combined_background, ink_background, scale=1 / 255)

        return image_output

    def apply_ink_effect(self, foreground_image, background_image):
        """Function to apply various ink effect.

        :param foreground_image: Foreground image with lines or text.
        :type foreground_image: numpy.array (numpy.uint8)
        :param background_image: The background image.
        :type background_image: numpy.array (numpy.uint8)
        """
        # make sure both images are in a same size
        bysize, bxsize = background_image.shape[:2]
        if foreground_image.shape[0] != bysize or foreground_image.shape[1] != bxsize:
            foreground_image = cv2.resize(foreground_image, (bxsize, bysize), interpolation=cv2.INTER_AREA)

        # pencil
        if self.ink_type == "pencil":
            image_merged = self.apply_pencil_effect(foreground_image, background_image)
        # pen
        elif self.ink_type == "pen":
            image_merged = self.apply_pen_effect(foreground_image, background_image)
        # marker
        elif self.ink_type == "marker":
            image_merged = self.apply_marker_effect(foreground_image, background_image)
        # highlighter
        else:
            image_merged = self.apply_highlighter_effect(foreground_image, background_image)

        return image_merged

    def generate_text(self, ink_background):
        """Generated texts drawing in background image.

        :param ink_backgrounde: The background image.
        :type ink_background: numpy.array (numpy.uint8)
        """
        # ink background is the max size
        max_height, max_width = ink_background.shape[:2]

        ink_draw_iterations = random.randint(
            self.ink_draw_iterations[0],
            self.ink_draw_iterations[1],
        )

        # background across all iterations
        combined_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

        for i in range(ink_draw_iterations):

            # foreground and background of text image
            texts_image = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")
            texts_background = np.full((max_height, max_width, 3), fill_value=255, dtype="uint8")

            # convert image to PIL
            texts_image_PIL = Image.fromarray(texts_image)
            draw = ImageDraw.Draw(texts_image_PIL)
            # set font and size
            font = ImageFont.truetype(
                random.choice(self.ink_text_font),
                size=int(random.randint(self.ink_draw_size_range[0], self.ink_draw_size_range[1]) / 4),
            )
            if self.ink_text == "random":
                text = random.choice(["DEMO", "APPROVED", "CHECKED", "ORIGINAL", "COPY", "CONFIDENTIAL"])
            else:
                text = self.ink_text

            # thickness of text
            ink_thickness = random.randint(
                self.ink_thickness_range[0],
                self.ink_thickness_range[1],
            )

            # draw text
            draw.text(
                (int(max_width / 2), int(max_height / 2)),
                text,
                font=font,
                stroke_width=ink_thickness,
                fill=self.ink_color,
            )

            # convert it back to numpy array
            texts_image = np.array(texts_image_PIL)

            # rotate image
            texts_image = rotate_image(
                texts_image,
                random.randint(self.ink_text_rotate_range[0], self.ink_text_rotate_range[1]),
            )

            # resize to make sure rotated image size is consistent
            texts_image = cv2.resize(texts_image, (max_width, max_height), interpolation=cv2.INTER_AREA)

            # remove additional blank area
            binary_image = binary_threshold(texts_image, threshold_method="threshold_otsu", threshold_arguments={})

            coordinates = cv2.findNonZero(255 - binary_image)
            x, y, xsize, ysize = cv2.boundingRect(coordinates)
            # minimum size
            xsize = min(5, binary_image.shape[1])
            ysize = min(5, binary_image.shape[0])
            texts_image = texts_image[y : y + ysize, x : x + xsize]

            if self.ink_location == "random":
                # random paste location
                xstart = random.randint(0, max(0, max_width - xsize - 1))
                ystart = random.randint(0, max(0, max_height - ysize - 1))
            else:
                xstart, ystart = self.ink_location
                if xstart < 0:
                    xstart = 0
                elif xstart + xsize >= max_width:
                    xstart = max_width - xsize - 1
                if ystart < 0:
                    ystart = 0
                elif ystart + ysize >= max_height:
                    ystart = max_height - ysize - 1

            text_background = texts_background[ystart : ystart + ysize, xstart : xstart + xsize]

            # apply foreground image to background
            text_background = self.apply_ink_effect(texts_image, text_background)

            texts_background[ystart : ystart + ysize, xstart : xstart + xsize] = text_background

            # reduce transparency for highligther
            if self.ink_type == "highlighter":
                texts_background = cv2.addWeighted(
                    texts_background,
                    0.5,
                    np.full_like(texts_background, fill_value=255, dtype="uint8"),
                    0.5,
                    0,
                )

            # combine backgrounds in each iteration
            combined_background = cv2.multiply(texts_background, combined_background, scale=1 / 255)

        # skeletonize image (optional)
        if self.ink_skeletonize:
            binary_image = cv2.cvtColor(255 - combined_background, cv2.COLOR_BGR2GRAY)
            binary_image[binary_image > 0] = 1
            max_iter = random.randint(
                self.ink_skeletonize_iterations_range[0],
                self.ink_skeletonize_iterations_range[1],
            )
            thin_mask = thin(binary_image, max_iter=max_iter) * 1
            for i in range(3):
                combined_background[:, :, i][thin_mask == 0] = 255

        # brighten image to reach minimum brightness (optional)
        if self.ink_min_brightness:
            combined_background = self.apply_brightness(combined_background)

        # merge image with texts with ink background
        image_output = cv2.multiply(combined_background, ink_background, scale=1 / 255)

        return image_output

    def generate_ink(
        self,
        ink_type=None,
        ink_draw_method=None,
        ink_draw_iterations=None,
        ink_location=None,
        ink_background=None,
        ink_background_size=None,
        ink_background_color=None,
        ink_color=None,
        ink_min_brightness=None,
        ink_min_brightness_value_range=None,
        ink_draw_size_range=None,
        ink_thickness_range=None,
        ink_brightness_change=None,
        ink_skeletonize=None,
        ink_text=None,
        ink_text_font=None,
        ink_text_rotate_range=None,
        ink_lines_coordinates=None,
        ink_lines_curvy=None,
        ink_lines_stroke_count_range=None,
    ):
        """Main function to print ink into the background.

        :param ink_type: Types of ink, select from "pencil", "pen", "marker" or "highlighter".
        :type ink_type: string, optional
        :param ink_draw_method: Content of ink generation, select from "lines" or "text".
        :param ink_draw_method: string, optional
        :param ink_draw_iterations: Tuple of ints determining the drawing iterations
        :param ink_draw_iterations: int, optional
        :param ink_location: Tuple of ints determining location of ink drawings.
                Or use "random: for random line location.
        :type ink_location: tuple or string, optional
        :param ink_background: Background of ink generation.
        :param ink_background: numpy array, optional
        :param ink_background_size: Tuple of ints (height, width) or (height, width, channels)
            determining the size of new background for ink generation.
            A new background will be created only if ink_background is not provided.
        :param ink_background_size: tuple, optional
        :param ink_background_color: Tuple of ints (BGR) determining the color of background.
        :type ink_background_color: tuple, optional
        :param ink_color: Tuple of ints (BGR) determining the color of ink.
        :type ink_color: tuple, optional
            :param ink_min_brightness: Flag to enable min brightness in the generated ink.
        :type ink_min_brightness: int, optional
        :param ink_min_brightness_value_range: Pair of ints determining the range for min brightness value in the generated ink.
        :type ink_min_brightness_value_range: tuple, optional
        :param ink_draw_size_range: Pair of floats determining the range for
               the size of the ink drawing.
        :type ink_draw_size_range: tuple, optional
        :param ink_thickness_range: Pair of floats determining the range for the thickness of the created ink.
        :type scribbles_thickness_range: tuple, optional
        :param ink_brightness_change: A list of value change for the brightness of the ink.
               If more than one value is provided, the final value will be randomly selected.
        :type ink_brightness_change: list, optional
        :param ink_skeletonize: Flag to enable skeletonization in the generated drawings.
        :type ink_skeletonize: int, optional
        :param ink_skeletonize_iterations_range: Pair of ints determining the number of iterations in skeletonization process.
        :type ink_skeletonize_iterations_range: int, optional
        :param ink_text: Text value of ink generation, valid only if ink_draw_method is "text".
        :param ink_text: string, optional
        :param ink_text_font: List contain paths to font types. Valid only if ink content is "text".
        :type ink_text_font: list, optional
        :param ink_text_rotate_range: Tuple of ints to determine rotation angle of "text" based drawings.
        :type ink_text_rotate_range: tuple, optional
        :param ink_lines_coordinates: A list contain coordinates of line.
        :type ink_lines_coordinates: list, optional
        :param ink_lines_stroke_count_range: Pair of floats determining the range for
               the number of created lines.
        :type ink_lines_stroke_count_range: tuple, optional
        """

        # If input is not None, replace self parameters

        if ink_type is not None:
            self.ink_type = ink_type
        if ink_draw_method is not None:
            self.ink_draw_method = ink_draw_method
        if ink_draw_iterations is not None:
            self.ink_draw_iterations = ink_draw_iterations
        if ink_location is not None:
            self.ink_location = ink_location
        if ink_background is not None:
            self.ink_background = ink_background
        if ink_background_size is not None:
            self.ink_background_size = ink_background_size
        if ink_background_color is not None:
            self.ink_background_color = ink_background_color
        if ink_color is not None:
            self.ink_color = ink_color
        if ink_min_brightness is not None:
            self.ink_min_brightness = ink_min_brightness
        if ink_min_brightness_value_range is not None:
            self.ink_min_brightness_value_range = ink_min_brightness_value_range
        if ink_draw_size_range is not None:
            self.ink_draw_size_range = ink_draw_size_range
        if ink_thickness_range is not None:
            self.ink_thickness_range = ink_thickness_range
        if ink_brightness_change is not None:
            self.ink_brightness_change = ink_brightness_change
        if ink_skeletonize is not None:
            self.ink_skeletonize = ink_skeletonize
        if ink_text is not None:
            self.ink_text = ink_text
        if ink_text_font is not None:
            self.ink_text_font = ink_text_font
        if ink_text_rotate_range is not None:
            self.ink_text_rotate_range = ink_text_rotate_range
        if ink_lines_coordinates is not None:
            self.ink_lines_coordinates = ink_lines_coordinates
        if ink_lines_curvy is not None:
            self.ink_lines_curvy = ink_lines_curvy
        if ink_lines_stroke_count_range is not None:
            self.ink_lines_stroke_count_range = ink_lines_stroke_count_range

        # retrieve or create background
        if isinstance(self.ink_background, np.ndarray):
            ink_background = self.ink_background.copy()
        else:
            ink_background = np.full(self.ink_background_size, fill_value=self.ink_background_color, dtype="uint8")

        # check if provided background is grayscale
        if len(ink_background.shape) < 3:
            background_is_grayscale = 1
        else:
            background_is_grayscale = 0

        # change to BGR for further processing
        if background_is_grayscale:
            ink_background = cv2.cvtColor(ink_background, cv2.COLOR_GRAY2BGR)

        # generate ink effect
        if self.ink_draw_method == "lines":
            image_output = self.generate_lines(ink_background)
        else:
            image_output = self.generate_text(ink_background)

        # return image in a same format
        if background_is_grayscale:
            image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2GRAY)

        return image_output
