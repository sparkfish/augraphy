import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation


class Folding(Augmentation):
    """Emulates folding effect from perspective transformation

    :param fold count: Number of applied foldings
    :type fold_count: int, optional
    :param fold_noise: Level of noise added to folding area. Range from
                        value of 0 to 1.
    :type fold_noise: float, optional
    :param gradient_width: Tuple (min, max) Measure of the space affected
                            by fold prior to being warped (in units of
                            percentage of width of page)
    :type gradient_width: tuple, optional
    :param gradient_height: Tuple (min, max) Measure of depth of fold (unit
                            measured as percentage page height)
    :type gradient_height: tuple, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """

    def __init__(
        self,
        fold_count=2,
        fold_noise=0.1,
        gradient_width=(0.1, 0.2),
        gradient_height=(0.01, 0.02),
        p=1,
    ):
        super().__init__(p=p)
        self.fold_count = fold_count
        self.fold_noise = fold_noise
        self.gradient_width = gradient_width
        self.gradient_height = gradient_height

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"Folding(fold_count={self.fold_count}, fold_noise={self.fold_noise}, gradient_width={self.gradient_width}, gradient_height={self.gradient_height},p={self.p})"

    # Perspective transform based on 4 points
    def four_point_transform(self, image, pts, dst, xs, ys):
        M = cv2.getPerspectiveTransform(pts, dst)
        if not isinstance(image, np.floating):
            image = image.astype("float")
        img_warped = cv2.warpPerspective(image, M, (xs, ys))
        # return the warped image
        return img_warped

    # Transform left side of folding area
    def warp_fold_left_side(
        self,
        img,
        ysize,
        fold_noise,
        fold_x,
        fold_width_one_side,
        fold_y_shift,
    ):

        img_fuse = img.copy()

        # 4 vectices of folding area
        xs = 0  # xleft
        xe = fold_width_one_side  # xright
        ys = 0  # ytop
        ye = ysize  # ybottom

        # before distortion
        top_left = [xs, ys]
        top_right = [xe, ys]
        bottom_left = [xs, ye]
        bottom_right = [xe, ye]

        # after distortion
        dtop_left = [xs, ys]
        dtop_right = [xe, ys + fold_y_shift]
        dbottom_left = [xs, ye]
        dbottom_right = [xe, ye + fold_y_shift]

        # image cropping points
        cxs = fold_x
        cxe = fold_x + fold_width_one_side
        cys = 0
        cye = ysize

        # points of folding area
        source_pts = np.array(
            [top_left, bottom_left, bottom_right, top_right],
            dtype=np.float32,
        )
        destination_pts = np.array(
            [dtop_left, dbottom_left, dbottom_right, dtop_right],
            dtype=np.float32,
        )

        # crop section of folding area
        img_crop = img[cys:cye, cxs:cxe]

        # get image dimension of cropped image
        if len(img_crop.shape) > 2:
            cysize, cxsize, cdim = img_crop.shape
        else:
            cysize, cxsize = img_crop.shape
            cdim = 2

        # warp folding area
        img_warped = self.four_point_transform(
            img_crop,
            source_pts,
            destination_pts,
            cxsize,
            cysize + fold_y_shift,
        )
        img_warped = self.add_noise(img_warped, 1, fold_noise / 2)

        if cdim > 2:
            img_fuse[cys:cye, cxs:cxe, :] = img_warped[:-fold_y_shift, :, :]
        else:
            img_fuse[cys:cye, cxs:cxe] = img_warped[:-fold_y_shift, :]

        return img_fuse

    # Transform right side of folding area
    def warp_fold_right_side(
        self,
        img,
        ysize,
        fold_noise,
        fold_x,
        fold_width_one_side,
        fold_y_shift,
    ):

        img_fuse = img.copy()

        # 4 vectices of folding area
        xs = 0  # xleft
        xe = fold_width_one_side  # xright
        ys = 0  # ytop
        ye = ysize  # ybottom

        # before distortion
        top_left = [xs, ys]
        top_right = [xe, ys]
        bottom_left = [xs, ye]
        bottom_right = [xe, ye]

        # after distortion
        dtop_left = [xs, ys + (fold_y_shift)]
        dtop_right = [xe, ys]
        dbottom_left = [xs, ye + (fold_y_shift)]
        dbottom_right = [xe, ye]

        # image cropping points
        cxs = fold_x + fold_width_one_side
        cxe = fold_x + (fold_width_one_side * 2)
        cys = 0
        cye = ysize

        # points of folding area
        source_pts = np.array(
            [top_left, bottom_left, bottom_right, top_right],
            dtype=np.float32,
        )
        destination_pts = np.array(
            [dtop_left, dbottom_left, dbottom_right, dtop_right],
            dtype=np.float32,
        )

        # crop section of folding area
        img_crop = img[cys:cye, cxs:cxe]

        # get image dimension of cropped image
        if len(img_crop.shape) > 2:
            cysize, cxsize, cdim = img_crop.shape
        else:
            cysize, cxsize = img_crop.shape
            cdim = 2

        # warp folding area
        img_warped = self.four_point_transform(
            img_crop,
            source_pts,
            destination_pts,
            cxsize,
            cysize + fold_y_shift,
        )
        img_warped = self.add_noise(img_warped, 0, fold_noise / 2)

        if cdim > 2:
            img_fuse[cys:cye, cxs:cxe, :] = img_warped[:-fold_y_shift, :, :]
        else:
            img_fuse[cys:cye, cxs:cxe] = img_warped[:-fold_y_shift, :]

        return img_fuse

    # Generate noise to edges of folding
    def add_noise(self, img, side, p=0.1):
        # side = flag to put more noise at certain side
        #   0  = left side
        #   1  = right side

        # get image dimension
        ysize, xsize = img.shape[:2]

        for y in range(ysize):
            for x in range(xsize):

                if side:  # more noise on right side
                    p_score = (((x) / xsize) ** 3) * p  # non linear score with power
                else:  # more noise on left side
                    p_score = (((xsize - x) / xsize) ** 3) * p  # non linear score with power

                if p_score > random.random():
                    img[y, x] = 0
        return img

    # Apply perspective transform 2 times and get single folding effect
    def apply_folding(
        self,
        img,
        ysize,
        xsize,
        gradient_width,
        gradient_height,
        fold_noise,
    ):

        min_fold_x = min(np.ceil(gradient_width[0] * xsize), xsize).astype("int")
        max_fold_x = min(np.ceil(gradient_width[1] * xsize), xsize).astype("int")
        fold_width_one_side = int(
            random.randint(min_fold_x, max_fold_x) / 2,
        )  # folding width from left to center of folding, or from right to center of folding

        # test for valid folding center line
        if (xsize - fold_width_one_side - 1) < (fold_width_one_side + 1):
            print("Folding augmentation is not applied, please increase image size")
            return img

        fold_x = random.randint(
            fold_width_one_side + 1,
            xsize - fold_width_one_side - 1,
        )  # center of folding

        fold_y_shift_min = min(np.ceil(gradient_height[0] * ysize), ysize).astype("int")
        fold_y_shift_max = min(np.ceil(gradient_height[1] * ysize), ysize).astype("int")
        fold_y_shift = random.randint(
            fold_y_shift_min,
            fold_y_shift_max,
        )  # y distortion in folding (support positive y value for now)

        if (fold_width_one_side != 0) and (fold_y_shift != 0):
            img_fold_l = self.warp_fold_left_side(
                img,
                ysize,
                fold_noise,
                fold_x,
                fold_width_one_side,
                fold_y_shift,
            )
            img_fold_r = self.warp_fold_right_side(
                img_fold_l,
                ysize,
                fold_noise,
                fold_x,
                fold_width_one_side,
                fold_y_shift,
            )
            return img_fold_r
        else:
            if fold_width_one_side == 0:
                print(
                    "Folding augmentation is not applied, please increase gradient width or image size",
                )
            else:
                print(
                    "Folding augmentation is not applied, please increase gradient height or image size",
                )
            return img

    # Applies the Augmentation to input data.
    def __call__(self, image, layer=None, force=False):
        if force or self.should_run():
            image = image.copy()

            # get image dimension
            if len(image.shape) > 2:
                ysize, xsize, _ = image.shape
            else:
                ysize, xsize = image.shape

            # apply folding multiple times
            image_fold = image.copy()
            for _ in range(self.fold_count):
                image_fold = self.apply_folding(
                    image_fold,
                    ysize,
                    xsize,
                    self.gradient_width,
                    self.gradient_height,
                    self.fold_noise,
                )

            return image_fold
