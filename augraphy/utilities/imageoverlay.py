import random

import cv2
import numpy as np

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult


class ImageOverlay(Augmentation):
    """Takes a background and foreground image and overlays foreground somewhere
    on background. Not all of foreground will necessarily be visible; some may
    be cut off by the edge of the background image.

    :param layer: The layer of image to overlay onto
    :type layer: string, optional
    :param background: the document on which to overlay the foreground
    :type background: np.array
    :param foreground: the image to overlay on the background document
    :type foreground: np.array
    :param p: the probability this augmentation will be applied
    :type p: float, optional
    """

    def __init__(self, foreground, layer="post", p=0.5):
        self.foreground = foreground
        self.layer = layer
        super().__init__(p=p)

    def workspace(self, background):
        """Creates an empty image on which to do the overlay operation"""

        xdim = background.shape[0] + (2 * self.foreground.shape[0])
        ydim = background.shape[1] + (2 * self.foreground.shape[1])

        return np.zeros((xdim, ydim, 3))

    def layerForeground(self, ambient, xloc, yloc):
        """Put self.foreground at (xloc,yloc) on ambient"""
        xstop = xloc + self.foreground.shape[0]
        ystop = yloc + self.foreground.shape[1]
        ambient[xloc:xstop, yloc:ystop] = self.foreground
        return ambient

    def overlay(self, background, foreground):
        """Centers the background image over workspace, then places foreground
        somewhere on the workspace, and finally crops to the
        background dimension
        """

        # Get the boundaries of the background image
        xstart = self.foreground.shape[0]
        ystart = self.foreground.shape[1]
        xstop = xstart + background.shape[0]
        ystop = ystart + background.shape[1]

        # Build the array we'll do work in
        ambient = self.workspace(background)

        # Center the background image
        ambient[xstart:xstop, ystart:ystop] = background

        # Choose somewhere to put the foreground
        xloc = random.randrange(0, xstop)
        yloc = random.randrange(0, ystop)

        # Place the foreground at (xloc,yloc)
        ambient = self.layerForeground(ambient, xloc, yloc)

        # Crop the workspace to the original background image dimensions
        cropped = ambient[xstart:xstop, ystart:ystop]

        return cropped

    def __repr__(self):
        repstring = (
            "ImageOverlay(\n"
            f"foreground={self.foreground},\n"
            f"layer={self.layer},\n"
            f"p={self.p})"
        )

    def __call__(self, data, force=False):
        img = data[self.layer][-1].result
        overlaid = self.overlay(img, self.foreground)
        data[self.layer].append(AugmentationResult(self, overlaid))
