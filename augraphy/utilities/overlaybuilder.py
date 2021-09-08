import math
import random

import cv2
import numpy as np


class OverlayBuilder:
    """Takes an input image, a number of times to duplicate that image, an image
    on which to overlay the result of this, and a page position, then produces
    an overlayable image with the input image copied that many times, spaced
    regularly across the edge of the page in the given position.

    :param foreground: the image to overlay on the background document
    :type foreground: np.array
    :param background: the document
    :type background: np.array
    :param ntimes: how many copies of the foreground image to draw
    :type ntimes: integer
    :param edge: which edge of the page the foreground copies should be
        placed on
    :type edge: string
    :param edgeOffset: how far from the edge of the page to draw the copies
    :type edgeOffset: integer
    """

    def __init__(self, foreground, background, ntimes, edge, edgeOffset):
        self.foreground = foreground
        self.background = background
        self.ntimes = ntimes
        self.edge = edge
        self.edgeOffset = edgeOffset

    def computeOffsets(self):
        """Determine where to place the foreground image copies"""
        xdim = self.background.shape[1]
        ydim = self.background.shape[0]

        imgWidth = self.foreground.shape[1]
        imgHeight = self.foreground.shape[0]

        remainingWidth = xdim - (self.ntimes * imgWidth)
        remainingHeight = ydim - (self.ntimes * imgHeight)

        offsetWidth = math.floor(remainingWidth / (self.ntimes + 1))
        offsetHeight = math.floor(remainingHeight / (self.ntimes + 1))

        return offsetWidth, offsetHeight

    def buildOverlay(self):
        """Construct the overlay image containing foreground copies"""
        xdim = self.background.shape[1]
        ydim = self.background.shape[0]

        overlayBase = np.ones((ydim, xdim, 3))

        offsetWidth, offsetHeight = self.computeOffsets()

        fgWidth = self.foreground.shape[1]
        fgHeight = self.foreground.shape[0]

        bgWidth = self.background.shape[1]
        bgHeight = self.background.shape[0]

        if self.edge == "left":
            yloc = offsetHeight
            for i in range(self.ntimes):
                overlayBase[
                    yloc : (yloc + fgHeight),
                    self.edgeOffset : (self.edgeOffset + fgWidth),
                ] = self.foreground
                yloc += fgHeight + offsetHeight

        elif self.edge == "right":
            yloc = offsetHeight
            for i in range(self.ntimes):
                overlayBase[
                    yloc : (yloc + fgHeight),
                    (bgWidth - (self.edgeOffset + fgWidth)) : (
                        bgWidth - self.edgeOffset
                    ),
                ] = self.foreground
                yloc += fgHeight + offsetHeight

        elif self.edge == "top":
            xloc = offsetWidth
            for i in range(self.ntimes):
                overlayBase[
                    self.edgeOffset : (self.edgeOffset + fgHeight),
                    xloc : (xloc + fgWidth),
                ] = self.foreground
                xloc += fgWidth + offsetWidth

        elif self.edge == "bottom":
            xloc = offsetWidth
            for i in range(self.ntimes):
                overlayBase[
                    (bgHeight - (self.edgeOffset + fgHeight)) : (
                        bgHeight - self.edgeOffset
                    ),
                    xloc : (xloc + fgWidth),
                ] = self.foreground
                xloc += fgWidth + offsetWidth

        return overlayBase
