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

        # overlaybase should be the background
        overlayBase = self.background
        # convert background to gray
        if len(overlayBase.shape) > 2:
            overlayBase_gray = cv2.cvtColor(overlayBase, cv2.COLOR_BGR2GRAY)
        else:
            overlayBase_gray = overlayBase

        # convert foreground to gray
        if len(self.foreground.shape) > 2:
            foreground_gray = cv2.cvtColor(self.foreground, cv2.COLOR_BGR2GRAY)
        else:
            foreground_gray = self.foreground

        offsetWidth, offsetHeight = self.computeOffsets()

        fgWidth = self.foreground.shape[1]
        fgHeight = self.foreground.shape[0]

        bgWidth = self.background.shape[1]
        bgHeight = self.background.shape[0]

        if self.edge == "left":
            yloc = offsetHeight
            for i in range(self.ntimes):
                base = overlayBase[
                    yloc : (yloc + fgHeight),
                    self.edgeOffset : (self.edgeOffset + fgWidth),
                ]
                base_gray = overlayBase_gray[
                    yloc : (yloc + fgHeight),
                    self.edgeOffset : (self.edgeOffset + fgWidth),
                ]
                # can be further vectorized here, need to think about it
                for y in range(fgHeight):
                    for x in range(fgWidth):
                        if foreground_gray[y, x] < base_gray[y, x]:  # foreground is darker, get value from foreground
                            base[y, x] = self.foreground[y, x]
                yloc += fgHeight + offsetHeight

        elif self.edge == "right":
            yloc = offsetHeight
            for i in range(self.ntimes):
                base = overlayBase[
                    yloc : (yloc + fgHeight),
                    (bgWidth - (self.edgeOffset + fgWidth)) : (bgWidth - self.edgeOffset),
                ]
                base_gray = overlayBase_gray[
                    yloc : (yloc + fgHeight),
                    (bgWidth - (self.edgeOffset + fgWidth)) : (bgWidth - self.edgeOffset),
                ]
                for y in range(fgHeight):
                    for x in range(fgWidth):
                        if foreground_gray[y, x] < base_gray[y, x]:  # foreground is darker, get value from foreground
                            base[y, x] = self.foreground[y, x]
                yloc += fgHeight + offsetHeight

        elif self.edge == "top":
            xloc = offsetWidth
            for i in range(self.ntimes):
                base = overlayBase[
                    self.edgeOffset : (self.edgeOffset + fgHeight),
                    xloc : (xloc + fgWidth),
                ]
                base_gray = overlayBase_gray[
                    self.edgeOffset : (self.edgeOffset + fgHeight),
                    xloc : (xloc + fgWidth),
                ]
                for y in range(fgHeight):
                    for x in range(fgWidth):
                        if foreground_gray[y, x] < base_gray[y, x]:  # foreground is darker, get value from foreground
                            base[y, x] = self.foreground[y, x]
                xloc += fgWidth + offsetWidth

        elif self.edge == "bottom":
            xloc = offsetWidth
            for i in range(self.ntimes):
                base = overlayBase[
                    (bgHeight - (self.edgeOffset + fgHeight)) : (bgHeight - self.edgeOffset),
                    xloc : (xloc + fgWidth),
                ]
                base_gray = overlayBase_gray[
                    (bgHeight - (self.edgeOffset + fgHeight)) : (bgHeight - self.edgeOffset),
                    xloc : (xloc + fgWidth),
                ]
                for y in range(fgHeight):
                    for x in range(fgWidth):
                        if foreground_gray[y, x] < base_gray[y, x]:  # foreground is darker, get value from foreground
                            base[y, x] = self.foreground[y, x]
                xloc += fgWidth + offsetWidth

        return overlayBase
