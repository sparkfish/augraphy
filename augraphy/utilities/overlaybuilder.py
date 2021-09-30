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
    :param nscales: Scales of foreground image size.
    :type nscales: tuple, optional
    :param edge: which edge of the page the foreground copies should be
        placed on
    :type edge: string
    :param edgeOffset: how far from the edge of the page to draw the copies
    :type edgeOffset: integer
    """

    def __init__(self, foreground, background, ntimes, nscales, edge, edgeOffset):
        self.foreground = foreground
        self.background = background
        self.ntimes = ntimes
        self.nscales = nscales
        self.edge = edge
        self.edgeOffset = max(0, edgeOffset)  # prevent negative

    def computeOffsets(self):
        """Determine where to place the foreground image copies"""
        xdim = self.background.shape[1]
        ydim = self.background.shape[0]

        imgWidth = self.foreground.shape[1]
        imgHeight = self.foreground.shape[0]

        remainingWidth = xdim - (self.ntimes * imgWidth)
        remainingHeight = ydim - (self.ntimes * imgHeight)

        # max to prevent negative offset
        offsetWidth = max(0, math.floor(remainingWidth / (self.ntimes + 1)))
        offsetHeight = max(0, math.floor(remainingHeight / (self.ntimes + 1)))

        return offsetWidth, offsetHeight

    def applyOverlay(
        self,
        overlayBase,
        overlayBase_gray,
        foreground_gray,
        fgHeight,
        fgWidth,
        bgHeight,
        bgWidth,
        offsetWidth,
        offsetHeight,
        ystart,
        yend,
        xstart,
        xend,
    ):
        """Applies overlay from foreground to background"""

        for i in range(self.ntimes):
            base = overlayBase[ystart:yend, xstart:xend]
            base_gray = overlayBase_gray[ystart:yend, xstart:xend]

            # can be further vectorized here, need to think about it
            for y in range(fgHeight):
                for x in range(fgWidth):
                    if foreground_gray[y, x] < base_gray[y, x]:  # foreground is darker, get value from foreground
                        # foreground is colour but base in gray
                        if len(self.foreground.shape) > len(base.shape):
                            base[y, x] = foreground_gray[y, x]
                        # same channel number
                        else:
                            base[y, x] = self.foreground[y, x]

            if self.edge == "left" or self.edge == "right":
                # for next loop ystart and yend
                ystart += fgHeight + offsetHeight
                yend = ystart + fgHeight
                # break when next ystart is > image y size
                if ystart >= bgHeight - fgHeight:
                    break
            elif self.edge == "top" or self.edge == "bottom":
                # for next loop xstart and xend
                xstart += fgWidth + offsetWidth
                xend = xstart + fgWidth
                # break when next xstart is > image x size
                if xstart >= bgWidth - fgWidth:
                    break

    def buildOverlay(self):
        """Construct the overlay image containing foreground copies"""

        # overlaybase should be the background
        overlayBase = self.background

        # sensitivity up to 0.01 (for scale down)
        min_value = max(1, self.nscales[0] * 100)
        max_value = max(1, self.nscales[1] * 100)
        new_fgHeight = int((self.foreground.shape[0] * random.randint(min_value, max_value)) / 100)
        new_fgWidth = int((self.foreground.shape[1] * random.randint(min_value, max_value)) / 100)
        self.foreground = cv2.resize(
            self.foreground,
            (int(new_fgWidth), int(new_fgHeight)),
            interpolation=cv2.INTER_AREA,
        )

        # foreground size (height & width)
        fgHeight, fgWidth = self.foreground.shape[:2]

        # background size (height & width)
        bgHeight, bgWidth = self.background.shape[:2]

        # if foreground size is larger than background, stop the operation and return background
        if (fgWidth > bgWidth) or (fgHeight > bgHeight):
            return overlayBase

        # prevent offset > background size
        if self.edge == "left" or self.edge == "right":
            if self.edgeOffset > bgWidth - fgWidth:
                self.edgeOffset = bgWidth - fgWidth
        elif self.edge == "top" or self.edge == "bottom":
            if self.edgeOffset > bgHeight - fgHeight:
                self.edgeOffset = bgHeight - fgHeight

        # compute offsets between foreground and background
        offsetWidth, offsetHeight = self.computeOffsets()

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

        if self.edge == "left":

            ystart = offsetHeight
            yend = ystart + fgHeight
            xstart = self.edgeOffset
            xend = self.edgeOffset + fgWidth

            self.applyOverlay(
                overlayBase,
                overlayBase_gray,
                foreground_gray,
                fgHeight,
                fgWidth,
                bgHeight,
                bgWidth,
                offsetWidth,
                offsetHeight,
                ystart,
                yend,
                xstart,
                xend,
            )

        elif self.edge == "right":

            ystart = offsetHeight
            yend = ystart + fgHeight
            xstart = bgWidth - self.edgeOffset - fgWidth
            xend = bgWidth - self.edgeOffset

            self.applyOverlay(
                overlayBase,
                overlayBase_gray,
                foreground_gray,
                fgHeight,
                fgWidth,
                bgHeight,
                bgWidth,
                offsetWidth,
                offsetHeight,
                ystart,
                yend,
                xstart,
                xend,
            )

        elif self.edge == "top":

            ystart = self.edgeOffset
            yend = self.edgeOffset + fgHeight
            xstart = offsetWidth
            xend = offsetWidth + fgWidth

            self.applyOverlay(
                overlayBase,
                overlayBase_gray,
                foreground_gray,
                fgHeight,
                fgWidth,
                bgHeight,
                bgWidth,
                offsetWidth,
                offsetHeight,
                ystart,
                yend,
                xstart,
                xend,
            )

        elif self.edge == "bottom":

            ystart = bgHeight - self.edgeOffset - fgHeight
            yend = bgHeight - self.edgeOffset
            xstart = offsetWidth
            xend = offsetWidth + fgWidth

            self.applyOverlay(
                overlayBase,
                overlayBase_gray,
                foreground_gray,
                fgHeight,
                fgWidth,
                bgHeight,
                bgWidth,
                offsetWidth,
                offsetHeight,
                ystart,
                yend,
                xstart,
                xend,
            )

        return overlayBase
