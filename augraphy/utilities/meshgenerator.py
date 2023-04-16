"""
Version: 0.0.1
*********************************



References:

- Perlin Noise: https://iq.opengenus.org/perlin-noise/

*********************************


"""
import math
import random


class Vector2:
    """
    Gives a 2D vector based on the input parameters.
    """

    def __init__(self, x, y):
        """
        Constructor Method that initializes the x and y components of the vector.
        """
        self.x = x
        self.y = y

    def dot(self, other):
        """
        Method that returns the dot product of the vector and another 'Vector2' object
        """
        return self.x * other.x + self.y * other.y


class Noise:
    """
    Generates 2D Perlin Noise, a type of gradient noise used to create natural looking
    textures, animations, and procedural meshes.
    This code generates smooth, continuous Perlin Noise Values that can be used for variety
    of applications in computer graphics and simulations.


    """

    def __init__(self):
        """
        Constructor method which creates the permutation table
        """
        self.permutation = self.makePermutation()

    def shuffle(self, arrayToShuffle):
        """
        Randomly shuffles array of integers from 0 to 255
        :param: arrayToShuffle
        :type: A list of integers
        """
        for e in range(len(arrayToShuffle) - 1, 0, -1):
            index = math.floor(random.random() * (e + 1))
            temp = arrayToShuffle[e]
            arrayToShuffle[e] = arrayToShuffle[index]
            arrayToShuffle[index] = temp

    def makePermutation(self):
        """
        Generates a permutation table, a randomly shuffled array of integers from 0 to 255.
        """
        permutation = []
        for i in range(256):
            permutation.append(i)
        self.shuffle(permutation)
        for i in range(256):
            permutation.append(permutation[i])
        return permutation

    def getConstantVector(self, v):
        """
        Used to generate the gradient in Perlin Noise Algorithm.
        The vector points in four of the cardinal directions.
        Returns one the four pre-defined 'Vector2' objects based on the
        last two bits of the input integer 'v'.
        :param v: Input integer from the permutation table
        :type v: int
        """
        # v is the value from the permutation table
        h = v & 3
        if h == 0:
            return Vector2(1.0, 1.0)
        elif h == 1:
            return Vector2(-1.0, 1.0)
        elif h == 2:
            return Vector2(-1.0, -1.0)
        else:
            return Vector2(1.0, -1.0)

    def fade(self, t):
        """
        Returns a smooth interpolation curve between 0 and 1
        :param t: Distance vector coordinate
        :type t: float
        """
        return ((6 * t - 15) * t + 10) * t * t * t

    def lerp(self, t, a1, a2):
        """
        Based on the Linear Interpolation Function that returns a value
        linearly interpolated between a1 and a2 based on the input parameter
        't'. It is a smoothening functions.
        :param t: Distance vector coordinate
        :type t: float
        :param a1: Dot product result from Gradient Vector and Distance Vector
        :type a1: float
        :param a2: Dot product result from Gradient Vector and Distaince Vector
        :type a2: float
        """

        return a1 + t * (a2 - a1)

    def noise2D(self, x, y):
        """
        Generates a noise mesh based on x and y coordinates given as an input. It performs the
        following steps:
        1. Calculates the integer grid coordinates of the bottom-left corner of the cell containing
        the input point, as well as the fractional offesets from that corner to the input point.
        2. Computes the dot product between the gradient vectors at each of the cell's four corners
        and the vectors pointing from those corners to the input point.
        3. Applies the fade and lerp function to interpolate(smoothening function) between these dot products
        and computes the final noise value at the input point.
        """
        X = math.floor(x) & 255
        Y = math.floor(y) & 255
        # calculating the distance vectors
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        topRight = Vector2(xf - 1.0, yf - 1.0)
        topLeft = Vector2(xf, yf - 1.0)
        bottomRight = Vector2(xf - 1.0, yf)
        bottomLeft = Vector2(xf, yf)
        # calculating the gradient vector based on the permutation table
        valueTopRight = self.permutation[self.permutation[X + 1] + Y + 1]
        valueTopLeft = self.permutation[self.permutation[X] + Y + 1]
        valueBottomRight = self.permutation[self.permutation[X + 1] + Y]
        valueBottomLeft = self.permutation[self.permutation[X] + Y]
        # the gradient vectors coordinates in the top right, top left, bottom right and bottom left
        dotTopRight = topRight.dot(self.getConstantVector(valueTopRight))
        dotTopLeft = topLeft.dot(self.getConstantVector(valueTopLeft))
        dotBottomRight = bottomRight.dot(self.getConstantVector(valueBottomRight))
        dotBottomLeft = bottomLeft.dot(self.getConstantVector(valueBottomLeft))
        # applying fade function to the distance coordinates
        u = self.fade(xf)
        v = self.fade(yf)
        # applying linear interpolation and returning the smoothened function
        return self.lerp(u, self.lerp(v, dotBottomLeft, dotTopLeft), self.lerp(v, dotBottomRight, dotTopRight))
