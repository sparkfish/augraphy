import pytest

from augraphy import *


def test_marlboro_500(random_image):
    augmented = marlboro_500(random_image)
