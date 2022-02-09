import pytest

from augraphy import *


def test_diahann_carroll(random_image):
    augmented = diahann_carroll(random_image)
