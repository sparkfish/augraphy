import pytest

from augraphy import *


def test_heparin_testosterone_rats(random_image):
    augmented = heparin_testosterone_rats(random_image)
