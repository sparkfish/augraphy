import pytest

from augraphy import *


def test_default_pipeline(random_image):
    augmented = default_augraphy_pipeline(random_image)
