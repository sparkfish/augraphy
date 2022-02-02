import pytest

from augraphy import *


def test_ira_jay_goldberg(random_image):
    augmented = ira_jay_goldberg(random_image)
