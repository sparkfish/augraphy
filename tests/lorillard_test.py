import pytest

from augraphy import *


def test_lorillard(random_image):
    augmented = lorillard(random_image)
