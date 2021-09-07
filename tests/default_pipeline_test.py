import random

import numpy as np
import pytest

from augraphy import *


@pytest.fixture
def random_image():
    xdim = random.randint(30, 500)
    ydim = random.randint(30, 500)
    return np.random.randint(low=0, high=255, size=(xdim, ydim, 3), dtype=np.uint8)


@pytest.fixture
def default_pipeline():
    return default_augraphy_pipeline()


def test_default_pipeline(random_image, default_pipeline):
    augmented = default_pipeline.augment(random_image)
