import random

import numpy as np
import pytest

from augraphy import *


@pytest.fixture
def random_image():
    xdim = random.randint(51, 500)
    ydim = random.randint(51, 500)
    return np.random.randint(low=0, high=255, size=(xdim, ydim, 3), dtype=np.uint8)


def test_default_pipeline(random_image):
    augmented = default_augment(random_image)["output"]
