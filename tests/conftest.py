import random

import numpy as np
import pytest


@pytest.fixture
def random_image():
    xdim = random.randint(30, 500)
    ydim = random.randint(30, 500)
    return np.random.randint(low=0, high=255, size=(xdim, ydim, 3), dtype=np.uint8)
