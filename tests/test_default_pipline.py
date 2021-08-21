import numpy as np
import random
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


# do this 10 times to increase chances of catching stray bugs
def test_default_pipeline10(random_image, default_pipeline):
    for i in range(10):
        augmented = default_pipeline.augment(random_image)
