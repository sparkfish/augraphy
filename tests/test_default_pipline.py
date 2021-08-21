import numpy as np
import random
import pytest

from augraphy import *

@pytest.fixture
def random_image():
    xdim = random.randint(1,500)
    ydim = random.randint(1,500)
    return np.random.randint(
        low=0,
        high=256,
        size=(xdim,ydim,3),
        dtype=np.uint8)

@pytest.fixture
def default_pipeline():
    return default_augraphy_pipeline()

def test_default_pipeline(random_image, default_pipeline):
    crappified = default_pipeline.augment(random_image)

    # just make sure at least one augmentation was applied
    assert crappified["output"] is not random_image
