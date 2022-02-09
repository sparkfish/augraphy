import pytest

from augraphy import *


def test_statesmans_manual(random_image):
    augmented = statesmans_manual(random_image)
