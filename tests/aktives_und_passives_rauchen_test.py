import pytest

from augraphy import *


def test_aktives_und_passives_rauchen(random_image):
    augmented = aktives_und_passives_rauchen(random_image)
