"""This module contains a class supporting composition of AugraphyPipelines"""

from augraphy.base.augmentationpipeline import AugraphyPipeline
from augraphy.base.augmentationresult import AugmentationResult

class ComposePipelines:
    """The composition of two AugraphyPipelines.
    Define both AugraphyPipelines elsewhere, then use this to compose them.
    ComposePipelines objects are callable on images (as numpy.ndarrays).

    :param first: The first AugraphyPipeline to apply.
    :type first: augraphy.base.AugraphyPipeline
    :param second: The second AugraphyPipeline to apply.
    :type second: augraphy.base.AugraphyPipeline
    """
    def __init__(self, first, second):
        self.first = first
        self.second = second


    def __call__(self, image):
        pipeline1 = self.first.augment(image)
        pipeline2 = self.second.augment(pipeline1["output"])

        newpipeline = dict()
        for key in pipeline1.keys():
            newkey = "pipeline1-" + key
            newpipeline[newkey]= pipeline1[key]

        for key in pipeline2.keys():
            newkey = "pipeline2-" + key
            newpipeline[newkey]= pipeline2[key]

        return newpipeline
