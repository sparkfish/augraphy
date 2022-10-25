"""This module contains a class supporting composition of AugraphyPipelines"""


class ComposePipelines:
    """The composition of multiple AugraphyPipelines.
    Define AugraphyPipelines elsewhere, then use this to compose them.
    ComposePipelines objects are callable on images (as numpy.ndarrays).

    :param pipelines: A list contains multiple augraphy.base.AugraphyPipeline.
    :type pipelines: list or tuple
    """

    def __init__(self, pipelines):
        self.pipelines = pipelines

    def __call__(self, image):

        augmented_image = image.copy()
        newpipeline = dict()

        for i, pipeline in enumerate(self.pipelines):
            data_output = pipeline.augment(augmented_image)
            augmented_image = data_output["output"]

            for key in data_output.keys():
                newkey = "pipeline" + str(i) + "-" + key
                newpipeline[newkey] = data_output[key]

        return newpipeline
