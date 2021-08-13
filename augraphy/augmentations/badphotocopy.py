import numpy as np
import random
import cv2

from augraphy.base.augmentation import Augmentation
from augraphy.base.augmentationresult import AugmentationResult
from augraphy.augmentations.lib import create_blob_with_shape, applyBlob_corners, applyBlob_random, applyBlob_full

class BadPhotocopy(Augmentation):
    """Create bad photocopy effect by adding various patterns of size varying
    blobs.  
    :param blob_density: % Amount of blobs in the image, range from 0-1.
    :type blob_density: float, optional
    :param p: The probability this Augmentation will be applied.
    :type p: float, optional
    """
    def __init__(
        self,
        blob_density=0.5,
        p=0.5
    ):
        super().__init__(p=p)
        self.blob_density = blob_density

    # Constructs a string representation of this Augmentation.
    def __repr__(self):
        return f"BadPhotocopy(blob_density={self.blob_density},p={self.p})"
 
    # Apply blobs pattern to create badphotocopy effect
    # You may change and modify this section for different blobs' pattern
    def apply_blobs_pattern(self, img, blob_density):
        
        img_bad= img.copy(0)
        img_bad = applyBlob_full(img_bad, blob_density)
         
        # random choice to apply blobs at image corner
        f_topleft = random.choice([0,1])
        f_topright = 1-f_topleft
        f_bottomleft = random.choice([0,1])
        f_bottomright = 1-f_bottomleft
              
        applyBlob_corners(
                img_bad,
                size_range=(40, 40),
                points_range=(500*blob_density, 500*blob_density),
                std_range=(150, 200),
                features_range=(25, 50),
                value_range=(5, 25),
                scale_blob=(3,4),
                f_topleft= f_topleft,
                f_topright= f_topright,
                f_bottomleft= f_bottomleft,
                f_bottomright= f_bottomright,
                inverse=1
                )
        
        applyBlob_corners(
                img_bad,
                size_range=(60, 60),
                points_range=(500*blob_density, 500*blob_density),
                std_range=(150, 200),
                features_range=(25, 50),
                value_range=(5, 25),
                scale_blob=(2,3),
                f_topleft= f_topleft,
                f_topright= f_topright,
                f_bottomleft= f_bottomleft,
                f_bottomright= f_bottomright,
                inverse=1
                )
        
        applyBlob_corners(
                img_bad,
                size_range=(80, 80),
                points_range=(800*blob_density, 800*blob_density),
                std_range=(150, 200),
                features_range=(25, 50),
                value_range=(5, 25),
                scale_blob=(1,2),
                f_topleft= f_topleft,
                f_topright= f_topright,
                f_bottomleft= f_bottomleft,
                f_bottomright= f_bottomright,
                inverse=1
                )
          
        applyBlob_corners(
                img_bad,
                size_range=(70, 70),
                points_range=(800*blob_density, 800*blob_density),
                std_range=(150, 200),
                features_range=(25, 50),
                value_range=(5, 25),
                scale_blob=(2,3),
                f_topleft= f_topleft,
                f_topright= f_topright,
                f_bottomleft= f_bottomleft,
                f_bottomright= f_bottomright,
                inverse=1
                )
        
        applyBlob_random(
                img_bad,
                size_range=(80, 80),
                points_range=(5*blob_density, 5*blob_density),
                std_range=(150, 200),
                features_range=(25, 50),
                value_range=(5, 25),
                scale_blob=(2,3),
                inverse=1
                )
        
        applyBlob_random(
                img_bad,
                size_range=(40, 40),
                points_range=(5*blob_density, 5*blob_density),
                std_range=(150, 200),
                features_range=(25, 50),
                value_range=(5, 25),
                scale_blob=(1,3),
                inverse=1
                )
        
        # convert to binary
        img_bad[img_bad>150] = 255
        img_bad[img_bad<255] = 0

        return img_bad
    
    # Applies the Augmentation to input data.
    def __call__(self, data, force=False):
        if force or self.should_run():
            image = data["ink"][-1].result.copy()
            
            # convert to non coloured image
            if len(image.shape)>2: 
                image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # apply blobs pattern to create badphotocopy effect
            image_badphotocopy = self.apply_blobs_pattern(image, self.blob_density)

            data["ink"].append(AugmentationResult(self, image_badphotocopy))

