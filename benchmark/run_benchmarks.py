# import libraries
import cv2
import numpy as np
from augraphy import *
from time import time
from tqdm import tqdm
from glob import glob
from pytablewriter import MarkdownTableWriter
import random
import argparse

# set a fixed random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
cv2.setRNGSeed(random_seed)

# function to run benchmarks
def run_benchmarks(augmentations, folder_path):

    # check jpg, png and bmp for now 
    file_paths = []
    for file_type in (["jpg", "png", "bmp"]):
        file_paths += glob(folder_path+"/*."+file_type, recursive=True)
    
    augmentations_init_time = []
    augmentations_augment_time = []
    augmentations_name = []

    for augmentation in augmentations:

        print("Running augmentation for "+augmentation.__name__)
        
        initialization_times = []
        augmentation_times = []
        for i in tqdm(range(len(file_paths))):
            image = cv2.imread(file_paths[i])
            start_time = time()
            # use default params
            augmentor = augmentation(p=1)
            # time to initialize the class
            initialization_times += [time()-start_time]

            start_time = time()
            # apply augmentation               
            image_output = augmentor(image)
            # time to augment the image
            augmentation_times += [time()-start_time]

        average_initialization_time = np.mean(initialization_times)
        average_augmentation_time   = np.mean(augmentation_times)

        augmentations_name         += [augmentation.__name__]
        augmentations_init_time    += [average_initialization_time]
        augmentations_augment_time += [average_augmentation_time]

        print("Average time per initialization = "+str(average_initialization_time)+" seconds")
        print("Average time per augmentation = "+str(average_augmentation_time)+" seconds")
        print(" ")
        
        
    return augmentations_name, augmentations_init_time, augmentations_augment_time


def generate_markdown_table(augmentations_name, augmentations_augment_time):
    
    table_name = "Benchmarking results"
    headers=["Augmentation", "Images per second"]
    value_matrix = []
    for name, augmentation_time in zip(augmentations_name, augmentations_augment_time): 
        value_matrix.append([name, np.round(1/augmentation_time,2)])
    
    writer = MarkdownTableWriter(
        table_name=table_name,
        headers=headers,
        value_matrix=value_matrix)
     
    writer.write_table()
 
       
def main(folder_path):
    
    # augmentations to be applied
    augmentations = [BadPhotoCopy, BindingsAndFasteners,BleedThrough,BookBinding,Brightness,BrightnessTexturize, ColorPaper,
                     DirtyDrum, DirtyRollers, Dithering, Faxify, Folding, Gamma, Geometric,
                     InkBleed, Jpeg, Letterpress, LightingGradient, LowInkPeriodicLines,LowInkRandomLines, Markup,
                     NoiseTexturize, PageBorder,PencilScribbles, SubtleNoise,WaterMark]

    # run and process image with each augmentation
    augmentations_name, augmentations_init_time, augmentations_augment_time = run_benchmarks(augmentations, folder_path)
        
    # generate table
    generate_markdown_table(augmentations_name, augmentations_augment_time)

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, help='Path to the folder of images')
    opt = parser.parse_args()
    
    main(opt.folder_path)

    
    


        
        

