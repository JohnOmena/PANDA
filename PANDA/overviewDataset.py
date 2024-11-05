import os
from utils import *


# There are two ways to load the data from the PANDA dataset:
# Option 1: Load images using openslide
import openslide
# Option 2: Load images using skimage (requires that tifffile is installed)
import skimage.io

# General packages
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL
from IPython.display import Image, display

# Plotly for the interactive viewer (see last section)
import plotly.graph_objs as go

# Location of the training images
data_dir = '/home/johnomena/Documents/TCC/train_images'
mask_dir = '/home/johnomena/Documents/TCC/train_label_masks'

# Location of training labels
train_labels = pd.read_csv('/home/johnomena/Documents/TCC/train.csv').set_index('image_id')

# Open the image (does not yet read the image into memory)
image = openslide.OpenSlide(os.path.join(data_dir, '005e66f06bce9c2e49142536caf2f6ee.tiff'))
# Read a specific region of the image starting at upper left coordinate (x=17800, y=19500) on level 0 and extracting a 256*256 pixel patch.
# At this point image data is read from the file and loaded into memory.
patch = image.read_region((17800,19500), 0, (256, 256))
print(patch)

# Display the image
display(patch)

# Close the opened slide after use
image.close()

example_slides = [
    '005e66f06bce9c2e49142536caf2f6ee',
    '00928370e2dfeb8a507667ef1d4efcbb',
    '007433133235efc27a39f11df6940829',
    '024ed1244a6d817358cedaea3783bbde',
]

for case_id in example_slides:
    biopsy = openslide.OpenSlide(os.path.join(data_dir, f'{case_id}.tiff'))
    print_slide_details(biopsy)
    biopsy.close()
    
    # Print the case-level label
    print(f"ISUP grade: {train_labels.loc[case_id, 'isup_grade']}")
    print(f"Gleason score: {train_labels.loc[case_id, 'gleason_score']}\n\n")
    
biopsy = openslide.OpenSlide(os.path.join(data_dir, '00928370e2dfeb8a507667ef1d4efcbb.tiff'))

x = 5150
y = 21000
level = 0
width = 512
height = 512

region = biopsy.read_region((x,y), level, (width, height))
display(region)

mask = openslide.OpenSlide(os.path.join(mask_dir, '08ab45297bfe652cc0397f4b37719ba1_mask.tiff'))
print_mask_details(mask, center='radboud')
mask.close()