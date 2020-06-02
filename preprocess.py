import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from PIL import Image
from torchvision import transforms, utils

root_dir = '/Users/jason/Downloads/chest_xray/'
new_root = '/Users/jason/Downloads/small_images/'

#preprocess dataset by setting images to grayscale and resizing them to a standard size
splits = ['val/', 'train/', 'test/']
types = ['PNEUMONIA/', 'NORMAL/']
for split in splits:
    for t in types:
        images = os.listdir(root_dir + split + t)
        for image in images:
            if image == '.DS_Store':
                continue
            im = Image.open(root_dir + split + t + image)
            grayscaled = transforms.functional.to_grayscale(im)
            resized = transforms.functional.resize(grayscaled, (64, 64))
            resized.save(new_root + split + t + image)

