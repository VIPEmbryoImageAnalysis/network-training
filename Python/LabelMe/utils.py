import cv2
import os
import numpy as np
from config import hues, labels, imshape


def add_masks(pred):
    blank = np.zeros(shape=imshape, dtype=np.uint8)

    for i, label in enumerate(labels):

        hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
        sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
        val = pred[:,:,i].astype(np.uint8)

        im_hsv = cv2.merge([hue, sat, val])
        im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
        blank = cv2.add(blank, im_rgb)

    return blank


