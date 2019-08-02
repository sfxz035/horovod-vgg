import skimage
import skimage.io
import skimage.transform
import numpy as np
import cv2 as cv
import os

def load_image(path):
    # load image
    img = skimage.io.imread(path)
    # img = img / 255.0
    # assert (0 <= img).all() and (img <= 1.0).all()
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    cv.namedWindow('a', 0)
    cv.imshow('a', resized_img)
    cv.waitKey(0)
    return resized_img

def read(path):
    class_files = os.listdir(path)
    name_list = []
    with open(path,'r') as f:
        for lines in f.readlines():
            a = lines.strip().split(' ',1)
            if a[1] == ' 1':
                a[1] = '0'
                name_list += [a]


path = 'C:\\Users\suof\Desktop\VOC2012\ImageSets\Main'
read(path)
# path = 'C:\\Users\suof\Desktop\VOC2012\JPEGImages\\2007_000027.jpg'
# load_image(path)