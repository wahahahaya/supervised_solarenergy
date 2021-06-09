import cv2
import numpy as np
import os
import glob
import random
import matplotlib.pyplot as plt

# get bounding box

root = "data/orignal_img"
files = os.listdir(root)
idx = 0
for i in range(0, len(files)):
    dir = root + '/' + files[i]
    img = cv2.imread(dir, cv2.IMREAD_UNCHANGED)
    ret, threshed_img = cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
        0,
        255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU,
    )
    contours, hier = cv2.findContours(
        threshed_img, cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if(h > 40 and w>40 and h<70  and w<50):
            idx+=1
            box = img[y:y+h,x:x+w]
            cv2.imwrite('data/bounding_box/'+str(idx)+'.jpg', box)


# image augmentation
def horizontal_flip(img, flag):
    if flag:
        return cv2.flip(img, 1)
    else:
        return img

def vertical_flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return img

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img

def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img


root = "data/class/normal"
files = os.listdir(root)
idx = 0
for i in range(0, len(files)):
    dir = root + '/' + files[i]
    img = cv2.imread(dir)
    ar_img = vertical_shift(img, 0.9)
    cv2.imwrite('data/augument/normal/'+str(idx)+'_11'+'.jpg', ar_img)
    idx+=1

root_ab = "data/class/abnormal"
files_ab = os.listdir(root_ab)
idx_ab=0
for i in range(0, len(files_ab)):
    dir = root_ab + '/' + files_ab[i]
    img = cv2.imread(dir)
    ar_img = vertical_shift(img, 0.9)
    cv2.imwrite('data/augument/abnormal/'+str(idx_ab)+'_11'+'.jpg', ar_img)
    idx_ab+=1