# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results import *
import time
import os

#image_dir = 'testing/images/'
image_dir = '../../AlphaPilot/Data_Training/'

img_file = glob.glob(image_dir + '*.JPG')
img_keys = [img_i.split(os.sep)[-1] for img_i in img_file]

train = False
train = True
if train:
    data_ofs = 0
    data_ofs = img_keys.index('IMG_0013.JPG')  # large + litle glare
    #data_ofs = img_keys.index('IMG_3565.JPG') # small 
    #data_ofs = img_keys.index('IMG_4443.JPG') # + ladders
    #data_ofs = img_keys.index('IMG_4746.JPG') # + ladders + glare + angle left
    #data_ofs = img_keys.index('IMG_9180.JPG') # + ladders + glare + angle
    #data_ofs = img_keys.index('IMG_0638.JPG') # large + angle
    #data_ofs = img_keys.index('IMG_0711.JPG') # smale + angle

    train_qty = 20
    Validate_qty = 2
    img_key_train = img_keys[data_ofs : data_ofs + train_qty]
    img_key_validate = img_keys[data_ofs + train_qty : data_ofs + train_qty + Validate_qty]
else:
    img_key_train = img_keys

# Instantiate a new detector
finalDetector = GenerateFinalDetections()
# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
for img_key in img_key_train:
    img =cv2.imread(image_dir + img_key)
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tic = time.monotonic()
    bb_all = finalDetector.predict(img = img, img_name= img_key)
    toc = time.monotonic()
    pred_dict[img_key] = bb_all
    time_all.append(toc-tic)

mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time,2)
    
print('95% confidence interval for inference time is {0:.3f}ms +/- {1:.3f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

with open('random_submission.json', 'w') as f:
    json.dump(pred_dict, f)
