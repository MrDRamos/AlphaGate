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

# Run this script from the AlphaGate directory
# python test2/submit/generate_submission.py
script_path = os.path.dirname(os.path.realpath(__file__))
image_dir = script_path + '/../../../Data_Training/'
#image_dir = script_path + '/../../../Data_LeaderboardTesting/'
ResultFilename = script_path + "/airvision_submission.json"
#ResultFilename = script_path + "/random_submission.json"

img_file = glob.glob(image_dir + '*.JPG')
img_keys = [img_i.split(os.sep)[-1] for img_i in img_file]



def UseTrainingImages(qty=20):
    global img_keys; data_ofs = 0
    data_ofs = img_keys.index('IMG_0013.JPG')  # large + litle glare
    #data_ofs = img_keys.index('IMG_3565.JPG') # small 
    ## data_ofs = img_keys.index('IMG_4443.JPG') # + ladders
    #data_ofs = img_keys.index('IMG_4746.JPG') # + ladders + glare + angle left
    ## data_ofs = img_keys.index('IMG_9180.JPG')  # + angle + partial gates
    #data_ofs = img_keys.index('IMG_0638.JPG')  # large + angle

    #data_ofs = img_keys.index('IMG_0664.JPG')  # large + angle
    data_ofs = img_keys.index('IMG_0711.JPG') # small + angle
    data_ofs = img_keys.index('IMG_1625.JPG') # no gate, 492, 5199

    #data_ofs = img_keys.index('IMG_3574.JPG')
    img_keys = img_keys[data_ofs: data_ofs + qty]

UseTrainingImages()

# Instantiate a new detector
finalDetector = GenerateFinalDetections()
# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
n = len(img_keys)
i = 0
for img_key in img_keys:
    i += 1
    print("{0} {1:4d}/{2}".format(img_key, i, n))
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

if not (ResultFilename is None):
    with open(ResultFilename, 'w') as f:
        json.dump(pred_dict, f)
