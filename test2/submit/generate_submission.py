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
image_dir = 'G:/AlphaPilot/Data_Training/'

img_file = glob.glob(image_dir + '*.JPG')
img_keys = [img_i.split(os.sep)[-1] for img_i in img_file]

data_ofs = 0
data_ofs = img_keys.index('IMG_0013.JPG') 
#data_ofs = img_keys.index('IMG_0013.JPG') 
#data_ofs = img_keys.index('IMG_3565.JPG') # small
train_qty = 6
Validate_qty = 2
img_key_train = img_keys[data_ofs : data_ofs + train_qty]
img_key_validate = img_keys[data_ofs + train_qty : data_ofs + train_qty + Validate_qty]

####################
from matplotlib import pyplot as plt
import math

def ConvertFloat_ToU8(SrcArray, MaxFloat, NegValues = False):
    scalefactor = 255.9999/MaxFloat
    tmp = np.copy(SrcArray) * scalefactor
    if NegValues:
        tmp[0 < tmp] = 0 # Only include negative values
        u8Array = np.uint8(-tmp)
    else:
        tmp[tmp < 0] = 0 # Only include positive values
        u8Array = np.uint8(tmp)
    return u8Array

def Split_FloatToU8(SrcArray):
    fstat = cv2.minMaxLoc(SrcArray)
    MaxFloat = max(fstat[1], -fstat[0])    
    neg_U8 = ConvertFloat_ToU8(SrcArray, MaxFloat, NegValues= True)
    pos_U8 = ConvertFloat_ToU8(SrcArray, MaxFloat, NegValues= False)
    return neg_U8,pos_U8 

def Grad_NegPos(img, Horizontal=True, ksize=3, UseScharr= False):
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#cv2.Sobel
    if UseScharr:
        if Horizontal:
            img_grad = cv2.Scharr(img, cv2.CV_16S, 1, 0)
        else:
            img_grad = cv2.Scharr(img, cv2.CV_16S, 0, 1)
    else:
        if Horizontal:
            img_grad = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize= ksize)
        else:
            img_grad = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize= ksize)
    
    neg_grad, pos_grad = Split_FloatToU8(img_grad)

    dbg_show = False
    #dbg_show = True
    if dbg_show:
        fstat = cv2.minMaxLoc(img_grad); print(fstat)
        nstat = cv2.minMaxLoc(neg_grad); print(nstat)
        pstat = cv2.minMaxLoc(pos_grad); print(pstat)
        plt.subplot(1,3,1), plt.imshow(img_grad, 'gray'), plt.title('Float-gradiant')
        plt.subplot(1,3,2), plt.imshow(neg_grad, 'gray'), plt.title('NegU8-gradiant')
        plt.subplot(1,3,3), plt.imshow(pos_grad, 'gray'), plt.title('PosU8-gradiant')
        plt.show()
    return neg_grad, pos_grad


def Get_Corners(img):
    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#cv2.goodFeaturesToTrack
    maxCorners  = 128
    qualityLevel = 0.1 # Parameter characterizing the minimal accepted quality of image corners
    minDistance = 1    # Minimum possible Euclidean distance between the returned corners
    blockSize = 3      # Default =3, Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    CornerS = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, blockSize = blockSize)

    dbg_show = False
    dbg_show = True
    if dbg_show:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#######
        colorS = ((255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(255,255,255),
                  (195,30,30),(30,195,30),(30,30,255),(195,195,120),(195,120,195),(120,195,195),(195,195,195))
        #colorS = [list(np.random.choice(range(256), size= 3)) for i in range(5)]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 5.0)
        K = 10
        ret,labelS,centerS = cv2.kmeans(CornerS, K, None, criteria, 16, cv2.KMEANS_PP_CENTERS)
        rad = int(math.sqrt(ret/maxCorners))
        for i, center in enumerate(centerS):
            color = colorS[i]
            x,y = center.astype(int).ravel()
            cv2.circle(img_rgb, (x,y), rad, color, 2)
            PointS = CornerS[labelS==i]
            for Point in PointS:
                x,y = Point.ravel()
                cv2.circle(img_rgb, (x,y), 1, color, -1)
#######
        """
        color = [255,0,0] 
        for Corner in CornerS:
            x,y = Corner.ravel()
            cv2.circle(img_rgb, (x,y), 1, color, -1)
        """            
        plt.imshow(img_rgb), plt.show()
    return CornerS



# Corner detection
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html#harris-corners
DoCorner = True
if DoCorner:
    idx = 0
    for img_key in img_key_train:
        img =cv2.imread(image_dir + img_key)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_dxmin, img_dxmax = Grad_NegPos(gray, Horizontal=True, UseScharr= False)
        img_dymin, img_dymax = Grad_NegPos(gray, Horizontal=False, UseScharr= False)

        edge_xmin = Get_Corners(img_dxmin)
        edge_xmax = Get_Corners(img_dxmax)
        edge_ymin = Get_Corners(img_dymin)
        edge_ymax = Get_Corners(img_dymax)

        """
        #https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
        # Next: Do K-Means Cluster analysis of the corner points 
        # Divide them into k=4 Gate-corners
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
        K = 10
        ret,label,center=cv2.kmeans(fcorners, K, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
        # Now separate the data, Note the flatten()
        corners = np.int0(fcorners)
        Group = []
        Group.append( fcorners[label.ravel()==0] )
        Group.append( fcorners[label.ravel()==1] )
        Group.append( fcorners[label.ravel()==2] )
        Group.append( fcorners[label.ravel()==3] )
        Group.append( fcorners[label.ravel()==4] )
        Group.append( fcorners[label.ravel()==5] )
        Group.append( fcorners[label.ravel()==6] )
        Group.append( fcorners[label.ravel()==7] )
        Group.append( fcorners[label.ravel()==8] )
        Group.append( fcorners[label.ravel()==9] )

        cn = [[0,0,255], [0,255,0], [255,0,0]]
        ci = 0
        for g in Group:            
            color = cn [ci % 3]
            for i in g:
                x,y = i.ravel()
                cv2.circle(img, (x,y), 3, color, -1)
            ci += 1


        #plt.subplot(2,3,idx+1), 
        plt.imshow(img)
        plt.title(img_key)
        plt.xticks([]),plt.yticks([])
        plt.show()
        idx += 1
        """
os._exit(0)
####################


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
    
print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

with open('random_submission.json', 'w') as f:
    json.dump(pred_dict, f)
