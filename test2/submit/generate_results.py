# This script is to be filled by the team members. 
# Import necessary libraries
# Load libraries
import os
import json
import cv2
import numpy as np

# Implement a function that takes an image as an input, performs any preprocessing steps and outputs a list of bounding box detections and assosciated confidence score. 
from matplotlib import pyplot as plt
import math
import time
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks
from scipy import polyfit

def ConvertFloat_ToU8(SrcArray, MaxFloat, NegValues = False):
    tmp = cv2.scaleAdd(SrcArray, 255.4999/MaxFloat -1, SrcArray)
    if NegValues:   # Only include negative values
        tmp.clip(max=0, out=tmp)
        u8Array = np.uint8(np.abs(tmp, out= tmp))
    else:           # Only include positive values
        u8Array = np.uint8(tmp.clip(min=0))
    return u8Array

# split the imput into pos / neg values (same shape). Return (inplace Neg),(new Pos) arrays
def SplitAbs_NegPos(SrcArray):
    pos = SrcArray.clip(min=0)
    SrcArray.clip(max=0, out=SrcArray)
    np.abs(SrcArray, out=SrcArray)
    return SrcArray, pos

def SplitFloat_NegPosU8(SrcArray):
    fstat = cv2.minMaxLoc(SrcArray)
    MaxFloat = max(fstat[1], -fstat[0]) 

    tmp = cv2.scaleAdd(SrcArray, 255.4999/MaxFloat -1, SrcArray)
    pos_U8 = np.uint8(tmp.clip(min=0))

    tmp.clip(max=0, out=tmp)
    neg_U8 = np.uint8(np.abs(tmp, out= tmp))
    return neg_U8, pos_U8 


def Sobel_SplitPosNeg(img, Horizontal=True, ksize=3, UseScharr= False):
    dtype = cv2.CV_16S
    #dtype = cv2.CV_32F
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#cv2.Sobel
    if UseScharr:
        if Horizontal:
            img_grad = cv2.Scharr(img, dtype, 1, 0)
        else:
            img_grad = cv2.Scharr(img, dtype, 0, 1)
    else:
        if Horizontal:
            img_grad = cv2.Sobel(img, dtype, 1, 0, ksize= ksize)
        else:
            img_grad = cv2.Sobel(img, dtype, 0, 1, ksize= ksize)
    if (dtype == cv2.CV_16S):
        neg_grad, pos_grad = SplitFloat_NegPosU8(img_grad)
    else:
        neg_grad, pos_grad = SplitAbs_NegPos(img_grad)

    dbg_show = False
#    dbg_show = True
    if dbg_show:
        fstat = cv2.minMaxLoc(img_grad); print(fstat)
        nstat = cv2.minMaxLoc(neg_grad); print(nstat)
        pstat = cv2.minMaxLoc(pos_grad); print(pstat)
        plt.subplot(1,3,1), plt.imshow(img_grad, 'gray'), plt.title('Float-gradiant')
        plt.subplot(1,3,2), plt.imshow(neg_grad, 'gray'), plt.title('NegU8-gradiant')
        plt.subplot(1,3,3), plt.imshow(pos_grad, 'gray'), plt.title('PosU8-gradiant')
        plt.show()
    return neg_grad, pos_grad


# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
def EqualizeIntensity(img, CLAHE=True):
    img_hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    h,l,s = cv2.split(img_hls)
    if (CLAHE):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        leq = clahe.apply(l)
    else:
        leq = cv2.equalizeHist(l)
    hls_eq = cv2.merge((h,leq,s))
    img_eq = cv2.cvtColor(hls_eq,cv2.COLOR_HLS2BGR)

    dbg_show = False
    dbg_show = True
    if dbg_show:
        plt.subplot(1,2,1), plt.hist(l.flatten(),256,[0,256], color = 'b')
        plt.subplot(1,2,2), plt.hist(leq.flatten(),256,[0,256], color = 'r')
        plt.show()
        plt.subplot(1,2,1), plt.imshow(img)
        plt.subplot(1,2,2), plt.imshow(img_eq)
        plt.show()

    return img_eq


def get_corners_xy(img, maxCorners=128):
    # https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=goodfeaturestotrack#cv2.goodFeaturesToTrack
    qualityLevel = 0.1 # Parameter characterizing the minimal accepted quality of image corners
    minDistance = 6    # Minimum possible Euclidean distance between the returned corners
    blockSize = 3      # Default =3, Size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    CornerS = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, blockSize=blockSize)
    # Try FAST feature detection
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_fast/py_fast.html

    dbg_show = False
#    dbg_show = True
    if dbg_show:
        img_abs = convert2absU8(img)
        img_rgb = cv2.cvtColor(img_abs, cv2.COLOR_GRAY2RGB)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 5.0)
        K = 10
        ret,labelS,centerS = cv2.kmeans(CornerS, K, None, criteria, 16, cv2.KMEANS_PP_CENTERS)
        rad = int(math.sqrt(ret/maxCorners))
        colorS = get_colors(K)
        for i, center in enumerate(centerS):
            color = colorS[i]
            x,y = center.astype(int).ravel()
            cv2.circle(img_rgb, (x,y), rad, color, 2)
            PointS = CornerS[labelS==i]
            for Point in PointS:
                x,y = Point.ravel()
                cv2.circle(img_rgb, (x,y), 1, color, -1)
        plt.imshow(img_rgb), plt.show()
    return CornerS


def convert2absU8(img):
    img_abs = np.absolute(img)
    _, img_max, _, _ = cv2.minMaxLoc(img_abs)
    img_U8 = ConvertFloat_ToU8(img_abs, img_max)
    return img_U8


# truncates values above 255
def scale_intensity(grayU8, factor, dst=None):
    gray = cv2.scaleAdd(grayU8, factor-1, grayU8, dst=dst)
    #gray = np.where((grayU8 * factor) > 255, 255, np.uint8(grayU8 * factor)) #Same but 10x slower!! 
    return gray


# https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
# returns a list of rbc colors that can be used as cv2 color arguments
def get_colors(count, stepsize=None):
    if stepsize == None:
        stepsize = int(180 / count)
    # we wrap the array once more with a dummy dimention, to shape it into an 
    # image as needed by cvtColor(). img_hsv[0,count]
    img_hsv = np.uint8([ [np.uint8([h*stepsize,255,255]) for h in range(count)] ])
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    rgb_colors = img_rgb[0].tolist() # unwrap the image array into a list of (r,g,b) values
    return rgb_colors


def plotbxy(px, py, color="red", linewidth=1, win=None):
    """
    plot box: px,py contain the min,max range coords
    """
    pltx = np.array([px[0], px[1], px[1], px[0], px[0]])
    plty = np.array([py[0], py[0], py[1], py[1], py[0]])
    if win is None:
        plt.plot(pltx, plty, color=color, linewidth=linewidth)
    else:
        win.plot(pltx, plty, color=color, linewidth=linewidth)


def plotcxy(cx, cy, color="red", marker='+', markersize=8, linewidth=1, win=None):
    if win is None:
        plt.plot([cx], [cy], color=color, marker=marker, markersize=markersize, linewidth=linewidth)
    else:
        win.plot([cx], [cy], color=color, marker=marker, markersize=markersize, linewidth=linewidth)


def plot_edge_clusters(img, edge_p, clusters):
    colorS = get_colors(clusters)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 5.0)
    ret,labelS,centerS = cv2.kmeans(edge_p, clusters, None, criteria, 16, cv2.KMEANS_PP_CENTERS)
    rad = int(math.sqrt(ret/edge_p.shape[0]))
    for i, center in enumerate(centerS):
        color = colorS[i]
        x,y = center.astype(int).ravel()
        cv2.circle(img, (x,y), rad, color, 2)
        PointS = edge_p[labelS==i]
        x1,x2 = int(np.min(PointS[:,0])), int(np.max(PointS[:,0]))
        y1,y2 = int(np.min(PointS[:,1])), int(np.max(PointS[:,1]))
        cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), color, 2)
        for Point in PointS:
            x,y = Point.ravel()
            cv2.circle(img, (x,y), 1, color, -1)


# img = gray 1 channnel
def plot_edge_features(gray, edge_x, edge_y, clusters = 10):
    img_abs = convert2absU8(gray)
    img_x = cv2.cvtColor(img_abs, cv2.COLOR_GRAY2RGB)
    img_x = scale_intensity(img_x, 0.5)
    img_y = np.copy(img_x)

    # Center of mass
    cxx,cxy = np.average(edge_x[:,:,0]), np.average(edge_x[:,:,1])
    cyx,cyy = np.average(edge_y[:,:,0]), np.average(edge_y[:,:,1])
    cv2.circle(img_x, (cxx, cxy), 10, (255,0,0), 4)
    cv2.circle(img_y, (cyx, cyy), 10, (255,0,0), 4)
#    n = edge_x.shape[0] + edge_y.shape[0]
#    cx, cy = int((cxx+ cyx) * edge_x.shape[0]/n) , int((cxy +cyy) * edge_y.shape[0]/n)

    if 0 < clusters:
        plot_edge_clusters(img_x, edge_x, clusters)
        plot_edge_clusters(img_y, edge_y, clusters)
    else:
        colorS = get_colors(2)
        for p in edge_x:
            x,y = p.ravel()
            cv2.circle(img_x, (x,y), 1, colorS[0], -1)    
        for p in edge_y:
            x,y = p.ravel()
            cv2.circle(img_y, (x,y), 1, colorS[1], -1) 
    lbl_x = "plot_edge_features: x-sobel %d" % edge_x.shape[0]
    lbl_y = "plot_edge_features: y-sobel %d" % edge_y.shape[0]
    plt.subplot(1,2,1), plt.imshow(img_x), plt.title(lbl_x)
    plt.subplot(1,2,2), plt.imshow(img_y), plt.title(lbl_y)
    plt.show()



#### 2D Edge0-Feature Histogram #####
# https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist#calchist
# https://docs.opencv.org/3.3.1/dd/d0d/tutorial_py_2d_histogram.html
def edge_hist(edges, img, bin_dx, bin_dy = None):
    if bin_dy == None:
        bin_dy = bin_dx    
    bin_x, bin_y = img.shape[1] // bin_dx, img.shape[0] // bin_dy
    rng_x, rng_y = [0, img.shape[1]],    [0, img.shape[0]]
    ex, ey       = edges[:,:,0].ravel(), edges[:,:,1].ravel()
    # Note: we revers x,y so that the numpy result has same shape as the input image
    hist, xbins, ybins = np.histogram2d(ey, ex, [bin_y, bin_x], [rng_y, rng_x])
#    hist, xbins, ybins = np.histogram2d(ex, ey, [bin_x, bin_y], [rng_x, rng_y])
    #hist = cv2.calcHist([edge_x], [0,1], None, histSize = bins, ranges= ranges)
    return hist, xbins, ybins


def hst_filter(hst, img_name= None):
    # removes small unconnected bins from the histogram
    ########## Histogtam blob Filters ##########
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=cv2.filter2d#cv2.filter2D
    #cv2.namedWindow("full", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("full", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # create tmp working copy of hst with 2 extra cols for 0-padding
    hst_flt = np.zeros((hst.shape[0]+4, hst.shape[1]+4), hst.dtype)
    hst_flt[2:-2,2:-2] = np.copy(hst)
    #plt.subplot(1,2,1), plt.imshow(hst_flt), plt.subplot(1,2,2), plt.imshow(hst), plt.show() 
    hst_01 = hst_flt.clip(max=1)

    flt = np.zeros_like(hst_01)
    k3 = np.array([ [ 0,-1, 0], 
                    [-1, 1,-1], 
                    [ 0,-1, 0] ])

    k41 = np.array([[ 0,-1,-1, 0], 
                    [-1, 0, 1,-1], 
                    [-1, 0, 0,-1], 
                    [ 0,-1,-1, 0] ])
    k42 = np.array([[ 0,-1,-1, 0], 
                    [-1, 1, 0,-1], 
                    [-1, 0, 0,-1], 
                    [ 0,-1,-1, 0] ])
    k43 = np.array([[ 0,-1,-1, 0], 
                    [-1, 0, 0,-1], 
                    [-1, 1, 0,-1], 
                    [ 0,-1,-1, 0] ])
    k44 = np.array([[ 0,-1,-1, 0], 
                    [-1, 0, 0,-1], 
                    [-1, 0, 1,-1], 
                    [ 0,-1,-1, 0] ])

    k5 = np.array([ [ 0,-1,-1,-1, 0], 
                    [-1, 0, 0, 0,-1], 
                    [-1, 0, 1, 0,-1], 
                    [-1, 0, 0, 0,-1], 
                    [ 0,-1,-1,-1, 0] ])
# K3
    cv2.filter2D(hst_01, -1, k3, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K4
    cv2.filter2D(hst_01, -1, k41, anchor=(2,1), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k42, anchor=(1,1), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k43, anchor=(1,2), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k44, anchor=(2,2), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(hst_01)) ####
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K3
    cv2.filter2D(hst_01, -1, k3, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K5
    cv2.filter2D(hst_01, -1, k5, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K3
    cv2.filter2D(hst_01, -1, k3, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K4
    cv2.filter2D(hst_01, -1, k41, anchor=(2,1), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k42, anchor=(1,1), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k43, anchor=(1,2), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k44, anchor=(2,2), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(hst_01)) ####
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K3
    cv2.filter2D(hst_01, -1, k3, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K5
    cv2.filter2D(hst_01, -1, k5, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K3
    cv2.filter2D(hst_01, -1, k3, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K4
    cv2.filter2D(hst_01, -1, k41, anchor=(2,1), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k42, anchor=(1,1), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k43, anchor=(1,2), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(flt)) ####
    #print(cv2.minMaxLoc(hst_01)) ####

    cv2.filter2D(hst_01, -1, k44, anchor=(2,2), dst= flt)
    flt.clip(min=0, out=flt) # <- flt contains the mask of point that will be removed
    np.subtract(hst_01, flt, out=hst_01)
    #print(cv2.minMaxLoc(hst_01)) ####
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

# K3
    cv2.filter2D(hst_01, -1, k3, dst= flt)
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_01), plt.subplot(1,3,3), plt.imshow(hst), plt.show()
    flt.clip(min=0, out=flt)
    np.subtract(hst_01, flt, out=hst_01)
    #np.multiply(hst_flt, hst_01, out= hst_flt) ###
    #plt.subplot(1,3,1), plt.imshow(flt), plt.subplot(1,3,2), plt.imshow(hst_flt), plt.subplot(1,3,3), plt.imshow(hst), plt.show()

    #print(cv2.minMaxLoc(hst_01)) ####
    np.multiply(hst_flt, hst_01, out= hst_flt)
##xx  plt.subplot(1,3,1), plt.imshow(hst_flt), plt.subplot(1,3,2), plt.imshow(hst), plt.subplot(1,3,3), plt.imshow(gray, 'gray'), plt.show()
    hst = hst_flt[2:-2, 2:-2]
    hst_01 = hst_01[2:-2,2:-2]
    return hst, hst_01
    ########## Histogtam blob Filters ##########

    

def find_gate_roi(gray, img_name= None):
    # Extract edge features
    maxCorners  = 250 #128  at lead 40/corner + Top/Btm AIRR Lables + some outliers which we will have to filter
    ksize = 3   # kernal size
    Use_PosNeg_Gradiant_Splitting = False # Tradeoff feature quantity vs speed - 100 ms
    Use_PosNeg_Gradiant_Splitting = True  # Tradeoff feature quantity vs speed  - 300 ms
    if Use_PosNeg_Gradiant_Splitting:
        # Split the sobel results into Pos,|Neg| features -> Yields More information -> more edges (28 ms- 14 ms/call)
        img_dxNeg, img_dxPos = Sobel_SplitPosNeg(gray, Horizontal=True, ksize=ksize, UseScharr= False)
        img_dyNeg, img_dyPos = Sobel_SplitPosNeg(gray, Horizontal=False, ksize=ksize, UseScharr= False)

        # (200 ms - 50 ms/img)
        edge_xNeg = get_corners_xy(img_dxNeg, maxCorners)
        edge_xPos = get_corners_xy(img_dxPos, maxCorners)
        edge_yNeg = get_corners_xy(img_dyNeg, maxCorners)
        edge_yPos = get_corners_xy(img_dyPos, maxCorners)

        # Merge the edges from the Pos,|Neg| sobel channels
        edge_x = np.append(edge_xNeg, edge_xPos, 0)
        edge_y = np.append(edge_yNeg, edge_yPos, 0)

    else:  # Faster  sobel without splitting - Result: Dont get as many edges as with splitting
        # (16 ms - 8 ms/img)
        img_dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        img_dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        # (100 ms - 50 ms/img)
        edge_x = get_corners_xy(img_dx, maxCorners)
        edge_y = get_corners_xy(img_dy, maxCorners)
    #plot_edge_features(gray, edge_x, edge_y)
    ##xx plot_edge_features(gray, edge_x, edge_y, clusters=10)

    # Conbine X & Y edges
    edgeS = np.append(edge_x, edge_y, 0)
    #edgeN = edgeS.shape[0]

    #### 2D Edge0-Feature Histogram #####
    pixelsPerBin = 20 #15 # 100 # 15 for small gate
    hst, hstx, hsty = edge_hist(edgeS, gray, pixelsPerBin)
    hst = hst.astype(np.uint8)
    
    #### Remove small unconnected bins from the histogram ####
    hst, hst_01 = hst_filter(hst)


    #### find the predominant blob ####
    filtsize = 5
    hsumy, hsumx = np.sum(hst_01, axis=0), np.sum(hst_01, axis=1)
    # plt.plot(hsumy), plt.plot(hsumx), plt.show()
    hsumy, hsumx = maximum_filter1d(hsumy, size=filtsize), maximum_filter1d(hsumx, size=filtsize)
    hsumx[0] = hsumy[0] = hsumx[hsumx.size - 1] = hsumy[hsumy.size - 1] = 0 # make sure the borders are included
    # plt.plot(hsumy), plt.plot(hsumx), plt.show()
    xmax = find_peaks(hsumy)[0]
    ymax = find_peaks(hsumx)[0]

    # Sanity check in case the image has no blobs
    if 0 == xmax.size or 0 == ymax.size:
        return None, (img_dxNeg, img_dxPos, img_dyNeg, img_dyPos)

    psumy, psumx = hsumy[xmax], hsumx[ymax]  # get the corresponding peak values
    xsrt, ysrt = np.argsort(psumy), np.argsort(psumx)  # get array to sort by peaks
    # we want coordinates of the largest 2 peaks
    px = np.array([xmax[xsrt[xsrt.size - 2]], xmax[xsrt[xsrt.size - 1]]])
    py = np.array([ymax[ysrt[ysrt.size - 2]], ymax[ysrt[ysrt.size - 1]]])
    px, py = np.sort(px), np.sort(py)  # sort the coord from low to high
    # plt.imshow(hst_01), plt.show()

    # If the gap between the 2 peeks has more 0's than data, then filter out the "strongest" peak
    span = hsumy[px[0]:px[1]]
    zeros = len([w for w in span if w == 0])
    pr = px
    psum = hsumy
    if max(0,span.size - filtsize) < 3 * zeros:
        pc = xmax[xsrt[xsrt.size - 1]]  #pm = xmax[xmax.size - 1]        
        pr[0] = pc - np.flip(psum[0:pc]).argmin(axis=0)
        pr[1] = pc + psum[pc:psum.size].argmin(axis=0)
    else:
        if 0 == zeros:
            pc = pr[0]
            pr[0] = pc - np.flip(psum[0:pc]).argmin(axis=0)
            pr[1] = pc + psum[pc:psum.size].argmin(axis=0)
    hst_01[:,:pr[0]] = 0
    hst_01[:, pr[1]:] = 0
    # plt.imshow(hst_01), plt.show()

    # If the gap between the 2 peeks has more 0's than data, then filter out the "strongest" peak
    span = hsumx[py[0]:py[1]]
    zeros = len([w for w in span if w == 0])
    pr = py
    psum = hsumx
    if max(0,span.size - filtsize) < 3 * zeros:
        pc = ymax[ysrt[ysrt.size - 1]]  #ymax[ymax.size - 1]
        pr[0] = pc - np.flip(psum[0:pc]).argmin(axis=0)
        pr[1] = pc + psum[pc:psum.size].argmin(axis=0)
    else:
        if 0 == zeros:
            pc = pr[1]
            pr[0] = pc - np.flip(psum[0:pc]).argmin(axis=0)
            pr[1] = pc + psum[pc:psum.size].argmin(axis=0)
    hst_01[:pr[0], :] = 0
    hst_01[pr[1]:, :] = 0
    # plt.imshow(hst_01), plt.show()


    # Filter the selected gate edges based on the innew & outer edges of the final 2D hst
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moment.html#scipy.stats.moment
    hst_flt = np.multiply(hst_01, hst)
    wsumy, wsumx = np.sum(hst_flt, axis=0), np.sum(hst_flt, axis=1)

    peaks = find_peaks(wsumy, prominence=1)
    pmax, psrt = peaks[0], np.argsort(peaks[1]['prominences'])
    if 1 < pmax.size:
        px = np.array([pmax[psrt[psrt.size - 2]], pmax[psrt[psrt.size - 1]]])

    peaks = find_peaks(wsumx, prominence=1)
    pmax, psrt = peaks[0], np.argsort(peaks[1]['prominences'])
    if 1 < pmax.size:
        py = np.array([pmax[psrt[psrt.size - 2]], pmax[psrt[psrt.size - 1]]])

    px, py = np.sort(px), np.sort(py) # sort the coord from low to high
    medx, medy = (px[0] + px[1])/2.0, (py[0] + py[1])/2.0
    gx, gy = np.int32((px + 0.5) * pixelsPerBin), np.int32((py + 0.5) * pixelsPerBin)


    if False:
        ##xx    if True:
        chout = np.array([ [px[0], py[0]], [px[1], py[0]], [px[1], py[1]], [px[0], py[1]] ])
        cx, cy = int((medx +0.5) * pixelsPerBin), int((medy + 0.5) * pixelsPerBin)
        cout = (chout + 0.5) * pixelsPerBin

        fig, axS = plt.subplots(1,2)
        ax = axS.ravel()
        #ax[0].plot([avgx], [avgy], marker='o', markersize=5, color="yellow")
        ax[0].plot([medx], [medy], marker='+', markersize=7, color="red")
        pxout, pyout = np.append(chout[:, 0], chout[0, 0]), np.append(chout[:, 1], chout[0, 1])
        ax[0].plot(pxout, pyout, color="red")
        ax[0].imshow(hst, interpolation='nearest'), ax[0].set_title("Hist ")
        
        ax[1].plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        ax[1].plot(pxout, pyout, color="red", linewidth=1)
        ax[1].imshow(gray, 'gray'), ax[1].set_title(img_name)
        plt.show()

    ######## Get center-of-mass for each gate corner using the raw edgepoint data
    """
    Run this step after find_gate_roi() returns a rough rectangular box ROI.
    We refine the Histogram ROI which only has a resolution of 20 pixels (=pixelsPerBin)
    We generate a large enough patch around each corner and use it to filter out the edge
    features within that corner patch. We then compute compute the center of mass for 
    each corner using the feature point coordinates.
    The final resolution is no longer dependent on the pixelsPerBin!
    """
    # Gate dim: inside = 8, out = 11 ->  width = 3/2 * /((8+11)/2) = 3/19
    gdx, gdy = gx[1] - gx[0], gy[1] - gy[0]
    wx, wy = int(gdx * 3/19), int(gdy * 3/19)  # Gate width
    gfx = gfy = 1
    if 1.5*gdx < gdy:
        gfx, gfy = 1.25, 2
    elif 1.5*gdy < gdx:
        gfx, gfy = 2, 1.25
    dwxp, dwyp = int(pixelsPerBin + gfx * wx), int(pixelsPerBin + gfy * wy)
    dwxn, dwyn = int(pixelsPerBin + 1.25 * wx), int(pixelsPerBin + 1.25 * wy)
    rx, ry = [max(0, gx[0] - dwxp), min(gray.shape[1], gx[1] + dwxp)], [max(0, gy[0] - dwyp), min(gray.shape[0], gy[1] + dwyp)]

    # Define a genrous patch around each corner. Try to avoid the center which distorts the results
    dx, dy = dwxp +dwxn, dwyp + dwyn
    e0x, e0y = np.array([rx[0], rx[0] + dx]), np.array([ry[0], ry[0] + dy])
    e1x, e1y = np.array([rx[1] -dx, rx[1]]), np.array([ry[0], ry[0] +dy])
    e2x, e2y = np.array([rx[1] -dx, rx[1]]), np.array([ry[1] -dy, ry[1]])
    e3x, e3y = np.array([rx[0], rx[0] + dx]), np.array([ry[1] - dy, ry[1]])

    # Extract the feature points within each patch
    ept = edgeS[:, 0,]
    ept0 = np.array([pnt for pnt in ept if e0x[0] < pnt[0] and pnt[0] < e0x[1] and e0y[0] < pnt[1] and pnt[1] < e0y[1] ])
    ept1 = np.array([pnt for pnt in ept if e1x[0] < pnt[0] and pnt[0] < e1x[1] and e1y[0] < pnt[1] and pnt[1] < e1y[1] ])
    ept2 = np.array([pnt for pnt in ept if e2x[0] < pnt[0] and pnt[0] < e2x[1] and e2y[0] < pnt[1] and pnt[1] < e2y[1] ])
    ept3 = np.array([pnt for pnt in ept if e3x[0] < pnt[0] and pnt[0] < e3x[1] and e3y[0] < pnt[1] and pnt[1] < e3y[1] ])

    # Compute the center of mass coordinates
    if 0 < ept0.size:
        ep0 = np.average(ept0[:, 0]), np.average(ept0[:, 1])
    else:
        ep0 = gx[0], gy[0]
    if 0 < ept1.size:
        ep1 = np.average(ept1[:, 0]), np.average(ept1[:, 1])
    else:
        ep1 = gx[1], gy[0]
    if 0 < ept2.size:
        ep2 = np.average(ept2[:,0]), np.average(ept2[:,1])
    else:
        ep2 = gx[1], gy[1]
    if 0 < ept3.size:
        ep3 = np.average(ept3[:,0]), np.average(ept3[:,1])
    else:
        ep3 = gx[0], gy[1]
    # Final 4 point bounding box
    bb = np.array([ep0, ep1, ep2, ep3])

    if False: # Show gray, Initial Hist ROI, Refined BB-Corners + EdgeFeatures
#    if True:
        img_r = gray[ry[0]:ry[1], rx[0]:rx[1]]
        bx, by = gx - rx[0], gy - ry[0]
        plt.imshow(img_r, "gray"), plotbxy(bx, by)
        plotbxy(e0x - rx[0], e0y - ry[0]), plotbxy(e1x - rx[0], e1y - ry[0])
        plotbxy(e2x - rx[0], e2y - ry[0]), plotbxy(e3x - rx[0], e3y - ry[0])
        
        # The final gate ROI coord
        plt_x, plt_y = bb[:, 0] - rx[0], bb[:, 1] - ry[0]
        plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])
        plt.plot(plt_x, plt_y, color="lime", linewidth=2)

        for pnt in ept0:
            plotcxy(pnt[0]- rx[0], pnt[1]- ry[0], color="blue")
        for pnt in ept1:
            plotcxy(pnt[0]- rx[0], pnt[1]- ry[0], color="blue")
        for pnt in ept2:
            plotcxy(pnt[0]- rx[0], pnt[1]- ry[0], color="blue")
        for pnt in ept3:
            plotcxy(pnt[0] - rx[0], pnt[1] - ry[0], color="blue")

        plt.show()
   
    return bb, (img_dxNeg, img_dxPos, img_dyNeg, img_dyPos)


def find_gate_edge(ax, ay, rx, ry, rw, img_p, img_n, axis, posdir, gray, img_name= None):
    """
    Find the max peak = gate edge
    ax,ay   : Anchor cordinates
    rx, ry  : Search Roi, tuples with min, max cordinats for x & y
    rw      : max with of gate + scafolding (if visable)
    img_p   : 1st image to find histogram peak of gate edge
              We assume that the light at the inner gate edges is captured on the posdir side,
              and it will cause a large peak at the end of the roi
    img_n   : This image is used to find the avg position of the all peaks
              which ~ the center of the gate edge
    posdir  : True when searching from left to right
    """
    if 0 == axis:
        rp = ax - rx[0]
    else:
        rp = ay - ry[0]
    rx = [max(0, rx[0]), min(img_p.shape[1], rx[1])]
    ry = [max(0, ry[0]), min(img_p.shape[0], ry[1])]

    hst_p = img_p[ry[0]:ry[1], rx[0]:rx[1]]
    sum_p = np.sum(hst_p, axis=axis)
    hgt_p = np.average(sum_p)*2
    pinf_p = find_peaks(sum_p, height=hgt_p, prominence=1)
    peak_p, psrt_p = pinf_p[0], pinf_p[1]['prominences']

    hst_n = img_n[ry[0]:ry[1], rx[0]:rx[1]]
    sum_n = np.sum(hst_n, axis=axis)
    hgt_n = np.average(sum_n)*2
    pinf_n = find_peaks(sum_n, height=hgt_n, prominence=1)
    peak_n, psrt_n = pinf_n[0], pinf_n[1]['prominences']
    
    #plt.subplot(2,2,1), plt.plot(sum_p)  , plt.subplot(2,2,2), plt.plot(sum_n) #plt.show()
    #plt.subplot(2, 2, 3), plt.imshow(hst_p), plt.subplot(2, 2, 4), plt.imshow(hst_n), plt.show()
    avg_p = cum_p = 0
    if 0 < peak_p.size:   
        avg_p = np.sum(np.multiply(peak_p, sum_p[peak_p]))
        cum_p = np.sum(sum_p[peak_p])
    if 0 < peak_n.size:   
        avg_p += np.sum(np.multiply(peak_n, sum_n[peak_n]))
        cum_p += np.sum(sum_n[peak_n])
    if 0 < cum_p:
        avg_p /= cum_p
        if posdir:
            peak_p = peak_p[avg_p <= peak_p]
            psrt_p = psrt_p[-peak_p.size:]
            peak_n = peak_n[avg_p <= peak_n]
            psrt_n = psrt_n[-peak_n.size:]
        else:
            peak_p = peak_p[peak_p <= avg_p]
            psrt_p = psrt_p[:peak_p.size]
            peak_n = peak_n[peak_n <= avg_p]
            psrt_n = psrt_n[:peak_n.size]

        max_n = max_p = int(avg_p)
        if 0 < peak_n.size:
            idx_n = np.argsort(psrt_n)[psrt_n.size-1]
            max_n = peak_n[idx_n]
        if 0 < peak_p.size:
            idx_p = np.argsort(psrt_p)[psrt_p.size-1]
            max_p = peak_p[idx_p]
        if posdir:
            if max_p < max_n and sum_p[max_p] < 2*sum_n[max_n]:
                rp = max_n
            else:
                rp = max_p
        else:
            if max_n < max_p and sum_p[max_p] < 2*sum_n[max_n]:
                rp = max_n
            else:
                rp = max_p

    # plt.subplot(2,1,1), plt.plot(sum_p), plt.subplot(2,1,2), plt.plot(sum_n), plt.show()
    # plt.plot(sum_p), plt.show()
    if 0 == axis:
        ex, ey = rx[0] + rp, ay
    else:
        ex, ey = ax, ry[0] + rp
#    if True:
    if False:
        plotbxy(rx,ry), plotcxy(ex,ey, color="Cyan")
        plt.imshow(gray, 'gray'), plt.title(img_name), plt.show()

    return ex, ey       


def my_prediction(img, img_name= None):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    # View if clipping high intenity to remove glare. Yes: its used below
    if False:
        gray_max = 255 - 20
        gray_fact = 255 / gray_max
        #gray_new = np.where((gray * gray_fact) > 255, 255, (gray * gray_fact)).astype(np.uint8)
        gray_new = np.where((gray * gray_fact) > 255, 255,  np.uint8(gray * gray_fact))
        plt.subplot(1, 2, 1), plt.imshow(gray)
        plt.subplot(1, 2, 2), plt.imshow(gray_new)
        plt.show()

    ############## Keep, This realy helps !! ############
    # Clipp highest intensities(Light bulbs & fixtures) to improve gradiants at lower intensities
#    if False:
    if True:
        # plt.imshow(gray, 'gray'), plt.title("gray - Original"), plt.show()
        #gray = np.where(gray >= 248, 0, gray)  # <- Artifially creates features in overexposed checker areas
        #gray = np.where(gray >= 248, 100, gray) # <- Artifially creates features in overexposed checker areas
        gray = scale_intensity(gray, 255 / (255 - 20), dst=gray)
        # save this trick for the end after we found the gate ROI to make all the edges light up !!!
        #gray = np.where(gray >= 248, 100, gray) # <- Artifially creates features in overexposed checker areas
        #gray = scale_intensity(gray, 255 / (255 - 20), dst=gray)
        #plt.imshow(gray, 'gray'), plt.title("gray - Preprocessed- Clip Max"), plt.show()

    bb, (img_dxNeg, img_dxPos, img_dyNeg, img_dyPos) = find_gate_roi(gray, img_name)
    if bb is None:
      return None

    if False: # Show 4 sobel's, gray & combined sobel
#    if True:       
        fig, axS = plt.subplots(2,3)
        ax = axS.ravel()

        img_dx = np.add(img_dxPos, img_dxNeg)
        img_dy = np.add(img_dyPos, img_dyNeg)
        img_dxy = np.add(img_dx, img_dy)

        plt_x, plt_y = bb[:, 0], bb[:, 1]
        plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])

        ax[0].plot(plt_x, plt_y, color="lime", linewidth=2)
        ax[0].imshow(img_dxNeg, 'gray'), ax[0].set_title("-dX Sobel "+ img_name)

        ax[1].plot(plt_x, plt_y, color="lime", linewidth=2)
        ax[1].imshow(img_dxPos, 'gray'), ax[1].set_title("+dX Sobel "+ img_name)

        ax[2].plot(plt_x, plt_y, color="lime", linewidth=2)
        ax[2].imshow(gray, 'gray'), ax[2].set_title(img_name)

        ax[3].plot(plt_x, plt_y, color="lime", linewidth=2)
        ax[3].imshow(img_dyNeg, 'gray'), ax[3].set_title("-dY Sobel "+ img_name)

        ax[4].plot(plt_x, plt_y, color="lime", linewidth=2)
        ax[4].imshow(img_dyPos, 'gray'), ax[4].set_title("+dY Sobel "+ img_name)

        ax[5].plot(plt_x, plt_y, color="lime", linewidth=2)
        ax[5].imshow(img_dxy, 'gray'), ax[4].set_title("+dY Sobel "+ img_name)
        plt.show()


    # Gate dim: inside = 8', out = 11' ->  width = 3/2 * /((8+11)/2) = 3/19
    # Gate-lengths, top=0, right=1 ...
    glen = np.array( [bb[1, 0] - bb[0, 0], bb[2, 1] - bb[1, 1], bb[2, 0] - bb[3, 0], bb[3, 1] - bb[0, 1]] )
    # Gate-width scale factor, top=0, right=1 ...
    gws = glen * 3.0 / 19
    gwx, gwy = (gws[0] + gws[2]) / 2, (gws[1] + gws[3]) / 2
    gwx, gwy = (2*gwx + gwy)/3, (gwx + 2*gwy)/3
    gw = np.array([gwy, gwx, gwy, gwx]) # note the inverted pattern

    if 6 < gw.min():  # Sanity check
        # Estimate scafolding-width protruding into the inner gate region
        if 1.25*gwx < gwy:
            # scafolding-width 1' = 2/3 Gate-Width)
            sw = gwy * 2.0/3
            sw0 = sw2 = sw / 4
            cosa = gwx / gwy
            sina = 1 - (cosa ** 2) / 2  # range 0.5 .. 1
            if glen[3] < glen[1]:
                sw1 = 0
                sw3 = sw * sina
                gw[1] *= 2-cosa
                gw[3] = gwy * cosa
            else:
                sw3 = 0
                sw1 = sw * sina
                gw[3] *= 2-cosa
                gw[1] = gwy * cosa
        elif 1.25*gwy < gwx:
            sw = gwy * 2.0/3
            sw1 = sw3 = sw / 4
            cosa = gwx / gwy
            sina = 1 - (cosa ** 2) / 2  # range 0.5 .. 1
            if glen[0] < glen[2]:
                sw2 = 0
                sw0 = sw * sina
                gw[2] *= 2-cosa
                gw[0] = gwx * cosa
            else:
                sw0 = 0
                sw2 = sw * sina
                gw[0] *= 2-cosa
                gw[2] = gwx * cosa
        else:
            sw0 = sw1 = sw2 = sw3 = min(gwx, gwy)/4
        sw = np.uint32([sw0 + .5, sw1 + .5, sw2 + .5, sw3 + .5])
        sw = sw/2  # //

        # +/= Patch width Gate-Width/4
        pw = np.uint32(gws / 4 + 0.5)
        # Patch length extending outward from the gate
        pout = np.uint32(gw * .25 + 0.5)
        # Patch length extending towards the center of the gate
        pin = np.uint32(np.add(gw * 0.75, sw) + 0.5)
        # Patch anchors offset along the sides
        pofs = np.uint32(gws * 1.25 + 0.5)
        # Patch anchors gap along the sides
        pgap = np.uint32((glen - 2 * pofs) / 3.0 + 0.5)
        # lateral skew along the sides
        gskew = np.array([bb[1, 1] - bb[0, 1], bb[2, 0] - bb[1, 0], bb[2, 1] - bb[3, 1], bb[3, 0] - bb[0, 0]])
        gskew *= 1.2 # we tend to underestamete the skes because we including not just checker-flag points
        # Patch lateral offsets per/gap
        plat = pgap * gskew / glen
        # We want 4 patches per side
        pi = np.arange(0, 4)

        # Top anchors
        px, py = np.int32(bb[0, 0] + pofs[0] + pi*pgap[0] +0.5) , np.int32(bb[0, 1] + pi*plat[0] +0.5)
        top = np.array([px,py])
        # Right hand side anchors
        px, py = np.int32(bb[1, 0] + pi*plat[1] + 0.5), np.int32(bb[1, 1] + pofs[1] + pi*pgap[1] + 0.5)
        rhs = np.array([px,py])
        # Bottom anchors
        px, py = np.int32(bb[3, 0] + pofs[2] + pi*pgap[2] +0.5) , np.int32(bb[3, 1] + pi*plat[2] +0.5)
        btm = np.array([px,py])
        # left hand side anchors
        px, py = np.int32(bb[0, 0] + pi*plat[3] + 0.5), np.int32(bb[0, 1] + pofs[3] + pi*pgap[3] + 0.5)
        lhs = np.array([px,py])

        if False:
#        if True:
            plt_p = np.hstack((top, rhs, btm, lhs))
            for i in range(plt_p.shape[1]):
                plotcxy(plt_p[0, i], plt_p[1, i], color="Red", linewidth=2)
            plt.imshow(gray, 'gray'), plt.show()

        patch = np.array([
            [[lhs[0,0] - pout[3], lhs[0,0] + pin[3]], [lhs[1,0] - pw[3], lhs[1,0] + pw[3]]],
            [[lhs[0,1] - pout[3], lhs[0,1] + pin[3]], [lhs[1,1] - pw[3], lhs[1,1] + pw[3]]],
            [[lhs[0,2] - pout[3], lhs[0,2] + pin[3]], [lhs[1,2] - pw[3], lhs[1,2] + pw[3]]],
            [[lhs[0,3] - pout[3], lhs[0,3] + pin[3]], [lhs[1,3] - pw[3], lhs[1,3] + pw[3]]],
            [[rhs[0,0] - pin[1], rhs[0,0] + pout[1]], [rhs[1,0] - pw[1], rhs[1,0] + pw[1]]],
            [[rhs[0,1] - pin[1], rhs[0,1] + pout[1]], [rhs[1,1] - pw[1], rhs[1,1] + pw[1]]],
            [[rhs[0,2] - pin[1], rhs[0,2] + pout[1]], [rhs[1,2] - pw[1], rhs[1,2] + pw[1]]],
            [[rhs[0,3] - pin[1], rhs[0,3] + pout[1]], [rhs[1,3] - pw[1], rhs[1,3] + pw[1]]],
            [[top[0,0] - pw[0], top[0,0] + pw[0]], [top[1,0] - pout[0], top[1,0] + pin[0]]],
            [[top[0,1] - pw[0], top[0,1] + pw[0]], [top[1,1] - pout[0], top[1,1] + pin[0]]],
            [[top[0,2] - pw[0], top[0,2] + pw[0]], [top[1,2] - pout[0], top[1,2] + pin[0]]],
            [[top[0,3] - pw[0], top[0,3] + pw[0]], [top[1,3] - pout[0], top[1,3] + pin[0]]],
            [[btm[0,0] - pw[2], btm[0,0] + pw[2]], [btm[1,0] - pin[2], btm[1,0] + pout[2]]],
            [[btm[0,1] - pw[2], btm[0,1] + pw[2]], [btm[1,1] - pin[2], btm[1,1] + pout[2]]],
            [[btm[0,2] - pw[2], btm[0,2] + pw[2]], [btm[1,2] - pin[2], btm[1,2] + pout[2]]],
            [[btm[0,3] - pw[2], btm[0,3] + pw[2]], [btm[1,3] - pin[2], btm[1,3] + pout[2]]] 
            ])

        if False:
#        if True:
            pwx, pwy = int(3*gws[0]), int(3*gws[1])
            bx,by = np.uint32([bb[0,0],bb[2,0]]), np.uint32([bb[0,1],bb[2,1]])
            pbx, pby = [max(0, bx[0]-pwx), min(gray.shape[1], bx[1]+pwx)], [max(0, by[0]-pwy), min(gray.shape[0], by[1]+pwy)]
            plt_g = gray[pby[0]:pby[1], pbx[0]:pbx[1]]
            plt.imshow(plt_g, 'gray'), plt.title(img_name)

            plt_x, plt_y = bb[:, 0]-pbx[0], bb[:, 1]-pby[0]
            plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])
            plt.plot(plt_x, plt_y, color="green", linewidth=2)

            for i in range(patch.shape[0]):
                plotbxy(patch[i,0]-pbx[0], patch[i,1]-pby[0], color="Red")

            plt_p = np.hstack((lhs, rhs, top, btm))
            plt_p[0, :] -= pbx[0]
            plt_p[1, :] -= pby[0]
            for i in range(plt_p.shape[1]):
                plotcxy(plt_p[0, i], plt_p[1, i], color="Red", linewidth=2)
            plt.show()

        side = 0  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 4):   # 0,1,2,3 patch within a side
            pn = 4*side + sn     # patch number
            lhs[0,sn], lhs[1,sn] = find_gate_edge(lhs[0,sn], lhs[1,sn], patch[pn,0], patch[pn,1], gwx, img_dxNeg, img_dxPos, axis=0, posdir=True, gray=gray, img_name=img_name)

        side = 1  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 4):   # 0,1,2,3 patch within a side
            pn = 4*side + sn     # patch number
            rhs[0,sn], rhs[1,sn] = find_gate_edge(rhs[0,sn], rhs[1,sn], patch[pn,0], patch[pn,1], gwx, img_dxPos, img_dxNeg, axis=0, posdir=False, gray=gray, img_name=img_name)

        side = 2  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 4):   # 0,1,2,3 patch within a side
            pn = 4*side + sn     # patch number
            top[0,sn], top[1,sn] = find_gate_edge(top[0,sn], top[1,sn], patch[pn,0], patch[pn,1], gwy, img_dyNeg, img_dyPos, axis=1, posdir=True, gray=gray, img_name=img_name)

        side = 3  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 4):   # 0,1,2,3 patch within a side
            pn = 4*side + sn     # patch number
            btm[0,sn], btm[1,sn] = find_gate_edge(btm[0,sn], btm[1,sn], patch[pn,0], patch[pn,1], gwy, img_dyPos, img_dyNeg, axis=1, posdir=False, gray=gray, img_name=img_name)

        lhs_l = linefit(lhs[1], lhs[0]) # x = ay + b
        rhs_l = linefit(rhs[1], rhs[0])
        top_l = linefit(top[0], top[1]) # y = ax + b
        btm_l = linefit(btm[0], btm[1])
        g1 = line_intersection(top_l, lhs_l)
        g2 = line_intersection(top_l, rhs_l)
        g3 = line_intersection(btm_l, rhs_l)
        g4 = line_intersection(btm_l, lhs_l)
        bb0 = bb ##//
        bb = np.array([g1, g2, g3, g4])

        if False:
            ##xx        if True:
            pwx, pwy = int(3*gws[0]), int(3*gws[1])
            bx,by = np.uint32([bb[0,0],bb[2,0]]), np.uint32([bb[0,1],bb[2,1]])
            pbx, pby = [max(0, bx[0]-pwx), min(gray.shape[1], bx[1]+pwx)], [max(0, by[0]-pwy), min(gray.shape[0], by[1]+pwy)]
            plt_g = gray[pby[0]:pby[1], pbx[0]:pbx[1]]
            plt.imshow(plt_g, 'gray'), plt.title(img_name)
            
            for i in range(patch.shape[0]):
                plotbxy(patch[i, 0]-pbx[0], patch[i, 1]-pby[0], color="Red")

            plt_p = np.hstack((lhs, rhs, top, btm))
            plt_p[0, :] -= pbx[0]
            plt_p[1, :] -= pby[0]
            for i in range(plt_p.shape[1]):
                plotcxy(plt_p[0, i], plt_p[1, i], color="lime", linewidth=2)

            plot_GT(img_name, offset=(pbx[0], pby[0]), color="Yellow", linewidth=2)

            # The final gate coord
            plt_x, plt_y = bb[:, 0] - pbx[0], bb[:, 1] - pby[0]
            plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])
            plt.plot(plt_x, plt_y, color="lime", linewidth=2)

            plt.show()

    else:
        #print("TODO: Just use the ROI ##//")
        bb = bb
    ### Find the exact gate corner positions by analyzing the sobel'd image

##xx    dbg_show = True
    dbg_show = False
    if dbg_show:
        plt.imshow(gray, 'gray'), plt.title(img_name)
        plot_GT(img_name, color="red", linewidth=2)
        # The final gate coord
        plt_x, plt_y = bb[:, 0], bb[:, 1]
        plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])
        plt.plot(plt_x, plt_y, color="lime", linewidth=2)
        plt.show()
    return bb


G_GT_LABES = None


def get_GT(img_name):
    global G_GT_LABES
    if G_GT_LABES is None:
        script_path = os.path.dirname(os.path.realpath(__file__))
        with open(script_path+"/training_GT_labels_v2.json", 'r') as f:
            G_GT_LABES = json.load(f)
    GT_box = G_GT_LABES[img_name]
    if GT_box is None or len(GT_box[0]) < 8:
        return None,None
    gt_x, gt_y = GT_box[0][::2], GT_box[0][1::2]
    gt_x, gt_y = np.append(gt_x, gt_x[0]), np.append(gt_y, gt_y[0])
    return gt_x, gt_y


def plot_GT(img_name, offset=(0,0), color="red", linewidth=2):
    pgt_x, pgt_y = get_GT(img_name)
    if not pgt_x is None:
        plt.plot(pgt_x -offset[0], pgt_y -offset[1], color="yellow", linewidth=linewidth)

def linefit(ax, ay, sensitivity=0.8):
    """
    Args: ax=[x0,x1,x2] independent variables
          ay=[y0,y1,y2]
    """
    fit = polyfit(ax, ay, 1)
    return fit[0], fit[1]

def linefit3(ax, ay, sensitivity=0.8):
    """
    Args: ax=[x0,x1,x2] (3 equally spaced independent variables)
          ay=[y0,y1,y2]
    """
    m = (ay[2] - ay[0]) / (ax[2] - ax[0]) * sensitivity
    return [m, sum(ay)/3 - m* sum(ax)/3]

def line_intersection(ly, lx):
    """
    Args: lx=[ax, bx], ly=[ay, by]  
    Horz line equ: y = ax*x + bx
    Vert line equ: x = ay*y + by
    """
    det = 1 - lx[0] * ly[0]
    return [(lx[0]*ly[1] +lx[1])/det, (ly[0]*lx[1] +ly[1])/det]


class GenerateFinalDetections():
    Foo_Enable = False
    def __init__(self):
        self.Foo_Enable = True
        self.seed = 2018
        
    def predict(self, img, img_name ="na"):
        bb = my_prediction(img, img_name)
        if bb is None:
            return [[]]
        
        # We could have more than 1 bb ..
        bb_all = np.array([np.append(bb, .5)])
        return bb_all.tolist()

