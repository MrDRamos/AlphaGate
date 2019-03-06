# This script is to be filled by the team members. 
# Import necessary libraries
# Load libraries
import json
import cv2
import numpy as np

# Implement a function that takes an image as an input, performs any preprocessing steps and outputs a list of bounding box detections and assosciated confidence score. 
from matplotlib import pyplot as plt
import math
import time
from scipy.ndimage import maximum_filter1d
from scipy.signal import find_peaks, medfilt


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


def Sobel_SplitPosNeg(img, Horizontal=True, ksize=3, UseScharr= False, UseSorbel= False, UseRoberts= False):
    dtype = cv2.CV_16S
    #dtype = cv2.CV_32F
    # https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=sobel#cv2.Sobel
    if UseScharr:
        if Horizontal:
            img_grad = cv2.Scharr(img, dtype, 1, 0)
        else:
            img_grad = cv2.Scharr(img, dtype, 0, 1)
    elif UseSorbel:
        if Horizontal:
            img_grad = cv2.Sobel(img, dtype, 1, 0, ksize= ksize)
        else:
            img_grad = cv2.Sobel(img, dtype, 0, 1, ksize= ksize)
    elif UseRoberts:
        if Horizontal:
            roberst_h = np.array( [[0, 2],
                                   [-2, 0]])
            img_grad = cv2.filter2D(img, -1, roberst_h)     
        else:
            roberst_v = np.array( [[2, 0],
                                   [0,-2]])
            img_grad = cv2.filter2D(img, -1, roberst_v)
        
    if (dtype == cv2.CV_16S):
        neg_grad, pos_grad = SplitFloat_NegPosU8(img_grad)
    else:
        neg_grad, pos_grad = SplitAbs_NegPos(img_grad)

    """
    th1 = cv2.adaptiveThreshold(img_grad, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    ret,th1 = cv2.threshold(img_grad,127,255,cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(th1,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
    pos_grad = th3
    neg_grad = th3
    """

    dbg_show = False
#    dbg_show = True
    if dbg_show:
        fstat = cv2.minMaxLoc(img_grad); print(fstat)
        nstat = cv2.minMaxLoc(neg_grad); print(nstat)
        pstat = cv2.minMaxLoc(pos_grad); print(pstat)
        plt.subplot(1,4,1), plt.imshow(img_grad, 'gray'), plt.title('Float-gradiant')
        plt.subplot(1,4,2), plt.imshow(neg_grad, 'gray'), plt.title('NegU8-gradiant')
        plt.subplot(1,4,3), plt.imshow(pos_grad, 'gray'), plt.title('PosU8-gradiant')
        plt.subplot(1,4,4), plt.imshow(th3, 'gray'), plt.title('Threshold')
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
    #dbg_show = True
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
    CornerS = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, blockSize = blockSize)

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
    #plt.subplot(1,2,1), plt.imshow(img_x), plt.title(lbl_x)
    #plt.subplot(1,2,2), plt.imshow(img_y), plt.title(lbl_y)
    #plt.show()


#### 2D Edge0-Feature Histogram #####
# https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist#calchist
# https://docs.opencv.org/3.3.1/dd/d0d/tutorial_py_2d_histogram.html
def edge_hist(edges, img, bin_dx, bin_dy = None):
    if bin_dy == None:
        bin_dy = bin_dx    
    bin_x, bin_y = img.shape[1] // bin_dx, img.shape[0] // bin_dy
    rng_x, rng_y = [0, img.shape[1]],    [0, img.shape[0]]
    ex, ey       = edges[:,:,0].ravel(), edges[:,:,1].ravel()
    # Note: we revers x,y so that the numpy result has shape propotions as the input image
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
    #plt.subplot(1,3,1), plt.imshow(hst_flt), plt.subplot(1,3,2), plt.imshow(hst), plt.subplot(1,3,3), plt.imshow(gray, 'gray'), plt.show()
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
        img_dxNeg, img_dxPos = Sobel_SplitPosNeg(gray, Horizontal=True, ksize=ksize, UseSorbel= True)
        img_dyNeg, img_dyPos = Sobel_SplitPosNeg(gray, Horizontal=False, ksize=ksize, UseSorbel= True)

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
    edgeN = edgeS.shape[0]

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
        return None, None, (img_dxNeg, img_dxPos, img_dyNeg, img_dyPos)

    psumy, psumx = hsumy[xmax], hsumx[ymax]  # get the corresponding peak values
    xsrt, ysrt = np.argsort(psumy), np.argsort(psumx)  # get array to sort by peaks
    # we want coordinates of the largest 2 peaks
    px = np.array([xmax[xsrt[xsrt.size - 2]], xmax[xsrt[xsrt.size - 1]]])
    py = np.array([ymax[ysrt[ysrt.size - 2]], ymax[ysrt[ysrt.size - 1]]])
    px, py = np.sort(px), np.sort(py)  # sort the coord from low to high
    #plt.imshow(hst_01), plt.show()

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
    #plt.imshow(hst_01), plt.show()

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
    #plt.imshow(hst_01), plt.show()


    # Filter the selected edges based on the innew & outer edges of the final 2D hst
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
##    if True:
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

    return gx, gy, (img_dxNeg, img_dxPos, img_dyNeg, img_dyPos)


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
    
    # plt.subplot(2,2,1), plt.plot(sum_p)  , plt.subplot(2,2,2), plt.plot(sum_n) #plt.show()
    # plt.subplot(2,2,3), plt.imshow(hst_p), plt.subplot(2,2,4), plt.imshow(hst_n), plt.show()
    avg_p = cum_p = 0
    if 0 < peak_p.size:   
        avg_p = np.sum(np.multiply(peak_p, sum_p[peak_p]))
        cum_p = np.sum(sum_p[peak_p])
    if 0 < peak_n.size:   
        avg_p = np.sum(np.multiply(peak_n, sum_n[peak_n]))
        cum_p = np.sum(sum_n[peak_n])
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

    bx, by, (img_dxNeg, img_dxPos, img_dyNeg, img_dyPos) = find_gate_roi(gray, img_name)
    if bx is None:
      return None

    cx, cy = int((bx[0]+bx[1])/2), int((by[0]+by[1])/2)
    if False:
#    if True:       
        fig, axS = plt.subplots(2,3)
        ax = axS.ravel()        
        plotbxy(bx,by,win=ax[0]), plotcxy(cx,cy, win=ax[0])
        ax[0].imshow(img_dxNeg, 'gray'), ax[0].set_title("-dX Sobel "+ img_name)

        plotbxy(bx,by,win=ax[1]), plotcxy(cx,cy, win=ax[1])
        ax[1].imshow(img_dxPos, 'gray'), ax[1].set_title("+dX Sobel "+ img_name)

        plotbxy(bx,by,win=ax[2]), plotcxy(cx,cy, win=ax[2])
        ax[2].imshow(gray, 'gray'), ax[2].set_title(img_name)

        plotbxy(bx,by,win=ax[3]), plotcxy(cx,cy, win=ax[3])
        ax[3].imshow(img_dyNeg, 'gray'), ax[3].set_title("-dY Sobel "+ img_name)

        plotbxy(bx,by,win=ax[4]), plotcxy(cx,cy, win=ax[4])
        ax[4].imshow(img_dyPos, 'gray'), ax[4].set_title("+dY Sobel "+ img_name)

        plotbxy(bx,by,win=ax[5]), plotcxy(cx,cy, win=ax[4])
        img_dx = np.add(img_dxPos, img_dxNeg)
        img_dy = np.add(img_dyPos, img_dyNeg)
        img_dxy = np.add(img_dx, img_dy)
        ax[5].imshow(img_dxy, 'gray'), ax[4].set_title("+dY Sobel "+ img_name)
        plt.show()

    
    ### Find the exact gate corner positions by analyzing the sobel'd image
    # Gate dim: inside = 8, out = 11 ->  width/2 = 3/2 * /((8+11)/2) = 3/19
    dx, dy = (bx[1] - bx[0])/2.0 , (by[1] - by[0])/2.0 
    wx, wy = 1+int(dx * 3 / 19), 1+int(dy * 3 / 19)  # Gate width/2

#    if False: #This step takes an extra 20 ms    
    if 3 < wx and 3 < wy: # Sanity check
        ##//img_dx = np.add(img_dxNeg, img_dxPos)
        ##//img_dy = np.add(img_dyNeg, img_dyPos)

        # estimate projected (scafolding width = 1)
        if dx < dy:
            cosa = dx / dy
            wsx = int(dy / 9 * (1 - (cosa ** 2) / 2))
            wsy = int(wsx/4)
        else:
            cosa = dy / dx
            wsy = int(dx / 9 * (1 - (cosa ** 2) / 2))
            wsx = int(wsy/4)
        wx4, wy4 = wx//2, wy//2
        gwx, gwy = 2*wx + wsx,  2*wy + wsy   # Max search length=  gate width + scafolding
        vwxn,vwxp, vwy = 3*wx4,5*wx4+wsx,wy4 # -vwxn,+vwxp roi width added to anchor, vwy= +/-vertical range
        hwyn,hwyp, hwx = 3*wy4,5*wy4+wsy,wx4 # -hwyn,+hwyp roi hight added to anchor, hwx= +/-horizonal range

        # The gate positions is only accurate to pixelsPerBin=20 from find_gate_roi()
#        if False:
        if True:
            w4_pixelsPerBin=5 #20/4        
            if wx4 < w4_pixelsPerBin:
                vwxn,vwxp = 6 * w4_pixelsPerBin, 5 * w4_pixelsPerBin
            if wy4 < w4_pixelsPerBin:
                hwyn,hwyp = 6 * w4_pixelsPerBin, 5 * w4_pixelsPerBin

        dcx, dcy = 5*wx4, 5*wy4 # anchor placement of search roi's along the edge
        lft = np.array([ [bx[0]    , bx[0], bx[0]]     , [by[0]+dcy, cy   , by[1]-dcy] ])
        rht = np.array([ [bx[1]    , bx[1], bx[1]]     , [by[0]+dcy, cy   , by[1]-dcy] ])
        top = np.array([[bx[0]+dcx, cx, bx[1]-+dcx], [by[0], by[0], by[0]]])
        btm = np.array([[bx[0]+dcx, cx, bx[1]-+dcx], [by[1], by[1], by[1]]])

        patch = np.array([
            [[lft[0,0] - vwxn, lft[0,0] + vwxp], [lft[1,0] - vwy, lft[1,0] + vwy]],
            [[lft[0,1] - vwxn, lft[0,1] + vwxp], [lft[1,1] - vwy, lft[1,1] + vwy]],
            [[lft[0,2] - vwxn, lft[0,2] + vwxp], [lft[1,2] - vwy, lft[1,2] + vwy]],
            [[rht[0,0] - vwxp, rht[0,0] + vwxn], [rht[1,0] - vwy, rht[1,0] + vwy]],
            [[rht[0,1] - vwxp, rht[0,1] + vwxn], [rht[1,1] - vwy, rht[1,1] + vwy]],
            [[rht[0,2] - vwxp, rht[0,2] + vwxn], [rht[1,2] - vwy, rht[1,2] + vwy]],
            [[top[0,0] - hwx, top[0,0] + hwx], [top[1,0] - hwyn, top[1,0] + hwyp]],
            [[top[0,1] - hwx, top[0,1] + hwx], [top[1,1] - hwyn, top[1,1] + hwyp]],
            [[top[0,2] - hwx, top[0,2] + hwx], [top[1,2] - hwyn, top[1,2] + hwyp]],
            [[btm[0,0] - hwx, btm[0,0] + hwx], [btm[1,0] - hwyp, btm[1,0] + hwyn]],
            [[btm[0,1] - hwx, btm[0,1] + hwx], [btm[1,1] - hwyp, btm[1,1] + hwyn]],
            [[btm[0,2] - hwx, btm[0,2] + hwx], [btm[1,2] - hwyp, btm[1,2] + hwyn]] ])

        if False:
##        if True:
            pwx, pwy = wx * 4, wy * 4
            pbx, pby = [max(0, bx[0]-pwx), min(gray.shape[1], bx[1]+pwx)], [max(0, by[0]-pwy), min(gray.shape[0], by[1]+pwy)]
            plt_g = gray[pby[0]:pby[1], pbx[0]:pbx[1]]
            plt.imshow(plt_g, 'gray'), plt.title(img_name)
            plotbxy(bx-pbx[0], by-pby[0])

            for i in range(patch.shape[0]):
                plotbxy(patch[i,0]-pbx[0], patch[i,1]-pby[0], color="Red")

            plt_p = np.hstack((lft, rht, top, btm))
            plt_p[0, :] -= pbx[0]
            plt_p[1, :] -= pby[0]
            for i in range(plt_p.shape[1]):
                plotcxy(plt_p[0, i], plt_p[1, i], color="Red", linewidth=2)
            plt.show()

##########//
##########//
        side = 0  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 3):   # 0,1,2 patch within a side
            pn = 3*side + sn       # patch number
            lft[0,sn], lft[1,sn] = find_gate_edge(lft[0,sn], lft[1,sn], patch[pn,0], patch[pn,1], gwx, img_dxNeg, img_dxPos, axis=0, posdir=True, gray=gray, img_name=img_name)

        side = 1  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 3):   # 0,1,2 patch within a side
            pn = 3*side + sn       # patch number
            rht[0,sn], rht[1,sn] = find_gate_edge(rht[0,sn], rht[1,sn], patch[pn,0], patch[pn,1], gwx, img_dxPos, img_dxNeg, axis=0, posdir=False, gray=gray, img_name=img_name)

        side = 2  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 3):   # 0,1,2 patch within a side
            pn = 3*side + sn       # patch number
            top[0,sn], top[1,sn] = find_gate_edge(top[0,sn], top[1,sn], patch[pn,0], patch[pn,1], gwy, img_dyNeg, img_dyPos, axis=1, posdir=True, gray=gray, img_name=img_name)

        side = 3  # 0=left, 1=right, 2=top, 3=bottom
        for sn in range(0, 3):   # 0,1,2 patch within a side
            pn = 3*side + sn       # patch number
            btm[0,sn], btm[1,sn] = find_gate_edge(btm[0,sn], btm[1,sn], patch[pn,0], patch[pn,1], gwy, img_dyPos, img_dyNeg, axis=1, posdir=False, gray=gray, img_name=img_name)

        lft_l = linefit3(lft[1], lft[0]) # x = ay + b
        rht_l = linefit3(rht[1], rht[0])
        top_l = linefit3(top[0], top[1]) # y = ax + b
        btm_l = linefit3(btm[0], btm[1])
        g1 = line_intersection(top_l, lft_l)
        g2 = line_intersection(top_l, rht_l)
        g3 = line_intersection(btm_l, rht_l)
        g4 = line_intersection(btm_l, lft_l)
        bb = np.array([g1, g2, g3, g4])

##        if False:
        if True:
            pwx, pwy = wx * 4, wy * 4
            pbx, pby = [max(0, bx[0]-pwx), min(gray.shape[1], bx[1]+pwx)], [max(0, by[0]-pwy), min(gray.shape[0], by[1]+pwy)]
            plt_g = gray[pby[0]:pby[1], pbx[0]:pbx[1]]
            plt.imshow(plt_g, 'gray'), plt.title(img_name)
            plotbxy(bx - pbx[0], by - pby[0])
            
            for i in range(patch.shape[0]):
                plotbxy(patch[i, 0]-pbx[0], patch[i, 1]-pby[0], color="Red")

            plt_p = np.hstack((lft, rht, top, btm))
            plt_p[0, :] -= pbx[0]
            plt_p[1, :] -= pby[0]
            for i in range(plt_p.shape[1]):
                plotcxy(plt_p[0, i], plt_p[1, i], color="lime", linewidth=2)

            # The final gate coord
            plt_x, plt_y = bb[:, 0] - pbx[0], bb[:, 1] - pby[0]
            plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])
            plt.plot(plt_x, plt_y, color="lime", linewidth=2)
            plt.show()

    else:
        #print("TODO: Just use the ROI ##//")
        bb = np.array([ [bx[0], by[0]], [bx[1], by[0]], [bx[1], by[1]], [bx[0], by[1]] ])
    ### Find the exact gate corner positions by analyzing the sobel'd image

##xx    dbg_show = True
    dbg_show = False
    if dbg_show:
        plt.imshow(gray, 'gray'), plt.title(img_name)
        # The final gate coord
        plt_x, plt_y = bb[:, 0], bb[:, 1]
        plt_x, plt_y = np.append(plt_x, plt_x[0]), np.append(plt_y, plt_y[0])
        plt.plot(plt_x, plt_y, color="lime", linewidth=2)
        plt.show()
    return bb

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

