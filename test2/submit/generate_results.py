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
from scipy.signal import find_peaks


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
            roberst_h = np.array( [[0, 1],
                                   [-1, 0]])
            img_grad = cv2.filter2D(img, -1, roberst_h)     
        else:
            roberst_v = np.array( [[1, 0],
                                   [0,-1]])
            img_grad = cv2.filter2D(img, -1, roberst_v)
        
    if (dtype == cv2.CV_16S):
        neg_grad, pos_grad = SplitFloat_NegPosU8(img_grad)
    else:
        neg_grad, pos_grad = SplitAbs_NegPos(img_grad)

#    dbg_show = False
    dbg_show = True
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
    CornerS = cv2.goodFeaturesToTrack(img, maxCorners, qualityLevel, minDistance, blockSize = blockSize)

#    dbg_show = False
    dbg_show = True
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


def show_edge_custers(img, edge_p, clusters):
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
def show_edges(gray, edge_x, edge_y, clusters = 10):
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
        show_edge_custers(img_x, edge_x, clusters)
        show_edge_custers(img_y, edge_y, clusters)
    else:
        colorS = get_colors(2)
        for p in edge_x:
            x,y = p.ravel()
            cv2.circle(img_x, (x,y), 1, colorS[0], -1)    
        for p in edge_y:
            x,y = p.ravel()
            cv2.circle(img_y, (x,y), 1, colorS[1], -1) 
    lbl_x = "show_edges: x-sobel %d" % edge_x.shape[0]
    lbl_y = "show_edges: y-sobel %d" % edge_y.shape[0]
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
    # Note: we revers x,y so that the numpy result has shape propotions as the input image
    hist, xbins, ybins = np.histogram2d(ey, ex, [bin_y, bin_x], [rng_y, rng_x])
#    hist, xbins, ybins = np.histogram2d(ex, ey, [bin_x, bin_y], [rng_x, rng_y])
    #hist = cv2.calcHist([edge_x], [0,1], None, histSize = bins, ranges= ranges)
    return hist, xbins, ybins


def get_edges(gray, img_name= None):
    # Extract edge features
    maxCorners  = 250 #128  at lead 40/corner + Top/Btm AIRR Lables + some outliers which we will have to filter
    ksize = 3   # kernal size
    Use_PosNeg_Gradiant_Splitting = False # Tradeoff feature quantity vs speed - 100 ms
    Use_PosNeg_Gradiant_Splitting = True  # Tradeoff feature quantity vs speed  - 300 ms
    if Use_PosNeg_Gradiant_Splitting:
        # Split the sobel results into Pos,|Neg| features -> Yields More information -> more edges (28 ms- 14 ms/call)
        ksize=3
        img_dxNeg, img_dxPos = Sobel_SplitPosNeg(gray, Horizontal=True, ksize=ksize, UseScharr= False, UseSorbel= True, UseRoberts= False)
        img_dyNeg, img_dyPos = Sobel_SplitPosNeg(gray, Horizontal=False, ksize=ksize, UseScharr= False, UseSorbel= True, UseRoberts= False)
        ksize=2
        img_dxNeg, img_dxPos = Sobel_SplitPosNeg(gray, Horizontal=True,  ksize=ksize, UseScharr= False, UseSorbel= False, UseRoberts= True)
        img_dyNeg, img_dyPos = Sobel_SplitPosNeg(gray, Horizontal=False, ksize=ksize, UseScharr= False, UseSorbel= False, UseRoberts= True)

                                      

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
    #show_edges(gray, edge_x, edge_y)
    ##xx show_edges(gray, edge_x, edge_y, clusters=10)

    # Conbine X & Y edges
    edgeS = np.append(edge_x, edge_yNeg, 0)
    edgeN = edgeS.shape[0]

    #### 2D Edge0-Feature Histogram #####
    pixelsPerBin = 20 #15 # 100 # 15 for small gate
    hst, hstx, hsty = edge_hist(edgeS, gray, pixelsPerBin)
    hst = hst.astype(np.uint8)
    
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
    ########## Histogtam blob Filters ##########



    # find the predominant blob
    filtsize = 5
    hsumy, hsumx = np.sum(hst_01, axis=0), np.sum(hst_01, axis=1)
    hsumy, hsumx = maximum_filter1d(hsumy, size=filtsize), maximum_filter1d(hsumx, size=filtsize)
    hsumx[0] = hsumy[0] = hsumx[hsumx.size - 1] = hsumy[hsumy.size - 1] = 0 # make sure the borders are included
    xmax = find_peaks(hsumy)[0]
    ymax = find_peaks(hsumx)[0]

    # Sanity check in case the image has no blobs
    if 0 == xmax.size or 0 == ymax.size:
        return np.array([[],[]]), (0,0)

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
    chout = np.array([ [px[0], py[0]], [px[1], py[0]], [px[1], py[1]], [px[0], py[1]] ])
    medx, medy = (px[0] + px[1])/2.0, (py[0] + py[1])/2.0
    cx, cy = medx * pixelsPerBin, medy * pixelsPerBin
    cout = (chout + 0.5) * pixelsPerBin

    if False:
##xx    if True:
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

    if True:
        fig, axS = plt.subplots(2,3)
        ax = axS.ravel()
        ax[0].plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        ax[0].plot(pxout, pyout, color="red", linewidth=1)
        ax[0].imshow(img_dxNeg, 'gray'), ax[0].set_title("-dX Sobel "+ img_name)

        ax[1].plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        ax[1].plot(pxout, pyout, color="red", linewidth=1)
        ax[1].imshow(img_dxPos, 'gray'), ax[1].set_title("+dX Sobel "+ img_name)

        ax[2].plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        ax[2].plot(pxout, pyout, color="red", linewidth=1)
        ax[2].imshow(gray, 'gray'), ax[2].set_title(img_name)

        ax[3].plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        ax[3].plot(pxout, pyout, color="red", linewidth=1)
        ax[3].imshow(img_dyNeg, 'gray'), ax[3].set_title("-dY Sobel "+ img_name)

        ax[4].plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        ax[4].plot(pxout, pyout, color="red", linewidth=1)
        ax[4].imshow(img_dyPos, 'gray'), ax[4].set_title("+dY Sobel "+ img_name)
        plt.show()
    return cout, (cx, cy)



def my_prediction(img, img_name= None):
    gray = None
    if False:  # See if removing blue channel improves hamdling glarae: No
        b, g, r = cv2.split(img)
        plt.subplot(1, 3, 1), plt.imshow(b, 'Blues')
        plt.subplot(1, 3, 2), plt.imshow(g, 'Greens')
        plt.subplot(1, 3, 3), plt.imshow(r, 'Reds')
        plt.show()
        if False: # Remove blue(Main Gate coler) -> Increases intensity changes
            b = b * 0
        img_coler_filtered = cv2.merge((b, g, r))
        gray = cv2.cvtColor(img_coler_filtered, cv2.COLOR_RGB2GRAY)
        plt.subplot(1, 2, 1), plt.imshow(img_coler_filtered)
        plt.subplot(1, 2, 2), plt.imshow(gray)
        plt.show()
    if (gray is None):
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

    cout, (cx, cy) = get_edges(gray, img_name)

    ####### TODO #######
    ### Find the exact gate corner positions by analyzing the image 
    ### within the inner & outer box ROI's found by get_edges()
    ####### TODO #######

##xx    dbg_show = True
    dbg_show = False
    if dbg_show:
        plt.plot([cx], [cy], marker='+', markersize=15, color="red")
        pxout, pyout = np.append(cout[:, 0], cout[0, 0]), np.append(cout[:, 1], cout[0, 1])
        plt.plot(pxout, pyout, color="red", linewidth=3)
        plt.imshow(gray, 'gray'), plt.title(img_name)
        plt.show()
    return cout


class GenerateFinalDetections():
    Foo_Enable = False
    def __init__(self):
        self.Foo_Enable = True
        self.seed = 2018
        
    def predict(self, img, img_name ="na"):
        edges = my_prediction(img, img_name)
        if edges is None:
            bb_all = []
        else:
            # We could have more than 1 bb ..
            bb_all = np.array([np.append(edges, .5)])
        return bb_all.tolist()

