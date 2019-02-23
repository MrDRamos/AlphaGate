# This script is to be filled by the team members. 
# Import necessary libraries
# Load libraries
import json
import cv2
import numpy as np

# Implement a function that takes an image as an input, performs any preprocessing steps and outputs a list of bounding box detections and assosciated confidence score. 
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


def Sobel_SplitPosNeg(img, Horizontal=True, ksize=3, UseScharr= False):
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
    #dbg_show = True
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

# truncates values above 255
def scale_intensity(grayU8, factor):
    gray = cv2.scaleAdd(grayU8, factor-1, grayU8)
    #gray = np.where((grayU8 * factor) > 255, 255, np.uint8(grayU8 * factor)) #Same but 10x slower!! 
    return gray

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
    lbl_x = "x-sobel %d" % edge_x.shape[0]
    lbl_y = "y-sobel %d" % edge_y.shape[0]
    plt.subplot(1,2,1), plt.imshow(img_x), plt.title(lbl_x)
    plt.subplot(1,2,2), plt.imshow(img_y), plt.title(lbl_y)
    plt.show()


def get_edges(gray):
    # Extract edge features
    maxCorners  = 250 #128  at lead 40/corner + Top/Btm AIRR Lables + some outliers which we will have to filter
    ksize = 3   # kernal size
    Use_PosNeg_Gradiant_Splitting = True # Tradeoff feature quantity vs speed
    if Use_PosNeg_Gradiant_Splitting:
        # Split the sobel results into Pos,|Neg| features -> Yields More information -> more edges
        img_dxNeg, img_dxPos = Sobel_SplitPosNeg(gray, Horizontal=True, ksize=ksize, UseScharr= False)
        img_dyNeg, img_dyPos = Sobel_SplitPosNeg(gray, Horizontal=False, ksize=ksize, UseScharr= False)

        edge_xNeg = get_corners_xy(img_dxNeg, maxCorners)
        edge_xPos = get_corners_xy(img_dxPos, maxCorners)
        edge_yNeg = get_corners_xy(img_dyNeg, maxCorners)
        edge_yPos = get_corners_xy(img_dyPos, maxCorners)

        # Merge the edges from the Pos,|Neg| sobel channels
        edge_x = np.append(edge_xNeg, edge_xPos, 0)
        edge_y = np.append(edge_yNeg, edge_yPos, 0)

    else:  # Faster sobel without splitting - Result: Dont get as many edges as with splitting
        img_dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
        img_dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
        edge_x = get_corners_xy(img_dx, maxCorners)
        edge_y = get_corners_xy(img_dy, maxCorners)
    #show_edges(gray, edge_x, edge_y)
    #show_edges(gray, edge_x, edge_y, clusters=10)
    
    # Center of mass
    cxx,cxy = np.average(edge_x[:,:,0]), np.average(edge_x[:,:,1])
    cyx,cyy = np.average(edge_y[:,:,0]), np.average(edge_y[:,:,1])
    n = edge_x.shape[0] + edge_y.shape[0]
    cx, cy = int((cxx+ cyx) * edge_x.shape[0]/n) , int((cxy +cyy) * edge_y.shape[0]/n)

    # mean radius of all edges
    rxx,rxy = np.average(np.abs(edge_x[:,:,0] -cx)), np.average(np.abs(edge_x[:,:,1] -cy))
    ryx,ryy = np.average(np.abs(edge_y[:,:,0] -cx)), np.average(np.abs(edge_y[:,:,1] -cy))
    rx, ry = int((rxx + ryx)/2), int((rxy + ryy)/2)
    
    # 0'th order gate position
    x1, y1 = cx +rx, cy -ry
    x2, y2 = cx -rx, cy -ry
    x3, y3 = cx -rx, cy +ry
    x4, y4 = cx +rx, cy +ry
    return  (x1,y1), (x2,y2), (x3,y3), (x4,y4)


def my_prediction(img):
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
    if True:
        gray = scale_intensity(gray, 255 / (255 - 20))

    (x1,y1), (x2,y2), (x3,y3), (x4,y4) = get_edges(gray)
    dbg_show = True
    #dbg_show = False
    if dbg_show:
        plt.imshow(img)
        plt.plot([x1,x2,x3,x4,x1],  [y1,y2,y3,y4,y1], color='r', linewidth=3)
        plt.show()
    return  (x1,y1), (x2,y2), (x3,y3), (x4,y4)

    #### 2D Edge0-Feature Histogram #####
    # https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=calchist#calchist
    # https://docs.opencv.org/3.3.1/dd/d0d/tutorial_py_2d_histogram.html
    """
Problem:  The images must have the same size
    images = [img1,img2,img3]
    channels = [0,2] # e.g. first & last
    mask = None
    bins = [20,16]
    ranges [min1,max1, min2,max2]
    hist = cv2.calcHist(images, channels, mask, histSize = bins, ranges)

    # Use matplotlib.pyplot.imshow() function to plot 2D histogram with different color maps.
    plt.imshow(hist, interpolation = 'nearest')
    plt.show()
"""


class GenerateFinalDetections():
    Foo_Enable = False
    def __init__(self):
        self.Foo_Enable = True
        self.seed = 2018
        
    def predict(self,img,img_name="na"):
        (x1,y1), (x2,y2), (x3,y3), (x4,y4) = my_prediction(img)
        bb_all = np.array([x1,y1,x2,y2,x3,y3,x4,y4,0.5])
        """
        np.random.seed(self.seed)
        n_boxes = np.random.randint(4)
        if n_boxes>0:
            bb_all = 400*np.random.uniform(size = (n_boxes,9))
            bb_all[:,-1] = 0.5
        else:
            bb_all = []
        """            
        return bb_all.tolist()
        
