# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:14:46 2023

@author: nilay
"""

import numpy as np
from gaussian import convolution
from gaussian import generate_gaussian_filter
import cv2 as cv

#converting to grayscale
def to_gray(img : np.ndarray):
    '''0.2989*R + 0.5870*G + 0.1140*B = G'''
        
    r_coef = 0.2989
    g_coef = 0.5870
    b_coef = 0.1140
        
    b,g,r = img[:,:,0] , img[:,:,1] , img[:,:,2]

    gray_im = (r_coef*r + g_coef*g + b_coef*b)/255
    return gray_im

def gradient_calculation(img: np.ndarray)->np.ndarray: #Non Maximum Suppression will also happen in this function
    
    #defining the Sobel filters for x and y axes
    Sobel_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    #performing convolution of image and filters using the convolution function imported from 
    #Gaussian blur file
    Grad_x = convolution(img, Sobel_x)
    Grad_y = convolution(img,Sobel_y)
    
    G = np.hypot(Grad_x, Grad_y) 
    G = G / G.max() * 255 #final gradient has been obtained
    theta = np.arctan2(Grad_y,Grad_x)
    
    #Now to perform non-maximum suppression on the gradient
    print(G.shape)
    M,N,Z = G.shape
    Z = np.zeros((M,N), dtype=np.int32) #resultant image array initialized
    angle = theta * 180 / np.pi #each element in array is converted to degrees
    angle[angle < 0] += 180 #negative angles converted to positive angles
    
    for i in range(1,M-1): #Note to self: most likely we are ignoring the pixels all around the edges, hence the limit
        for j in range(1,N-1): 
            q = 255
            r = 255
            
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                r = G[i, j-1] #draw grid of dots and understand this part
                q = G[i, j+1] 
            elif (22.5 <= angle[i,j] < 67.5): #didn't understand for this case, LOOK AGAIN
                r = G[i-1, j+1]
                q = G[i+1 , j-1]
            elif (67.5 <= angle[i,j] < 112.5):
                r = G[i-1, j]
                q = G[i+1, j]
            elif (112.5 <= angle[i,j] < 157.5):
                r = G[i+1, j+1]
                q = G[i-1, j-1] 
            
            if (G[i,j] >= q ) and (G[i,j] >= r): 
                Z[i,j] = G[i,j] #corresponding output pixel set to current pixel
            else:
                Z[i,j] = 0 #corresponding pixel set to black
    
    return Z

def threshold(img, lowThresholdRatio = 0.05, highThresholdRatio = 0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold*lowThresholdRatio
    
    M,N = img.shape
    res = np.zeros((M,N), dtype = np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    
    #finding locations if pixels as per threshold conditions
    strong_i, strong_j = np.where(img >= highThreshold)
    zero_i , zero_j = np.where(img<lowThreshold)
    weak_i , weak_j = np.where((img <= highThreshold) & (img>=lowThreshold))
    
    #thresholding the pixels in the res image according to rule
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak      
    
    return (res,weak,strong)

def hysteresis(img, weak, strong=255): #weak pixels set to strong if connected to strong pixels, else 0
    M, N = img.shape
    
    for i in range(1,M-1):
        for j in range(2,N-1):
            if(img[i,j] == weak):
                if(
                    (img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)
                        ):
                    img[i,j] = strong
                else:
                    img[i,j]=0
    return img
    

path = "D:\MY FILES\Image Processing\Canny Edge Detection Project\kitten.jpg"
#Import image from file/source
im = cv.imread(path, 1)
cv.imshow("image", im)
#Convert to grayscale
gray_im = to_gray(im)
cv.imshow("Gray", gray_im)
#Gaussian blur the image
gauss_filter = generate_gaussian_filter(2.5 , [11,11])
blurred = convolution(gray_im*255, gauss_filter)
cv.imshow("Blurred" , blurred)
#Perform gradient calculation using Sobel filters
#Non Maximum Suppression
suppressed_img = gradient_calculation(blurred)
cv.imshow("Suppr", suppressed_img.astype(np.uint8))
#Double thresholding and hysteresis
threshold_img, weak = threshold(suppressed_img)[0:2]
cv.imshow("thresholded", threshold_img.astype(np.uint8))
final = hysteresis(threshold_img,weak)
cv.imshow("final", final.astype(np.uint8))
cv.waitKey(0)
        
        