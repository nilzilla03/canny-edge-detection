# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:08:34 2023

@author: nilay
"""

import numpy as np
from PIL import Image
import cv2 as cv

#generating the kernel
def generate_gaussian_filter(sigma : float, filter_shape : list):
    #sigma is standard deviation
    
    m,n = filter_shape
    m_half = m//2
    n_half = n //2
    
    gaussian_filter = np.zeros((m,n) , np.float32)
    
    kernel_sum = 0
    for y in range(-m_half, m_half):
        for x in range(-n_half, n_half):
            normal = 1/(2*np.pi*sigma**2)  
            exp_term = np.exp(-(x**2 + y**2)/(2*(sigma**2)))
            gaussian_filter[y+m_half, x+n_half] = normal * exp_term
            kernel_sum = kernel_sum + (normal*exp_term)
    return gaussian_filter/kernel_sum
            
#defining convolution function
def convolution(image : np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if len(image.shape)==3:
        m_i, n_i, c_i = image.shape
        
        #if the image is grayscale, shape will not have 3 dimensions
    elif len(image.shape)==2:
            image = image[..., np.newaxis]
            m_i , n_i, c_i = image.shape
    else:
        raise Exception("Shape of image not supported")
    
    m_k, n_k = kernel.shape
    
    y_strides = m_i - m_k + 1 
    x_strides = n_i - n_k + 1
    
    img = image.copy()
    output_shape = (y_strides, x_strides, c_i) #size of output image
    output = np.zeros(output_shape, dtype = np.float32) #initializing output to zero
    
    count = 0
    
    output_tmp = output.reshape(output_shape[0]*output_shape[1] , output_shape[2])
    
    for i in range(y_strides):
        for j in range(x_strides):
            for c in range(c_i):
                sub_matrix = img[i:i+m_k , j:j+n_k, c]
                output_tmp[count,c] = np.sum(sub_matrix*kernel)
                
            count+=1
    
    output = output_tmp.reshape(output_shape)
    output = output.astype(np.uint8)
    return output


    
    
    
    