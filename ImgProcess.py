import torch
import torchvision
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
import pandas as pd
from tqdm.auto import tqdm
import pytorch_lightning as pl
from PIL import Image
import random
import os
import csv

def tv_norm(input, tv_beta):
    '''
    计算图片的梯度
    '''
    img = input[0, 0, :]
    ##上下两行的变化值，加了个beta的权重
    row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(tv_beta))
    return row_grad + col_grad


def preprocess_image(img):
    '''
    图片预处理，返回可以用于模型的状态
    '''
    
    ## 对于每个channel的means和stds
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1] ## BGR2RGB
    
    ## 正规化
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1))) ## WHC2CWH

    preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()    

    preprocessed_img_tensor.unsqueeze_(0) ## 增加维度
    return Variable(preprocessed_img_tensor, requires_grad = False)


def numpy_to_torch(img, requires_grad = True):
    '''
    转换为torch
    '''
    
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    
    output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad = requires_grad)
    return v


def loadImg(imgPath):
    # file_path = "../diffusion/"
    # img_path = "test_rocket.jpg"
    # refer_img_path = "314328828_2b52ae145e_120_55852171@N00.jpg"

    img = cv2.imread(imgPath, 1) ## BRG
    if img is None :
        return False, False
    intImg = cv2.resize(img, (256, 256))
    floatImg = np.float32(intImg) / 255 #（0，255）的INT值转为（0，1）的Float值
    
    return intImg, floatImg

def blurImg(intImg, floatImg):
    '''
    生成模糊图片：高斯， 中值，混合
    '''
    gussianBlur = cv2.GaussianBlur(floatImg, (11, 11), 5)
    medianBlur = np.float32(cv2.medianBlur(intImg, 11))/255
    mixBlur = (gussianBlur + medianBlur) / 2
    
    return gussianBlur, medianBlur, mixBlur

def medianBlurImg(intImg) :
    '''
    生成中值模糊图片
    '''
    medianBlur = np.float32(cv2.medianBlur(intImg, 11))/255
    
    return medianBlur

