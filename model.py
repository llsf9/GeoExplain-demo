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

from classification.train_base import MultiPartitioningClassifier
from ImgProcess import * 
from utils import * 


def gps_inference(img, model):
    '''
    将模型切换到eval模式并返回预测值
    '''
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
        
    img_reshape = torch.reshape(img, (1, img.size(0), img.size(1), img.size(2), img.size(3)))
    data = [img_reshape, {"img_id":"0", "img_path": "None"}]

    if torch.cuda.is_available():
        data[0] = data[0].cuda()
    img_paths, pred_classes, pred_latitudes, pred_longitudes, hierarchy_preds = model.inference(data)
    
    return pred_latitudes['hierarchy'], pred_longitudes['hierarchy'], hierarchy_preds

def loadModel(checkpoint, hparams):
    '''
    加载已训练的模型
    '''
    print("Load model from ", checkpoint)
    model = MultiPartitioningClassifier.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        hparams_file=str(hparams),
        map_location=None,
    )
    return model

def getGV(csvPath, imgPath, row_lat=2, row_lng=3):
    '''
    获取某个图片的真实值
    ''' 
    with open(csvPath) as f :
        f_csv = csv.reader(f)
        headers = next(f_csv)
        for row in f_csv :
            if imgPath in row :
                ylat = float(row[row_lat]) 
                ylng = float(row[row_lng])
                return ylat, ylng
        print("!!!no such image!!!")
            
            
            
def initMask() :
    '''
    初始化MASK
    '''
    mask = np.ones((28, 28), dtype = np.float32)
    mask = numpy_to_torch(mask)
    
    return mask

def upSample(upsample, mask) :
    upsampled_mask = upsample(mask)
    upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))
    
    return upsampled_mask

def generateNoise() :
    cv2.setRNGSeed(1024)
    noise = np.zeros((256, 256, 3), dtype = np.float32)
    cv2.randn(noise, 0, 0.2)
    noise = numpy_to_torch(noise)
    
    return noise

def calLoss(perturbated_input, mask, batchId, model, ground_value, y_lat, y_lng, tv_beta, l1_coeff, tv_coeff, printLog = False, debug = False) :
    ##计算loss
    pre_lat, pre_lng, pre_prob = gps_inference(perturbated_input, model)

    pre_prob = torch.nn.Softmax()(pre_prob)
    pre_value_prob = pre_prob[0,ground_value]
    
    ## 归一化
    prob_sta = (pre_value_prob - torch.mean(pre_prob).item()) /  torch.sqrt(torch.var(pre_prob)).item() /4000

    err = get_distance(pre_lat.item(),pre_lng.item(), y_lat, y_lng)
    
    sizeloss = l1_coeff*torch.mean(torch.abs(1 - mask)) ## 计算blur面积，量级 [0~1]*l1
    normloss = tv_coeff*tv_norm(mask, tv_beta) 
    classifyloss = prob_sta

    sumloss = sizeloss + normloss + classifyloss
    
    if (printLog and batchId % 100 == 0) or debug:
        print(f'sumloss: {sumloss} sizeloss: {sizeloss} normloss: {normloss} classifyloss: {classifyloss} err: {err} classifyloss: {classifyloss}')
    
    # if debug :
    #     print(f'sumloss: {sumloss} sizeloss: {sizeloss} normloss: {normloss} classifyloss: {classifyloss} err: {err} classifyloss: {classifyloss}')
    #     return sumloss, pre_prob, pre_prob

    return sumloss

def calLoss2(perturbated_input, mask, batchId, model, ground_value, y_lat, y_lng, tv_beta, l1_coeff, tv_coeff, printLog = False, debug = False) :
    '''
    Calculation the loss function with no Softmax
    '''
    ##计算loss
    pre_lat, pre_lng, pre_prob = gps_inference(perturbated_input, model)

    # pre_prob = torch.nn.Softmax()(pre_prob)
    pre_value_prob = pre_prob[0,ground_value]
    
    ## 归一化
    prob_sta = (pre_value_prob - torch.mean(pre_prob).item()) /  torch.sqrt(torch.var(pre_prob)).item() /4000
    
    sizeloss = l1_coeff*torch.mean(torch.abs(1 - mask)) ## 计算blur面积，量级 [0~1]*l1
    normloss = tv_coeff*tv_norm(mask, tv_beta) 
    classifyloss = prob_sta

    sumloss = sizeloss + normloss + classifyloss
    
    if (printLog and batchId % 100 == 0) or debug:
        print(f'sumloss: {sumloss} sizeloss: {sizeloss} normloss: {normloss} classifyloss: {classifyloss} err: {err} classifyloss: {classifyloss}')
    
    # if debug :
    #     print(f'sumloss: {sumloss} sizeloss: {sizeloss} normloss: {normloss} classifyloss: {classifyloss} err: {err} classifyloss: {classifyloss}')
    #     return sumloss, pre_prob, pre_prob

    return sumloss,  pre_lat, pre_lng