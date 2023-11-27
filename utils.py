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
from math import radians, cos, sin, asin, sqrt
import os
import csv

def torch_fix_seed(seed=42): 
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    random.seed(seed)
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True # 再現性はあるが、速度が低下する可能性
     
def get_distance(lat1,lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r

def addJpgImgFile(dirPath, skipListPath):
    '''
    return : 包含jpg结尾的图片文件名的list
    '''
    img_list = os.listdir(dirPath)
    print(f"Original data num : {len(img_list)}")

    if skipListPath :
        with open(skipListPath, 'r') as f:
            skiplist = f.readlines()
        skiplist = [line.rstrip("\n") for line in skiplist]
    else: skiplist=[]

    newImgList = []
    for img_name in img_list : 
        if ".jpg" in img_name and img_name not in skiplist:
            newImgList.append(img_name)
    print(f"Processed data num : {len(newImgList)}")
    
    return newImgList 
        
