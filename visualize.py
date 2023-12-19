import copy
import torch
import torchvision
import cv2
import numpy as np
import pandas as pd
from PIL import Image 
import matplotlib.pyplot as plt

def resultVis(resPath, mask, mixBlur, intImg, floatImg, upsample, img_path, epoch,saveFig = True) :
    ##可视化
    vis_mask = upsample(mask)
    ## 把多余的次元删掉，并转换到img的次元顺序
    vis_mask = vis_mask.cpu().data.numpy()[0]
    vis_mask = np.transpose(vis_mask, (1, 2, 0))
    vis_mask = (vis_mask - np.min(vis_mask)) / np.max(vis_mask)

    failFlag = False
    if np.all(vis_mask<0.6):
        vis_mask = np.full((256,256,1), 0.)
        failFlag = True
    else:
        vis_mask = 1 - vis_mask
    
    heatmap = cv2.applyColorMap(np.uint8(255*vis_mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    cam = 1.0*heatmap + np.float32(intImg)/255
    cam = cam / np.max(cam)

    vis_result = np.multiply(1 - vis_mask, floatImg) + np.multiply(vis_mask, mixBlur)
    
    sharpMask = visSharpMask(vis_mask)
    
    if saveFig :
        # 对比图
        fig = plt.figure(figsize = (15,15))
        plt.subplot(2, 2, 1)
        plt.imshow(cv2.cvtColor(np.uint8(255*floatImg), cv2.COLOR_BGR2RGB)) 
        plt.subplot(2, 2, 2)
        plt.imshow(cv2.cvtColor(np.uint8(255*vis_result), cv2.COLOR_BGR2RGB))
        plt.subplot(2, 2, 3)
        plt.imshow(cv2.cvtColor(np.uint8(255*cam), cv2.COLOR_BGR2RGB))
        plt.subplot(2, 2, 4)
        plt.imshow(cv2.cvtColor(np.uint8(255*sharpMask), cv2.COLOR_BGR2RGB)) 
        fig.savefig(resPath+"/comparison/"+img_path)
        plt.close() 

        # 分别的图片
        cv2.imwrite(resPath+"/vis_result/"+img_path, np.uint8(255*vis_result))
        cv2.imwrite(resPath+"/sharpMask/"+img_path, np.uint8(255*sharpMask))
        cv2.imwrite(resPath+"/heatmap/"+img_path, np.uint8(255*heatmap))
       
    return vis_result, floatImg, sharpMask, heatmap, cam, failFlag

def visSharpMask(vis_mask) :
    sharpMask = copy.deepcopy(vis_mask)
    for i, h in enumerate(vis_mask) :
        for j, cell in enumerate(h) :
            if cell < 0.6 :
                sharpMask[i][j] = 0
            else :
                sharpMask[i][j] = 1
    return sharpMask 

def listVis(max_iterations, lossList, img_name, ylable, saveFig = True) :
    plt.figure()
    plt.plot(range(max_iterations), lossList)
    plt.xlabel("epoch")
    plt.ylabel(ylable)
    if saveFig : 
        plt.savefig("{}/{}.jpg".format(resPath, img_name))

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()
    
                

    