import random
import torch
import os
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import datetime
import csv
# pycse v2.2.0, mistune v0.8.4
from pycse import nlinfit
from scipy import stats
from scipy.io import savemat
import cv2
import warnings
from torch import Tensor
from torch.nn import functional as F


def calculateMSCN(img):  # img_abs_path: str
    """
    calculate MSCN
    """
    # preprocess_image
    if isinstance(img, str):
        if os.path.exists(img):
            imdist = cv2.resize(cv2.imread(img, 0), (224, 224)).astype(np.float32)
        else:
            raise FileNotFoundError('The image is not found on your system.')
    else:
        raise ValueError('You can only pass image to the constructor.')
    # calculate MSCN
    mu = cv2.GaussianBlur(imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    mu_sq = mu * mu
    sigma = cv2.GaussianBlur(imdist * imdist, (7, 7), 7 / 6, borderType=cv2.BORDER_CONSTANT)
    sigma = np.sqrt(abs((sigma - mu_sq)))
    structdis = (imdist - mu) / (sigma + 1)
    mscn = torch.from_numpy(structdis).unsqueeze(0)
    # print(mscn.shape)  # torch.Size([1, 224, 224])
    # print(mscn)
    return mscn


def calculateGrad(img):  # img_abs_path: str
    """
    calculate Grad
    """
    # preprocess_image
    if isinstance(img, str):
        if os.path.exists(img):
            imdist = cv2.resize(cv2.imread(img, 0), (224, 224)).astype(np.float32)
        else:
            raise FileNotFoundError('The image is not found on your system.')
    else:
        raise ValueError('Input img not is str.')
    scharrx = cv2.Scharr(imdist[:, :], cv2.CV_16U, 1, 0)
    scharry = cv2.Scharr(imdist[:, :], cv2.CV_16U, 0, 1)
    grad = np.sqrt(scharrx ** 2 + scharry ** 2)
    # print(grad.shape)  # (224, 224)
    grad = torch.from_numpy(np.array(grad)).unsqueeze(0)
    # print(grad.shape)  # torch.Size([1, 224, 224])
    # print(grad)
    return grad


def calculate_srocc_krocc_plcc_rmse(img_mos, img_pre):
    """
    Calculate: SROCC, KROCC, PLCC, RMSE
    """
    mos = np.array(img_mos)
    pre = np.array(img_pre)

    # Definition Logistic Function
    def logistic_function(x, beta1, beta2, beta3, beta4, beta5):
         return beta1 * (0.5 - 1.0 / (1 + np.exp(beta2 * (x - beta3)))) + beta4 * x + beta5

    # Initialize Parameters
    beta0 = np.array([10, 0, np.mean(pre), 0.1, 0.1])
    # Fitting, Reference
    beta, _, _ = nlinfit(logistic_function, pre, mos, beta0, maxfev=99999999)
    # Mapping
    map = logistic_function(pre, *beta)

    # Calculate
    # SROCC (Spearman Rank Order Correlation Coefficient)
    srocc, _ = stats.spearmanr(pre, mos)
    # KROCC (Kendall Rank Order Correlation Coefficient)
    krocc, _ = stats.kendalltau(pre, mos)
    # PLCC (Pearson Linear Correlation Coefficient)
    plcc, _ = stats.pearsonr(map, mos)
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((np.square(map - mos))))
    # 保留4位小数
    srocc = round(abs(srocc), 4)
    krocc = round(abs(krocc), 4)
    plcc = round(abs(plcc), 4)
    rmse = round(rmse, 4)

    return srocc, krocc, plcc, rmse


if __name__ == '__main__':
    pass
    # calculateMSCN(img='./1.bmp')
    calculateGrad(img='./1.bmp')



