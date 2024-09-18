
import sys
import os
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR100, MNIST
from torchvision import transforms
from tqdm import trange
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from pytorch_pretrained_vit import ViT
import json
import PIL
from tqdm import tqdm, trange
from collections import Counter
from transformers import BertTokenizer
import ruamel_yaml as yaml
from scipy.optimize import linear_sum_assignment
import math

import clip
sys.path.append('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/utils')
sys.path.append('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap')
from util import load_config_file
from simple_tokenizer import SimpleTokenizer
from model.model import CLIP

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from sklearn import svm

from utils import setup_logger


def svd(X, n_components=2):
    # using SVD to compute eigenvectors and eigenvalues
    # M = np.mean(X, axis=0)
    # X = X - M
    U, S, Vt = np.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]


save_folder = "/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/nw_train_img2text2048_checkpoints_learn_5e-4_1e-1"
step_list = [50000,100000,160000]
# step_list = [5000,10000,15000,20000]
# step_list = list(range(10,201,50))
# step_list = list(range(10,100,10))
# step_list = [1,100,1000,2000,3000,4000]
# step_list = list(range(1,10))

global logger
logger = setup_logger("CLIP_COCO_Extraction", save_folder, 0, filename = "svm.txt")


for step in step_list:
    logger.info(f"Extrating step {step}")
    input_path = os.path.join(save_folder, f'{step}.npy')
    all_img_features, all_text_features = np.load(input_path)

    # features_2d = svd(np.concatenate([all_img_features, all_text_features], 0), n_components=2)
    # all_img_features, all_text_features = features_2d[:len(all_img_features),:], features_2d[len(all_img_features):,:]

    image_features = torch.from_numpy(all_img_features)
    text_features = torch.from_numpy(all_text_features)
    features_train = np.concatenate([all_img_features[:192,:], all_text_features[:192,:]], 0)
    labels_train = np.concatenate([np.zeros(192), np.ones(192)], 0)
    features_test = np.concatenate([all_img_features[192:,:], all_text_features[192:,:]], 0)
    labels_test = np.concatenate([np.zeros(64), np.ones(64)], 0)

    clf = svm.SVC()
    clf.fit(features_train, labels_train)
    predicts_test = clf.predict(features_test)
    acc = np.sum(predicts_test == labels_test) / 128
    logger.info(f"SVM classification accuracy: {acc}")


    """
    image_features = torch.from_numpy(all_img_features)
    text_features = torch.from_numpy(all_text_features)
    features_train = np.concatenate([all_img_features[:192*8,:], all_text_features[:192*8,:]], 0)
    labels_train = np.concatenate([np.zeros(192*8), np.ones(192*8)], 0)
    features_test = np.concatenate([all_img_features[192*8:,:], all_text_features[192*8:,:]], 0)
    labels_test = np.concatenate([np.zeros(64*8), np.ones(64*8)], 0)

    clf = svm.SVC()
    clf.fit(features_train, labels_train)
    predicts_test = clf.predict(features_test)
    acc = np.sum(predicts_test == labels_test) / (128*8)
    logger.info(f"SVM classification accuracy: {acc}")
    """