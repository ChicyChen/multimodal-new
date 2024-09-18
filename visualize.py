
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

def svd(X, n_components=2):
    # using SVD to compute eigenvectors and eigenvalues
    # M = np.mean(X, axis=0)
    # X = X - M
    U, S, Vt = np.linalg.svd(X)
    # print(S)
    return U[:, :n_components] * S[:n_components]

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


save_folder = "/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/nw_train_align_checkpoints_learn_5e-4_1e-1"
step_list = [10000,20000]
# step_list = [5000,10000,15000,20000]
# step_list = list(range(10,201,50))
# step_list = list(range(10,100,10))
# step_list = [1,100,1000,2000,3000,4000]
# step_list = list(range(1,10))


for step in step_list:
    input_path = os.path.join(save_folder, f'nm_{step}.npy')
    all_img_features, all_text_features = np.load(input_path)
    image_features = torch.from_numpy(all_img_features)
    text_features = torch.from_numpy(all_text_features)
    # features_2d = np.concatenate([all_img_features, all_text_features], 0)

    features_2d = svd(np.concatenate([all_img_features, all_text_features], 0), n_components=2)

    plt.figure(figsize=(7, 7))
    plt.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], c='red')
    plt.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], c='blue')
    # connect the dots
    for i in range(len(all_img_features)):
        plt.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], c='black', alpha=0.1)
    save_path = os.path.join(save_folder, f"2d_nm_{step}.png")
    plt.savefig(save_path) 
    plt.close()

    features_2d = svd(np.concatenate([all_img_features, all_text_features], 0), n_components=3)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    r = 1.0
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(r*x, r*y, r*z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    ax.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], features_2d[:-len(all_img_features), 2], c='red')
    ax.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], features_2d[-len(all_img_features):, 2], c='blue')
    for i in range(len(all_img_features)):
        ax.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], [features_2d[i, 2], features_2d[len(all_img_features)+i, 2]], c='black', alpha=0.1)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    set_axes_equal(ax)
    save_path = os.path.join(save_folder, f"3d_nm_{step}.png")
    plt.savefig(save_path) 
    plt.close()


for step in step_list:
    input_path = os.path.join(save_folder, f'{step}.npy')
    all_img_features, all_text_features = np.load(input_path)
    image_features = torch.from_numpy(all_img_features)
    text_features = torch.from_numpy(all_text_features)
    # features_2d = np.concatenate([all_img_features, all_text_features], 0)

    features_2d = svd(np.concatenate([all_img_features, all_text_features], 0), n_components=2)

    plt.figure(figsize=(7, 7))
    plt.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], c='red')
    plt.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], c='blue')
    # connect the dots
    for i in range(len(all_img_features)):
        plt.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], c='black', alpha=0.1)
    save_path = os.path.join(save_folder, f"2d_{step}.png")
    plt.savefig(save_path) 
    plt.close()

    features_2d = svd(np.concatenate([all_img_features, all_text_features], 0), n_components=3)

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    r = 1.0
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(r*x, r*y, r*z, cmap=plt.cm.YlGnBu_r, alpha=0.2)
    ax.scatter(features_2d[:-len(all_img_features), 0], features_2d[:-len(all_img_features), 1], features_2d[:-len(all_img_features), 2], c='red')
    ax.scatter(features_2d[-len(all_img_features):, 0], features_2d[-len(all_img_features):, 1], features_2d[-len(all_img_features):, 2], c='blue')
    for i in range(len(all_img_features)):
        ax.plot([features_2d[i, 0], features_2d[len(all_img_features)+i, 0]], [features_2d[i, 1], features_2d[len(all_img_features)+i, 1]], [features_2d[i, 2], features_2d[len(all_img_features)+i, 2]], c='black', alpha=0.1)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    set_axes_equal(ax)
    save_path = os.path.join(save_folder, f"3d_{step}.png")
    plt.savefig(save_path) 
    plt.close()