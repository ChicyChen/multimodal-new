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
from model.model import CLIP_2MAlign

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from utils import set_seed, mkdir, setup_logger, load_config_file

from dataloader.dataset import CLIP_COCO_dataset
from dataloader.data_loaders import get_dataloader




def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])
transform = _transform(224)

# data = json.load(open('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/dataloader/test_num2048.json'))
# num_imgs = 2048
data = json.load(open('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/dataloader/test.json'))
num_imgs = 256
id2file = {item['id']: item['coco_url'] for item in data['images']}
id2caption = {item['image_id']: item['caption'] for item in data['annotations']}
file2caption = {id2file[id]: id2caption[id] for id in id2file}
filenames = [(filename, file2caption[filename]) for filename in file2caption]

tokenizer = SimpleTokenizer()

"""
config = load_config_file('dataloader/data_config_tiny.yaml')
train_dataset = CLIP_COCO_dataset(config, tokenizer)
config.train_batch_size = 128 
config.per_gpu_train_batch_size = 128
config.num_workers = 4
config.device = "cuda" if torch.cuda.is_available() else "cpu"
config.n_gpu = torch.cuda.device_count() # config.n_gpu 
set_seed(seed=11, n_gpu=config.n_gpu)
train_dataloader = get_dataloader(config, train_dataset, is_train=True)
"""

MODEL_CONFIG_PATH = '/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/utils/model_config.yaml'
model_config = load_config_file(MODEL_CONFIG_PATH)

save_folder = "/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/nw_train_2m_align2048_checkpoints_learn_5e-4_1e-1"
step_list = [1,5,10]

device = "cuda"

global logger
logger = setup_logger("CLIP_COCO_Extraction", save_folder, 0, filename = "evaluate.txt")

for step in step_list:
    logger.info(f"Extrating step {step}")

    model_params = dict(model_config.RN50)
    model_params['vision_layers'] = tuple(model_params['vision_layers'])
    model_params['vision_patch_size'] = None
    model = CLIP_2MAlign(**model_params)

    model = model.to(device)

    if step > 0:
        checkpoint_path = os.path.join(save_folder,f'checkpoint_{step}.pt')
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        model.set_W(checkpoint['model_W'])

    model.eval()
    # model.train()

   
    """
    with torch.no_grad():
        image_feature_list = []
        text_feature_list = []
        for step, batch in enumerate(train_dataloader):
            input_images, _ = batch
            input_images = input_images.to(torch.device(config.device))
            input_texts = torch.clone(input_images)
            image_features, text_features = model(input_images, input_texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_feature_list.append(image_features)
            text_feature_list.append(text_features)
        image_feature_list = torch.cat(image_feature_list, dim=0)
        text_feature_list = torch.cat(text_feature_list, dim=0)
        center_dist = torch.norm(torch.mean(image_feature_list,dim=0)-torch.mean(text_feature_list,dim=0))
        print("Center distance after reloading =", center_dist)
    """


    image_features, image_features_nm, image_norm, text_features,text_features_nm, text_norm, image_inputs, text_inputs = [], [], [], [], [], [], [], []
    for filename in tqdm(file2caption):
        
        caption = file2caption[filename]
        filename = f"/scratch/qingqu_root/qingqu1/shared_data/coco2017/{filename.replace('http://images.cocodataset.org/', '')}"
        # number classes: 80
        
        im = PIL.Image.open(filename)

        image_input = transform(im).unsqueeze(0).to(device)
        sot_token = tokenizer.encoder["<|startoftext|>"]
        eot_token = tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]
        result = torch.zeros(77, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        text_input = result.unsqueeze(0).to(device)
        # text_input = torch.clone(image_input)

        with torch.no_grad():

            image_feature = model.encode_image(image_input)
            image_features_nm.append(image_feature)
            image_norm.append(torch.norm(image_feature[0]))

            text_feature = model.encode_text(text_input)
            text_features_nm.append(text_feature)
            text_norm.append(torch.norm(text_feature[0]))
        
            image_inputs.append(image_input)
            text_inputs.append(text_input)
            image_features.append(image_feature)
            text_features.append(text_feature)

        if len(image_features) == num_imgs: break

    image_inputs = torch.cat(image_inputs, dim=0)
    text_inputs = torch.cat(text_inputs, dim=0)

    image_features = torch.cat(image_features, dim=0)
    text_features = torch.cat(text_features, dim=0)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    image_features_nm = torch.cat(image_features_nm, dim=0)
    text_features_nm = torch.cat(text_features_nm, dim=0)
    image_norm = torch.tensor(image_norm)
    text_norm = torch.tensor(text_norm)

    if step == 0:
        u,s,vh = torch.linalg.svd(text_features.T @ image_features)
        text_features = text_features @ u @ vh
        text_features_nm = text_features_nm @ u @ vh

    center_dist = torch.norm(torch.mean(image_features, dim=0) - torch.mean(text_features, dim=0))
    logger.info(f"Center distance: {center_dist}")

    mean_image_norm = torch.mean(image_norm)
    mean_text_norm = torch.mean(text_norm)
    logger.info(f"Mean image norm: {mean_image_norm}")
    logger.info(f"Mean text norm: {mean_text_norm}")

    inter = (image_features.float() @ text_features.float().T)
    mean_sim = torch.diagonal(inter).float().mean()
    logger.info(f"Mean cosine similarity: {mean_sim}")
    mean_acc = (inter.argmax(dim=-1).cpu() == torch.arange(num_imgs)).float().mean()
    logger.info(f"Mean accuracy: {mean_acc}")
    

    save_path = os.path.join(save_folder, f'{step}.npy')
    np.save(save_path, [image_features.cpu().float().numpy(), text_features.cpu().float().numpy()])

    save_path = os.path.join(save_folder, f'nm_{step}.npy')
    np.save(save_path, [image_features_nm.cpu().float().numpy(), text_features_nm.cpu().float().numpy()])
    