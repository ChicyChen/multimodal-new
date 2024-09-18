import os
import clip
import torch

import numpy as np
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from tqdm import tqdm

import sys
sys.path.append('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/utils')
sys.path.append('/scratch/qingqu_root/qingqu1/siyich/multimodal-gap')
from simple_tokenizer import SimpleTokenizer
from model.model import CLIP
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from utils import set_seed, mkdir, setup_logger, load_config_file

MODEL_CONFIG_PATH = '/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/utils/model_config.yaml'
model_config = load_config_file(MODEL_CONFIG_PATH)

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.4225, 0.4012, 0.3659), (0.2681, 0.2635, 0.2763)), # COCO mean, std
    ])

tokenizer = SimpleTokenizer()

# Load the model
device = "cuda" 

model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
# checkpoint_path = os.path.join(save_folder,f'checkpoint_{step}.pt')
checkpoint_path = '/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/train_checkpoints_temp7e-2_increase/checkpoint_10000.pt'
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


# Load the dataset
root = os.path.expanduser("~/.cache")
train = CIFAR100(root, download=True, train=True, transform= _transform(224))
test = CIFAR100(root, download=True, train=False, transform= _transform(224))


def get_features(dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            # need to check whether encoding the right shape
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# Calculate the image features
train_features, train_labels = get_features(train)
test_features, test_labels = get_features(test)

# Perform logistic regression
classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

# Evaluate using the logistic regression classifier
predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
print(f"Accuracy = {accuracy:.3f}")