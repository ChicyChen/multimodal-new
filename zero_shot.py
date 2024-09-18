import os
import clip
import torch
from torchvision.datasets import CIFAR100

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
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # default clip
    ])

tokenizer = SimpleTokenizer() # seems to be the same as clip

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"

model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
# checkpoint_path = os.path.join(save_folder,f'checkpoint_{step}.pt')
checkpoint_path = '/scratch/qingqu_root/qingqu1/siyich/multimodal-gap/train_checkpoints_temp7e-2/checkpoint_10000.pt'
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
preprocess = _transform(224)

_, preprocess = clip.load('RN50', device)


# Download the dataset
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)

num_data = len(cifar100)
# num_data = 10
top_1 = 0
top_5 = 0

for idx in range(num_data):

    # Prepare the inputs
    image, class_id = cifar100[idx]
    # print(cifar100.classes[class_id])
    image_input = preprocess(image).unsqueeze(0).to(device)

    # text_inputs = []
    # for c in cifar100.classes:
    #     sot_token = tokenizer.encoder["<|startoftext|>"]
    #     eot_token = tokenizer.encoder["<|endoftext|>"]
    #     tokens = [sot_token] + tokenizer.encode(f"a photo of a {c}") + [eot_token]
    #     result = torch.zeros(77, dtype=torch.long)
    #     result[:len(tokens)] = torch.tensor(tokens)
    #     text_input = result.unsqueeze(0).to(device)
    #     text_inputs.append(text_input)
    # text_inputs = torch.cat(text_inputs).to(device)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values_1, indices_1 = similarity[0].topk(1)
    values_5, indices_5 = similarity[0].topk(5)

    if class_id in indices_1:
        top_1 += 1
    if class_id in indices_5:
        top_5 += 1

    # # Print the result
    # print("\nTop predictions:\n")
    # for value, index in zip(values, indices):
    #     print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")


print("Top 1:", top_1/num_data)
print("Top 5:", top_5/num_data)
