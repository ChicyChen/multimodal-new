import torch
import torch.nn.functional as F
import numpy as np
import os
from omegaconf import OmegaConf

from dataloader.dataset import CLIP_COCO_dataset
from dataloader.data_loaders import get_dataloader

# # from model.model import CLIP
import clip
# from clip.model import CLIP
from model.model import CLIP

from utils.simple_tokenizer import SimpleTokenizer
from utils.custom_schedulers import get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils import set_seed, mkdir, setup_logger, load_config_file

from torch.optim import Adam, AdamW # both are same but AdamW has a default weight decay

import argparse




def train(config, train_dataset, model):
    '''
    Trains the model.
    '''
    
    config.train_batch_size = config.per_gpu_train_batch_size * max(1, config.n_gpu)    
    train_dataloader = get_dataloader(config, train_dataset, is_train=True)

    # total training iterations
    t_total = len(train_dataloader) // config.gradient_accumulation_steps \
                * config.num_train_epochs
    
    optimizer = AdamW(model.parameters(), lr=config.optimizer.params.lr, eps=config.optimizer.params.eps, weight_decay=config.optimizer.params.weight_decay)

    # Warmup iterations = 20% of total iterations
    # num_warmup_steps = int(0.20 * t_total)
    num_warmup_steps = 0
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps= num_warmup_steps, num_training_steps= t_total)
    scheduler = None

    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    model = model.to(torch.device(config.device))
    model.train()

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", config.num_train_epochs)
    logger.info("  Number of GPUs = %d", config.n_gpu)

    logger.info("  Batch size per GPU = %d", config.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, & accumulation) = %d",
                   config.train_batch_size * config.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if scheduler:
        logger.info("  warmup steps = %d", num_warmup_steps)


    global_step, global_loss, global_acc = 0, 0.0, 0.0
    model.zero_grad()

    for epoch in range(int(config.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            input_images, input_texts = batch

            input_images = input_images.to(torch.device(config.device))
            input_texts = input_texts.to(torch.device(config.device))
            
            image_features, text_features = model(input_images, input_texts)

            image_pos = torch.zeros_like(image_features)
            image_pos[::,-1] = image_pos[::,-1] + 1
            text_pos = torch.zeros_like(text_features)
            text_pos[::,-1] = text_pos[::,-1] - 1

            # loss
            image_loss = F.mse_loss(image_features, image_pos)
            text_loss  = F.mse_loss(text_features, text_pos)

            loss = (image_loss + text_loss) / 2

            if config.n_gpu > 1: 
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

            loss.backward()

            global_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                global_step += 1
                optimizer.step() # PYTORCH 1.x : call optimizer.step() first then scheduler.step()
                

                if scheduler:
                    scheduler.step() 
                    
                model.zero_grad()

                if global_step % config.logging_steps == 0:
                    logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, loss: {:.4f} ({:.4f})".format(epoch, global_step, 
                        optimizer.param_groups[0]["lr"], loss.item(), global_loss / global_step)
                    )

                if (config.save_steps > 0 and global_step % config.save_steps == 0) or \
                        global_step == t_total:
                    # saving checkpoint
                    save_checkpoint(config, epoch, global_step, model, optimizer) 
                    

    return global_step, global_loss / global_step


def save_checkpoint(config, epoch, global_step, model, optimizer):
    '''
    Checkpointing. Saves model and optimizer state_dict() and current epoch and global training steps.
    '''
    checkpoint_path = os.path.join(config.saved_checkpoints, f'checkpoint_{global_step}.pt')
    save_num = 0
    while (save_num < 10):
        try:

            if config.n_gpu > 1:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
            else:
                torch.save({
                    'epoch' : epoch,
                    'global_step' : global_step,
                    'model_state_dict' : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)

            logger.info("Save checkpoint to {}".format(checkpoint_path))
            break
        except:
            save_num += 1
    if save_num == 10:
        logger.info("Failed to save checkpoint after 10 trails.")
    return

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--DATA_CONFIG_PATH", default='dataloader/data_config_num2.yaml', type=str, required=False)
    parser.add_argument("--TRAINER_CONFIG_PATH", default='utils/train_config_initial.yaml', type=str, required=False, help="path of directory containing COCO training images")
    parser.add_argument("--MODEL_CONFIG_PATH", default='utils/model_config.yaml', type=str, required=False, help="path of directory containing COCO training images")
    parser.add_argument("--train_annotation_file", default=None, type=str, required=False, help="path of COCO annotation file")
    parser.add_argument("--saved_checkpoints", default=None, type=str, required=False)
    parser.add_argument("--logs", default=None, type=str, required=False)
    parser.add_argument("--num_train_epochs", default=None, type=int, required=False)
    parser.add_argument("--lr", default=None, type=float, required=False)

    



    args = parser.parse_args()

    # DATA_CONFIG_PATH = 'dataloader/data_config.yaml'
    # # DATA_CONFIG_PATH = 'dataloader/data_config_tiny.yaml'
    # TRAINER_CONFIG_PATH = 'utils/train_config.yaml'
    # MODEL_CONFIG_PATH = 'utils/model_config.yaml'

    data_config = load_config_file(args.DATA_CONFIG_PATH)
    train_config = load_config_file(args.TRAINER_CONFIG_PATH)
    model_config = load_config_file(args.MODEL_CONFIG_PATH)

    config = OmegaConf.merge(train_config, data_config)

    # config = OmegaConf.merge(OmegaConf.create(vars(args)), config)  
    # merging cli arguments, if data path given in cli args use those
    if args.train_annotation_file : 
        config.train_annotation_file = args.train_annotation_file
    if args.saved_checkpoints : 
        config.saved_checkpoints = args.saved_checkpoints
    if args.logs : 
        config.logs = args.logs
    if args.num_train_epochs : 
        config.num_train_epochs = args.num_train_epochs
    if args.lr :
        config.optimizer.params.lr = args.lr

        

    global logger
    # creating directories for saving checkpoints and logs
    mkdir(path=config.saved_checkpoints)
    mkdir(path=config.logs)

    logger = setup_logger("CLIP_COCO_TRAIN", config.logs, 0, filename = "training_logs.txt")

    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    config.n_gpu = torch.cuda.device_count() # config.n_gpu 
    set_seed(seed=11, n_gpu=config.n_gpu)

    # getting text tokenizer
    tokenizer = SimpleTokenizer()
    
    # creating RN50 CLIP model
    model_params = dict(model_config.RN50_3)
    model_params['vision_layers'] = tuple(model_params['vision_layers'])
    model_params['vision_patch_size'] = None
    model = CLIP(**model_params)

    logger.info(f"Training/evaluation parameters {train_config}")

    # getting dataset for training
    train_dataset = CLIP_COCO_dataset(config, tokenizer)

    # TODO: using clip preprocess
    # _, preprocess = clip.load('RN50', "cpu")
    # train_dataset = CLIP_COCO_dataset(config, tokenizer, clip_token = True, clip_prepro = preprocess)


    # Now training
    global_step, avg_loss = train(config, train_dataset, model)
    
    logger.info("Training done: total_step = %s, avg loss = %s", global_step, avg_loss)
    

if __name__ == "__main__":
    main()