import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F

from src.models.MotionTransformer import MotionTransformer
from src.utils.data_utils import *
from src.data.dataset import *
from src.utils.utils import make_folder

debug_path = './debug'

def pad_motion(motion, max_seq_len=61):
    # print(f"Next motion sequence length: {next_motion.size(1)}")
    padding = (motion[:, -1, :].unsqueeze(0)).repeat(1, max_seq_len - motion.size(1), 1) 
    padded_seq = torch.cat([motion, padding], dim=1)
    # print("motion:", motion.shape)
    # print("padded motion:", padded_seq.shape)
    return padded_seq

# def train_loop(model, opt, loss_fn, dataloader):
    

#     return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[:, 0], batch[:, 1]
            X, y = X.to(device), y.to(device)  # GPU로 이동

            y_input = y[:, :-1]  # 마지막 토큰을 제외한 입력 시퀀스
            y_expected = y[:, 1:]  # 첫 번째 토큰을 제외한 실제 예측 대상 시퀀스

            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)  # 타겟 마스크 생성

            pred = model(X, y_input, tgt_mask)  # 모델 예측
            
            loss = loss_fn(pred.permute(0, 2, 1), y_expected)  # 손실 계산
            total_loss += loss.item()

    return total_loss / len(dataloader)


def fit(model, epochs, opt, loss_fn, train_dataloader, val_dataloader, debug_path=None, device=None):  
    debug_path = os.path.join(debug_path, 'prediction')
    train_loss_list, validation_loss_list = [], []

    print("Training and validating model")
    for epoch in tqdm(range(epochs), desc='Train'):
        # print("-"*25, f"Epoch {epoch + 1}","-"*25)
        model.train()
        total_loss = 0

        for batch_data in train_dataloader:
            '''
            initial_pose: (bs, 1, 72)
            object_boxes: (bs, num_objs, 3)
            next_motion_desc: (bs, 384)
            gt_motion: (bs, num_frames, 76)
            '''
            data_name, frame_list, label, initial_pose, gt_motion, motion_descr, object_box = batch_data

            # padded_motion = pad_motion(gt_motion)
            tgt_mask = get_tgt_mask(gt_motion[:,:-1,:]).to(device)
            # predict EoM token (EoM: torch.zeros((1, 76)))
            output = model(initial_pose, object_box, motion_descr, gt_motion[:,:-1,:], tgt_mask=tgt_mask) # output: (bs, seq_len-1, 76)
            loss = loss_fn(output, gt_motion[:,1:,:])  # loss calculate (0, 2, 1)
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()

        # save prediction result for debugging
        data = {'name': data_name, 'frame_list':frame_list, 'label':label,
                'object_box': object_box.detach().cpu().numpy(), 
                'prediction': output.detach().cpu().numpy(),
                'gt': gt_motion.detach().cpu().numpy()}
        
        with open(os.path.join(debug_path, f'{epoch}.pkl'), 'wb') as output:
            pickle.dump(data, output)

        torch.save(model.state_dict(), os.path.join(debug_path, "../last.pt"))

        # loss calculation
        train_loss = total_loss / len(train_dataloader)
        train_loss_list.append(train_loss)
        print(f"Training loss: {train_loss:.4f}")        
        
    return train_loss_list, None


def get_tgt_mask(tgt):
    '''
    Make (seq_len, seq_len) target mask
    '''
    # bs = tgt.size(0)
    # seq_len = tgt.size(1)
    # mask = torch.tril(torch.ones(seq_len, seq_len) == 1)
    # mask = mask.unsqueeze(0)
    # mask = mask.expand(bs,-1, -1)
    # mask = mask.float()
    # mask = mask.masked_fill(mask==0, float('-inf')) # convert zeros to -inf
    # mask = mask.masked_fill(mask==1, float(0.0))
    # print("tgt mask:", mask.shape)
    # print(mask)
    size = tgt.size(1)
    mask = torch.tril(torch.ones(size, size) == 1)
    mask = mask.float()
    mask = mask.masked_fill(mask==0, float('-inf')) # convert zeros to -inf
    mask = mask.masked_fill(mask==1, float(0.0))    # convert ones to 0
    return mask


def pad_motion(motion, max_seq_len=61):
    # print(f"Next motion sequence length: {next_motion.size(1)}")
    padding = (motion[:, -1, :].unsqueeze(0)).repeat(1, max_seq_len - motion.size(1), 1) 
    padded_seq = torch.cat([motion, padding], dim=1)
    # print(f"Padded next motion sequence length: {padded_seq.size(1)}")
    # print("next motion:", next_motion.shape)
    # print("padded_seq:", p)
    return padded_seq

def train(train_dataloader, val_dataloader, debug_path='./debug', device=None):
    '''
    initial_poses shape: torch.Size([1, 76])
    object_boxes shape: torch.Size([2, 3])
    next_motion_descriptions shape: torch.Size([384])
    next_motions shape: torch.Size([25, 76])
    '''
    debug_path = os.path.join(debug_path, 'train')

    learning_rate = 0.0001
    dropout_p = 0.1
    num_epochs = 100

    num_tokens = 76 # output dimension
    dim_model = 512
    num_heads = 8
    num_encoder_layers = 6
    num_decoder_layers = 6

    # save configs
    config = {
        "learning_rate": learning_rate,
        "dropout_p": dropout_p,
        "num_epochs": num_epochs,
        "num_tokens": num_tokens,
        "dim_model": dim_model,
        "num_heads": num_heads,
        "num_encoder_layers": num_encoder_layers,
        "num_decoder_layers": num_decoder_layers
    }

    with open(os.path.join(debug_path, "config.json"), 'w') as json_file:
        json.dump(config, json_file, indent=4)

    dim_pose = 72
    dim_object = 3
    dim_description = 384 
    dim_motion = 76

    # initialize MotionTransformer
    model = MotionTransformer(num_tokens=num_tokens, dim_model=dim_model, num_heads=num_heads, 
                              num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, 
                              dropout_p=dropout_p, device=device, dim_pose=dim_pose, dim_object=dim_object, 
                              dim_description=dim_description, dim_motion=dim_motion).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=1, eta_min=1e-6, last_epoch=-1)
    loss_fn = nn.MSELoss(reduction='sum')

    # train
    train_loss_list, validation_loss_list = fit(model, num_epochs, opt, loss_fn, train_dataloader, val_dataloader, \
                                                debug_path=debug_path, device=device)
    
    # save trained model
    torch.save(model.state_dict(), os.path.join(debug_path, "model.pt"))

    return train_loss_list, validation_loss_list, model
