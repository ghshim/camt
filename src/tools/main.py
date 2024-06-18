import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from src.data.dataset import *
from src.tools.train import *
from src.tools.test import *
from src.utils.utils import make_folder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gahyeon/Desktop/data/camt/', help='the directory of data')
    parser.add_argument('--output_dir', type=str, default='./result', help='the directory of data')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    debug_path = './debug'
    debug_path = make_folder(debug_path, 'run')
    make_folder(debug_path, 'test')
    make_folder(debug_path, 'train')
    os.makedirs(os.path.join(debug_path, 'train', 'prediction'))
    os.makedirs(os.path.join(debug_path, 'test', 'prediction'))

    train_dataloader, val_dataloader, test_dataloader = get_loader(args.data_dir, device=device)
    
    train_loss_list, validation_loss_list, model = train(train_dataloader, val_dataloader, debug_path=debug_path, device=device)
    print("Train loss list:", train_loss_list)
    print("Validation loss list:", validation_loss_list)

    plt.plot(train_loss_list, marker='o', linestyle='-')
    plt.title('Train Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(debug_path, 'train', 'train_loss.png'))

    test(model, test_dataloader, debug_path=debug_path, device=device)