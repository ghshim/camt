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
    train_dataloader, val_dataloader, test_dataloader = get_loader(args.data_dir)
    
    train_loss_list, validation_loss_list, model = train(train_dataloader, val_dataloader)
    print("Train loss list:", train_loss_list)
    print("Validation loss list:", validation_loss_list)

    # plt.plot(train_loss_list, marker='o', linestyle='-')

    output_path = make_folder(args.output_dir, 'output')

    # 그래프 제목과 축 라벨 설정
    plt.title('Train Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 그래프 표시
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'train_loss.png'))

    test_loss, pred_result, next_motion_description, frame_lists = test(model, nn.MSELoss(), test_dataloader, device)
    print("Test loss:", test_loss)
    filename = os.path.join(output_path, 'pred_result.pkl')
    with open(filename, 'wb') as output:
        pickle.dump({'loss':test_loss, 'prediction':pred_result, 'label':next_motion_description, 'frame_lists':frame_lists}, output)