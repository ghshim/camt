import os
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.MotionTransformer import MotionTransformer
from src.utils.data_utils import *
from src.data.dataset import *
from src.utils.utils import make_folder

debug_path = './debug'


def test(model, dataloader, max_len=62, debug_path='./debug', device=None):
    EoM = torch.zeros(1, 1, 76).to(device)
    debug_path = os.path.join(debug_path, 'test')
    
    model.eval()

    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(dataloader, desc='Test')):
            data_name, frame_list, label, initial_pose, gt_motion, motion_descr, object_box = batch_data
            
            bs = initial_pose.shape[0]
            
            # initial_pose + global velocity (zeros) + rotational velocity (zeros)
            curr_pose = torch.cat([initial_pose, torch.zeros((bs,1,3), device=device), torch.zeros((bs,1,1), device=device)], dim=2)

            # store predicted motion
            prediction = torch.zeros((bs, max_len, 76)).to(device)
            # predict next possible motion
            for i in range(max_len):
                prediction[:,i,:] = curr_pose # save current pose 
                output = model(initial_pose, object_box, motion_descr, curr_pose) # (bs, seq_len-1, 76)
                # If EoM is last pose
                # if torch.eq(output, curr_pose).sum() != 0:
                #     print("End of Motion")
                #     break
                # If EoM is zero
                if torch.eq(output, EoM).sum() != 0:
                    print("End of Motion")
                    break
                curr_pose = output

            print(prediction)
            
            data = {'name': data_name, 'frame_list':frame_list, 'label':label,
                    'object_box': object_box.detach().cpu().numpy(), 
                    'prediction': prediction.detach().cpu().numpy(),
                    'gt': gt_motion.detach().cpu().numpy()}
        
            with open(os.path.join(debug_path, f'prediction/{idx}.pkl'), 'wb') as output:
                pickle.dump(data, output)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/gahyeon/Desktop/data/camt/', help='the directory of data')
    parser.add_argument('--model_path', type=str, default='./debug/run/train/model.pt', help='the directory of data')
    args = parser.parse_args()

    debug_path = './debug'
    debug_path = make_folder(debug_path, 'run')
    make_folder(debug_path, 'test')
    os.makedirs(os.path.join(debug_path, 'test', 'prediction'))


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MotionTransformer(num_tokens=76).to(device)
    model.load_state_dict(torch.load(args.model_path))

    generator = torch.Generator()
    generator.manual_seed(0)    
    train_dataloader, val_dataloader, test_dataloader = get_loader(args.data_dir, device=device)
    
    test(model, test_dataloader, debug_path=debug_path, device=device)