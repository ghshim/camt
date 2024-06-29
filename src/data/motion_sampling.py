import numpy as np
import os
import pickle
import json
import smplx
import torch
import argparse
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.utils.constants import JOINT_NAMES, JOINT_HIERARHCY, NUM_JOINTS, SKELETON
from src.utils.data_utils import *


def extract_frame_number(frame_name):
    '''
    Extract frame number from PROX format frame name

    Args:
        frame_name (str): frame file name (e.g. s001_frame_00003__00.00.00.063.jpg)

    Returns:
        rame_number (int): int type frame number
    '''
    frame_number_str = frame_name.split('_')[2]
    return int(frame_number_str)


def split_continuous_motion(frame_list, interaction_list, calibrate_value=15):
    '''
    Split continuous motion in the given frame and interaction list

    Args:
        frame_list (list):
        interaction_list (list):
        calibrate_value (int): 
    
    Returns:
        continuous_motions (list):
    '''
    assert len(frame_list) == len(interaction_list)

    frame_list.sort(key=extract_frame_number)  # sort by frame number

    continuous_motions = []
    current_group = [(frame_list[0], interaction_list[0])]
    current_label = interaction_list[0]

    for i in range(1, len(frame_list)):
        frame_number = extract_frame_number(frame_list[i])
        prev_frame_number = extract_frame_number(current_group[-1][0])
        label = interaction_list[i]

        # if the current frame and previous frame numbers are consecutive and the labels are the same
        if frame_number <= prev_frame_number + calibrate_value and label == current_label:
            current_group.append((frame_list[i], label))
        else:
            continuous_motions.append(current_group)
            current_group = [(frame_list[i], label)]
            current_label = label

    continuous_motions.append(current_group)  
    
    return continuous_motions


def load_prox(data, frame_list, model_folder, prox_dir, gender=None, num_pca_comps=6, xform=True):
    '''
    Load all data in the data directory
    '''
    fitting_dir = os.path.join(prox_dir, "fittings", data)
    recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    fitting_dir = os.path.join(fitting_dir, 'results')
    scene_name = recording_name.split("_")[0]
    cam2world_dir = os.path.join(prox_dir, 'cam2world')
    # scene_dir = os.path.join(prox_dir, 'scenes')
    # recording_dir = os.path.join(prox_dir, 'recordings', recording_name)
    # color_dir = os.path.join(recording_dir, 'Color')

    female_subjects_ids = [162, 3452, 159, 3403]
    subject_id = int(recording_name.split('_')[1])
    if subject_id in female_subjects_ids:
        gender = 'female'
    else:
        gender = 'male'

    body_model_dict = {
        'male': smplx.create(os.path.join(model_folder,'SMPLX_MALE.npz'), model_type='smplx',
                             gender='male', ext='npz', num_betas = 10,
                             num_pca_comps=num_pca_comps),
        'female': smplx.create(os.path.join(model_folder,'SMPLX_FEMALE.npz'), model_type='smplx',
                               gender='female', ext='npz', num_betas = 10,
                               num_pca_comps=num_pca_comps),
        'neutral': smplx.create(os.path.join(model_folder,'SMPLX_NEUTRAL.npz'), model_type='smplx',
                               gender='neutral', ext='npz', num_betas = 10,
                               num_pca_comps=num_pca_comps)
    }
    # print(smpl_params.keys())
    if gender is not None:
        body_model = body_model_dict[gender]
    else:
        body_model = body_model_dict['neutral']

    if xform:
        with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
            trans = np.array(json.load(f))

    '''get all smpl pkl files in the data directory'''
    poses = {}
    for frame_name in sorted(frame_list):
        # print('viz frame {}'.format(img_name))
        # print(os.path.join(fitting_dir, img_name, '000.pkl'))
        with open(os.path.join(fitting_dir, frame_name, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        
        torch_param = {}
        
        param['left_hand_pose'] = param['left_hand_pose'][:, :num_pca_comps]
        param['right_hand_pose'] = param['right_hand_pose'][:, :num_pca_comps]
        
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation']:
                continue
            else:
                torch_param[key] = torch.tensor(param[key])

        output = body_model(return_verts=True, **torch_param)
        joints = output.joints.detach().cpu().numpy().squeeze(0) 
        transl = output.transl.detach().cpu().numpy().squeeze(0)
        
        # if joints has NaN value, continue
        if np.all(np.isnan(joints)):
            poses[frame_name] = None
            print(f"This [{frame_name}] is None")
            continue
        
        if xform:
            # print(joints)
            # get 3D joint positiions from smplx model output and apply cam2world transformation
            joints = c2w_transform_joints(trans, joints[:NUM_JOINTS, :])
            
            # print(joints)
            # get translation and apply cam2world transformation
            transl = c2w_transform(trans, transl)
            

        pose = {'joints':joints, 'transl':transl}
        poses[frame_name] = pose
    
    return poses


def generate_random_clip(duration, max_time):
    random.seed(1000) # fix random seed
    start_time = round(random.uniform(0, max(duration - 3, 0)))
    max_clip_duration = min(max_time, duration - start_time)
    end_time = start_time + round(random.uniform(2, max_clip_duration))
    return start_time, end_time


def sample_frame(data, sampling_time=3):
    duration = len(data) / 30 
    num_samples = duration // sampling_time
    if num_samples < 1:
        return [data]
    
    samples = []
    for i in range(int(num_samples)):
        start_time, end_time = generate_random_clip(duration, sampling_time)
        start_frame_idx, end_frame_idx = start_time * 30, end_time * 30
        samples.append(data[start_frame_idx:end_frame_idx+1])
    return samples


def save_motions(samples, filename):
    print("save motions...")
    data = [] # contain all samples from PROX
    for data_name, motions in samples.items():
        print("data name:", data_name)
        data_dict = {"name": data_name, "motions": []}
    
        for sample in motions:
            motion_dict = {} # contain sample motion in {data_name}
            poses = sample['motion'] # motion['motion'] contain poses ({frame_name:{'joints':joints, 'transl":transl}})
            label = sample['label']
            
            for frame_name, pose in poses.items():
                if pose is None:
                    continue
                frame_dict = {}
                frame_dict['joints'] = pose['joints']
                frame_dict['transl'] = pose['transl']
                motion_dict[frame_name] = frame_dict
            motion_dict['label'] = label
            # print("motion_dict length", len(motion_dict))
            
            data_dict['motions'].append(motion_dict)

        data.append(data_dict)

    with open(filename, 'wb') as output:
        pickle.dump(data, output)

    print(f"Samples are saved at {filename}.")


def sample(args):
    prox_dir = args.prox_dir
    label_dir = args.label_dir
    smplx_model_dir = args.smplx_model_dir
    output_dir = args.output_dir
    sampling = args.sampling
    sampling_time = args.sampling_time
    xform = args.xform
    saving = args.saving

    if sampling:
        print("Sample PROX")
        print("Sampling time:", sampling_time)
        print()
    
    os.makedirs(output_dir, exist_ok=True)

    label_dir = os.path.join(prox_dir, "interaction_labels_1")
    # fitting_dir = os.path.join(prox_dir, "fittings")
    # cam2world_dir = os.path.join(prox_dir, "cam2world")
    
    data_list = os.listdir(label_dir)
    data_dirs = [data for data in data_list]
    data_list = [os.path.splitext(data)[0] for data in data_dirs] # get data name stored with .json files
    
    count = {'0':0, '1':0, '2':0}
    samples = {}

    # data_list: all the data in PROX
    for data_name in data_list:
        print(f"Process {data_name}...")
        json_path = os.path.join(label_dir, data_name, data_name+'.json')
        # basename = data_name.split('_')[0]
        # # cam2world_path = os.path.join(cam2world_dir, basename+'.json')
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        frame_list = json_data['frame_name']
        interaction_list = json_data['interaction_sentences']

        # get pkl files of the frame with 1 interaction labels
        # fitting_path = os.path.join(fitting_dir, data)
        
        # get continuous frames list
        continuous_motions = split_continuous_motion(frame_list, interaction_list) 
        motions = []
        for each_m in continuous_motions:
            if len(each_m) < 10: continue

            frame_list = [m[0] for m in each_m] # frames in motion sample
            label = each_m[0][1] # interaction label of motion sample

            if sampling:
                sampled_frame_list = sample_frame(frame_list, sampling_time=sampling_time) # sampling 
                for sample in sampled_frame_list:
                    motion = load_prox(data_name, sample, smplx_model_dir, prox_dir, xform=xform)
                    if motion is not None:
                        motions.append({'motion':motion, 'label':label})
                    else:
                        print("motion is none")
                    
                    # only for testing
                    duration = len(sample) / 30
                    if duration < 1: count['0'] += 1
                    elif duration < 2: count['1'] += 1
                    elif duration < 3: count['2'] += 1
                    
            else:
                motion = load_prox(data_name, frame_list, smplx_model_dir, prox_dir, xform=xform)
                if motion is not None:
                    motions.append({'motion':motion, 'label':label})
            
        samples[data_name] = motions
    
    if saving:
        os.makedirs(output_dir, exist_ok=True)
        save_motions(samples, os.path.join(output_dir, 'prox.pkl'))

    print(count)
    print("sample count:", sum(count.values()))

   
if __name__ == '__main__':
    '''
    python ./src/data/motion_sampling.py --sampling True --sampling_time 2 --xform True --saving True
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--prox_dir', type=str, help='the directory of prox dataset', default='/home/gahyeon/Desktop/data/prox')
    parser.add_argument('--output_dir', type=str, default='/home/gahyeon/Desktop/data/camt/', help='outptu dir to save motion')
    parser.add_argument('--label_dir', type=str, help='the directory of prox dataset', default='/home/gahyeon/Desktop/data/prox/interaction_labels_1')
    parser.add_argument('--smplx_model_dir', type=str, help='The directory of smplx body mdoel', default="/home/gahyeon/Desktop/models/body_models/smplx")
    parser.add_argument('--sampling', default=False)
    parser.add_argument('--sampling_time', type=int, default=2)
    parser.add_argument('--xform', default=True)
    parser.add_argument('--saving', default=False)

    args = parser.parse_args()

    sample(args)

    with open(os.path.join(args.output_dir, "prox.pkl"), 'rb') as input_file:
        data = pickle.load(input_file)

    count = 0
    for data_dict in data:
        data_name = data_dict['name']
        motion_list = data_dict['motions']

        print(data_name)
        
        for motion_dict in motion_list:
            # print(motion_dict.keys())
            label = motion_dict.pop('label')
            print(label)
            # sentence_embedding = motion_dict.pop('sentence_embedding')
            
            # forward_dir = motion_dict.pop('forward_direction')
            # global_vel = motion_dict.pop('global_velocity')
            # rot_vel = motion_dict.pop('rotational_velocity')
            frame_list = list(motion_dict.keys())
            # print(len(frame_lists))
            # count += 1
            # continue
            joints = []
            transl = []
            for frame in frame_list:
                joints.append(motion_dict[frame]['joints'])
                transl.append(motion_dict[frame]['transl'])
                
            print(np.array(joints).shape)
            print(np.array(transl).shape)
            count += 1
            # for joint in test_joints:
            #     # test_joints.append(joint.get_joint_value(absolute=True))
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.set_xlim([-5, 5])
            #     ax.set_ylim([-5, 5])
            #     ax.set_zlim([-5, 5])
            #     human_body = joint #+ transl
            #     ax.scatter(human_body[:, 0], human_body[:, 1], human_body[:, 2], c='r')
            #     for joint_start, joint_end in SKELETON:
            #         ax.plot([human_body[joint_start, 0], human_body[joint_end, 0]],
            #                 [human_body[joint_start, 1], human_body[joint_end, 1]],
            #                 [human_body[joint_start, 2], human_body[joint_end, 2]], 'k-')
            #     plt.show()
            
            # print(motion['sentence_embedding'])
        print()
    print(count)
            