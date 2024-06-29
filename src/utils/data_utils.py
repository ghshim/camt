import numpy as np
import os
import pickle
import torch
import json

from src.utils.constants import JOINT_HIERARHCY
from src.data.motion import *

def find_parent_and_children(joint_idx, joints):
    parent, children = None, None
    # find parent node
    for key, value in JOINT_HIERARHCY.items():
        if joint_idx in value:
            parent = {key: joints[key]}
            break
    # find children node
    for key, value in JOINT_HIERARHCY.items():
        if joint_idx == key:
            children = {}
            for v in value:
                children[v] = joints[v]
            break
    return parent, children


def c2w_transform(trans, data_c):
    '''
    Transform cam coordinate to world coordinate
    v_w = R_c2w • v_c + t_c2w
    '''
    R_c2w = trans[:3, :3] # rotation
    t_c2w = trans[:3, 3]  # translation
    return np.dot(R_c2w, data_c) + t_c2w


def c2w_transform(trans, data_c):
    R_c2w = trans[:3, :3] # rotation matrix
    t_c2w = trans[:3, 3]  # translation matrix
    data_w = np.dot(data_c - t_c2w, np.linalg.inv(R_c2w))
    return data_w

def c2w_transform_joints(trans, joints):
    R_c2w = trans[:3, :3] # rotation
    t_c2w = trans[:3, 3]  # translation
    return np.dot(joints, R_c2w) + t_c2w


def transform(trans_mat, data):
    transformed_data = np.dot(np.hstack((data, np.ones((len(data), 1)))), trans_mat.T)
    transformed_data = transformed_data[:, :3]  
    return transformed_data


def transl_transform(trans_mat, transl):
    transl_homogeneous = np.append(transl, 1)
    transformed_transl = np.dot(trans_mat, transl_homogeneous)
    return transformed_transl[:3]


def create_object_token(centers, sizes):
    centers = torch.tensor(centers, dtype=torch.float32)
    sizes = torch.tensor(sizes, dtype=torch.float32)
    # print("centers shape:", centers.shape)
    # print("sizes shape:", sizes.shape)

    object_tokens = torch.cat((centers, sizes), dim=0)
    # print("object_tokens shape:", object_tokens.shape)

    return object_tokens


def normalize(data, key, stats):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    
    normalized_data = (data - mean) / (std + 1e-8)  # avoid division by zero

    stats[key] = {'mean': mean.tolist(), 'std': std.tolist()}
    
    return normalized_data, stats


# def create_pose_vector(poses, translation, forward_direction, global_velocity, rotational_velocity, JOINT_NUM=22, save=True):
#     '''
#     Create pose vector

#     Args:
#         poses
#         translation
#         forward_direction
#         global_velocity
#         rotational_velocity

#     Returns:
#         init_pose_vector: (1, 72)
#         next_pose_vector: (NUM_SEQ, 76)
#     '''
#     # Converting inputs to tensors
#     poses = torch.tensor(poses, dtype=torch.float32) # (NUM_FRAMES, JOINT_NUM, 3)
#     translation_resized = torch.unsqueeze(torch.tensor(translation, dtype=torch.float32), dim=1)  # (NUM_FRAMES, 1, 3)
#     forward_direction_resized = torch.unsqueeze(torch.tensor(forward_direction, dtype=torch.float32), dim=1)  # (NUM_FRAMES, 1, 3)
#     global_velocity_resized = torch.unsqueeze(torch.tensor(global_velocity, dtype=torch.float32), dim=1)  # (NUM_FRAMES-1, 1, 3)
#     rotational_velocity_resized = torch.unsqueeze(torch.unsqueeze(torch.tensor(rotational_velocity, dtype=torch.float32), dim=1), dim=1)  # (NUM_FRAMES-1, 1, 1)
#     stats = {}
    
#     # Flatten data
#     poses_flatten = poses.view(-1, 22 * 3)  # (NUM_FRAMES, 22*3)
#     translation_flatten = translation_resized.view(-1, 3)  # (NUM_FRAMES, 3)
#     forward_direction_flatten = forward_direction_resized.view(-1, 3)  # (NUM_FRAMES, 3)
#     global_velocity_flatten = global_velocity_resized.view(-1, 3)  # ((NUM_FRAMES-1), 3)
#     rotational_velocity_flatten = rotational_velocity_resized.view(-1, 1)  # ((NUM_FRAMES-1), 1)

#     stats = {}
#     # Normalize each component and save 
#     poses_flatten, stats = normalize(poses_flatten, 'poses', stats)
#     translation_flatten, stats = normalize(translation_flatten, 'translation', stats)
#     forward_direction_flatten, stats = normalize(forward_direction_flatten, 'forward_direction', stats)
#     global_velocity_flatten, stats = normalize(global_velocity_flatten, 'global_velocity', stats)
#     rotational_velocity_flatten, stats = normalize(rotational_velocity_flatten, 'rotational_velocity', stats)
    
#     if save:
#         # Save stats to json file
#         with open('normalization_stats.json', 'w') as f:
#             json.dump(stats, f, indent=4)

#     init_pose_vector = torch.cat([poses_flatten[:1],
#                                   translation_flatten[:1],
#                                   forward_direction_flatten[:1]], dim=1)
    
#     next_pose_vector = torch.cat([poses_flatten[1:], 
#                                   translation_flatten[1:],
#                                   forward_direction_flatten[1:],
#                                   global_velocity_flatten,
#                                   rotational_velocity_flatten], dim=1)
    
#     return init_pose_vector, next_pose_vector


def load_data(data_dir, relative=True):
    # with open(os.path.join(data_dir, "exp1_prox.pkl"), 'rb') as input_file:
    #     motion_data_list = pickle.load(input_file)
    # print("Motion data path:", os.path.join(data_dir, "exp1_prox.pkl"))
    with open(os.path.join(data_dir, "prox_xform.pkl"), 'rb') as input_file:
        motion_data_list = pickle.load(input_file)
    print("Motion data path:", os.path.join(data_dir, "prox_xform.pkl"))
    with open(os.path.join(data_dir, "scene.pkl"), 'rb') as input_file:
        scene_data_list = pickle.load(input_file)
    print("Scene data path:", os.path.join(data_dir, "scene.pkl"))

    '''Load scene'''
    scene_dict = {}
    for scene_name in scene_data_list:
        obj_dict = scene_data_list[scene_name]
        obj_names = []; centers = []; sizes = []; 

        for obj_name, obj_info in obj_dict.items():
            obj_names.append(obj_name)
            centers.append(obj_info['center'])
            sizes.append(obj_info['size'])

        obj_token = create_object_token(np.array(centers), np.array(sizes))
        
        scene_dict[scene_name] = {'obj_names': obj_names, 'obj_token': obj_token}


    '''Load motion'''
    data_dict = {'name':[], 'frame_lists':[], 'object_boxes':[], 'labels':[], 'motions':[], 'translations':[], \
                 'forward_directions':[], 'global_velocities':[], 'rotational_velocities':[]}

    for data in motion_data_list: # all samples
        data_name = data['name']
        scene_name = data_name.split("_")[0]
        motion_list = data['motions']

        # if scene information is not in scene_dict, ignore this motion
        if scene_name not in scene_dict.keys():
            continue
        
        for motion_dict in motion_list: # each motion of sample
            label = motion_dict.pop('label')
            # forward_dir = motion_dict.pop('forward_direction')
            # global_vel = motion_dict.pop('global_velocity')
            # rot_vel = motion_dict.pop('rotational_velocity')
            
            poses = []; transl = []
            frame_list = list(motion_dict.keys())
            for frame in frame_list:
                poses.append(motion_dict[frame]['joints'])
                transl.append(motion_dict[frame]['transl'])
            
            poses = np.array(poses)
            transl = np.array(transl)

            forward_dir = compute_forward_dir(poses)
            global_vel, rot_vel = compute_velocity(poses, forward_dir)
            
            if relative:                
                # get relative joints 
                rel_joints = np.copy(poses)
                for idx, children_idx in JOINT_HIERARHCY.items():
                    for child_idx in children_idx:
                        poses[:,child_idx,:] -= rel_joints[:,idx,:]
            
            # apply gaussian filter to poses(motion) for smoothing
            filtered_motion = apply_gaussian_filter(np.array(poses))
            
            data_dict['name'].append(data_name)
            # add frame_list of current motion
            data_dict['frame_lists'].append(frame_list)
            # add label of current motion
            data_dict['labels'].append(label)
            # add 3d bounding box token
            data_dict['object_boxes'].append(scene_dict[scene_name]['obj_token'])
            # add motion data
            data_dict['motions'].append(filtered_motion)
            data_dict['translations'].append(transl)
            data_dict['forward_directions'].append(forward_dir)
            data_dict['global_velocities'].append(global_vel)
            data_dict['rotational_velocities'].append(rot_vel)
            
            # get pose vectors
            # init_pose_vector, next_pose_vector = create_pose_vector(filtered_motion, translations, \
            #                                                             forward_dir, global_vel, rot_vel)
            
            
            # add initial pose and next_motion of current motion
            # initial_poses.append(init_pose_vector)
            # next_motions.append(next_pose_vector)
    if relative:
        print("Use relative joint values")
        with open("./debug/relative_data.pkl", 'wb') as output:
            pickle.dump(data, output)
    
    else:
        print("Use abosolute joint values")
        with open("./debug/abs_data.pkl", 'wb') as output:
            pickle.dump(data, output)

    return data_dict
