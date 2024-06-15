import numpy as np
import os
import pickle
import torch

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


def transform(trans_mat, data):
    transformed_data = np.dot(np.hstack((data, np.ones((len(data), 1)))), trans_mat.T)
    transformed_data = transformed_data[:, :3]  
    return transformed_data


def transl_transform(trans_mat, translations):
    # Convert translations to homogeneous coordinates
    num_frames = translations.shape[0]
    homogeneous_translations = np.hstack((translations, np.ones((num_frames, 1))))
    
    # Apply the transformation matrix
    transformed_homogeneous = (trans_mat @ homogeneous_translations.T).T
    
    # Convert back from homogeneous coordinates
    transformed_translations = transformed_homogeneous[:, :3] / transformed_homogeneous[:, 3][:, np.newaxis]
    
    return transformed_translations


def create_object_token(centers, sizes):
    centers = torch.tensor(centers, dtype=torch.float32)
    sizes = torch.tensor(sizes, dtype=torch.float32)
    # print("centers shape:", centers.shape)
    # print("sizes shape:", sizes.shape)

    object_tokens = torch.cat((centers, sizes), dim=0)
    # print("object_tokens shape:", object_tokens.shape)

    return object_tokens


def create_pose_vector(poses, translation, forward_direction, global_velocity, rotational_velocity, JOINT_NUM=22):
    '''
    Create pose vector

    Args:
        poses
        translation
        forward_direction
        global_velocity
        rotational_velocity

    Returns:
        init_pose_vector: (1, 72)
        next_pose_vector: (NUM_SEQ, 76)
    '''
    poses = torch.tensor(poses, dtype=torch.float32) # (NUM_FRAMES, JOINT_NUM, 3)
    translation_resized = torch.unsqueeze(torch.tensor(translation, dtype=torch.float32), dim=1)  # (NUM_FRAMES, 1, 3)
    forward_direction_resized = torch.unsqueeze(torch.tensor(forward_direction, dtype=torch.float32), dim=1)  # (NUM_FRAMES, 1, 3)
    global_velocity_resized = torch.unsqueeze(torch.tensor(global_velocity, dtype=torch.float32), dim=1)  # (NUM_FRAMES-1, 1, 3)
    rotational_velocity_resized = torch.unsqueeze(torch.unsqueeze(torch.tensor(rotational_velocity, dtype=torch.float32), dim=1), dim=1)  # (NUM_FRAMES-1, 1, 1)

    poses_flatten = poses.view(-1, 22 * 3)  # (NUM_FRAMES, 22*3)
    translation_flatten = translation_resized.view(-1, 3)  # (NUM_FRAMES, 3)
    forward_direction_flatten = forward_direction_resized.view(-1, 3)  # (NUM_FRAMES, 3)
    global_velocity_flatten = global_velocity_resized.view(-1, 3)  # ((NUM_FRAMES-1), 3)
    rotational_velocity_flatten = rotational_velocity_resized.view(-1, 1)  # ((NUM_FRAMES-1), 1)

    init_pose_vector = torch.cat([poses_flatten[:1],
                                  translation_flatten[:1],
                                  forward_direction_flatten[:1]], dim=1)
    
    next_pose_vector = torch.cat([poses_flatten[1:], 
                                  translation_flatten[1:],
                                  forward_direction_flatten[1:],
                                  global_velocity_flatten,
                                  rotational_velocity_flatten], dim=1)
    
    return init_pose_vector, next_pose_vector


def load_data(data_dir, relative=True):
    with open(os.path.join(data_dir, "exp1_prox.pkl"), 'rb') as input_file:
        motion_data_list = pickle.load(input_file)
    
    with open(os.path.join(data_dir, "scene.pkl"), 'rb') as input_file:
        scene_data_list = pickle.load(input_file)

    initial_poses = []; next_motions = []; labels = []; 
    object_boxes = []
    frame_lists = []
    
    '''Load scene'''
    scene_dict = {}
    for scene_name in scene_data_list:
        obj_dict = scene_data_list[scene_name]
        obj_names = []; centers = []; sizes = []

        for obj_name, obj_info in obj_dict.items():
            obj_names.append(obj_name)
            centers.append(obj_info['center'])
            sizes.append(obj_info['size'])
        
        obj_token = create_object_token(np.array(centers), np.array(sizes))
        
        scene_dict[scene_name] = {'obj_names': obj_names, 'obj_token': obj_token}

    '''Load motion'''
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
            
            poses = []; translations = []
            frame_list = list(motion_dict.keys())
            for frame in frame_list:
                poses.append(motion_dict[frame]['joints'])
                translations.append(motion_dict[frame]['transl'])

            poses = np.array(poses); translations = np.array(translations)
            forward_dir = compute_forward_dir(poses)
            global_vel, rot_vel = compute_velocity(poses, forward_dir)

            if relative:
                # get relative joints 
                rel_joints = np.copy(poses)
                for idx, children_idx in JOINT_HIERARHCY.items():
                    for child_idx in children_idx:
                        poses[:,child_idx,:] -= rel_joints[:,idx,:]
            
            # apply gaussian filter to poses(motion) for smoothing
            filtered_motion = apply_gaussian_filter(poses)

            # get pose vectors
            init_pose_vector, next_pose_vector = create_pose_vector(filtered_motion, translations, \
                                                                        forward_dir, global_vel, rot_vel)
            
            # add frame_list of current motion
            frame_lists.append(frame_list)
            # add label of current motion
            labels.append(label)
            # add initial pose and next_motion of current motion
            initial_poses.append(init_pose_vector)
            next_motions.append(next_pose_vector)
            # 3d bounding box token
            object_boxes.append(scene_dict[scene_name]['obj_token'])
        
    return frame_lists, initial_poses, next_motions, labels, object_boxes
