import numpy as np
import os
import pickle
import torch

from src.utils.constants import JOINT_HIERARHCY


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