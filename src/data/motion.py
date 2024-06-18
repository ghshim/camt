import numpy as np
import os
import pickle
import torch

from src.utils.constants import JOINT_HIERARHCY, JOINT_NAMES, EPSILON


def apply_gaussian_filter(motion, sigma=1.0):
    """
    This code is borrowed from https://github.com/SinMDM/SinMDM/

    poses: (FRAME_NUM, JOINT_NUM, 3)
    """
    from scipy.ndimage import gaussian_filter
    
    motion = motion.reshape(motion.shape[0], -1)
    # print(motion.shape)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    
    filtered_motion = motion.reshape(motion.shape[0], -1, 3)
    return filtered_motion


def compute_forward_dir(motion):
    """
    Calculates the 3D forward direction of the body in the local coordinate system.
    3-dimensional forward direction of the body in local coordinate system
    3. joint poisions are defined in the body's local coordinate system, with the origin at the ground where the root poistion is projected.
    The forward direction of the body (Z-axis) is computed using vectors across the left and right shoulders and hips averaged and cross producted with the vertical axis (Y-axis)

    Args:
        joints: A numpy array of shape (num_joints, 3) containing 3D joint positions.

    Returns:
        A numpy array of shape (3,) representing the forward direction (Z-axis).
    """
    shoulder_l_joint = JOINT_NAMES.index('left_shoulder') # 16
    shoulder_r_joint = JOINT_NAMES.index('right_shoulder') # 17
    hip_l_joint = JOINT_NAMES.index('left_hip') # 1
    hip_r_joint = JOINT_NAMES.index('right_hip') # 2

    forward_dirs = []
    for pose in motion:
        # Extract left shoulder, right shoulder, left hip, and right hip joints.
        shoulder_l = pose[shoulder_l_joint, :]
        shoulder_r = pose[shoulder_r_joint, :]
        hip_l = pose[hip_l_joint, :]
        hip_r = pose[hip_r_joint, :]
        
        # calculate shoulder vector and hip vector.
        shoulder_vec = shoulder_l - shoulder_r
        hip_vec = hip_l - hip_r

        # average shoulder and hip vectors.
        avg_vec = (shoulder_vec + hip_vec) / 2.0

        # conduct cross product with the vertical axis (Y-axis)
        # 1) vertical axis (Y-axis) is (0, 1, 0).
        vertical_axis = np.array([0, 1, 0])
        # 2) cross product to get the forward direction (Z-axis).
        dir = np.cross(avg_vec, vertical_axis)

        forward_dirs.append(dir)

    return np.array(forward_dirs)


def compute_velocity(motion, forward_dir, dt=0.033):
    """
    3-dimensional global velocity of the body
    4. global velocity in the XZ-plane and rotational velocity of the body around the vertical axis (Y-axis) are appended 
    to the input representation for each frame. These can be integrated over time to recover the global translation and rotation of the motion
    """
    assert dt > EPSILON
    
    # calculate linear global velocity in XZ-plane using forward direction
    pelvis_joint = JOINT_NAMES.index('pelvis') # 0
    global_velocity = (motion[1:, pelvis_joint, :] - motion[:-1, pelvis_joint, :]) / dt
    
    # calculate rotational velocity around the vertical axis (Y-axis)
    rotational_velocity = []
    for i in range(1, len(motion)):
        pose1, pose2 = forward_dir[i-1], forward_dir[i]
        dot_product = np.dot(pose1, pose2)
        magnitude_product = np.linalg.norm(pose1) * np.linalg.norm(pose2)
        cosine_angle = dot_product / magnitude_product
        theta = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        rotational_velocity.append(theta)


    return np.array(global_velocity), np.array(rotational_velocity)