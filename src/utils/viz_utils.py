import pickle
import os
import argparse
import cv2
import shutil
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import json
import torch

from src.utils.data_utils import find_parent_and_children, transl_transform
from src.utils.constants import NUM_JOINTS, SKELETON, JOINT_NAMES, JOINT_HIERARHCY


def parse_object_tokens(object_tokens):
    """
    Extract centers and sizes from the concatenated object tokens.
    
    Parameters:
    - object_tokens: Concatenated tensor of centers and sizes, shape (2 * N, 3)
    
    Returns:
    - centers: Tensor of centers, shape (N, 3)
    - sizes: Tensor of sizes, shape (N, 3)
    """
    # Number of objects
    N = object_tokens.shape[0] // 2
    
    # Extract centers and sizes
    centers = object_tokens[:N, :]
    sizes = object_tokens[N:, :]
    
    return centers, sizes


def get_bbox_corners(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    
    # Compute half sizes
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    
    # Compute the 8 corners
    corners = [
        (cx - hx, cy - hy, cz - hz),
        (cx + hx, cy - hy, cz - hz),
        (cx - hx, cy + hy, cz - hz),
        (cx + hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz),
        (cx + hx, cy - hy, cz + hz),
        (cx - hx, cy + hy, cz + hz),
        (cx + hx, cy + hy, cz + hz)
    ]
    
    return corners


def parse_motion_vector(motion_vector):
    num_frames = motion_vector.shape[0]
    
    poses = np.zeros((num_frames, NUM_JOINTS, 3))
    poses[:] = motion_vector[:, :NUM_JOINTS*3].reshape(num_frames, NUM_JOINTS, 3)
    
    translation = np.zeros((num_frames, 3))
    translation[:] = motion_vector[:, NUM_JOINTS*3:NUM_JOINTS*3+3]
    
    forward_direction = np.zeros((num_frames, 3))
    forward_direction[:] = motion_vector[:, NUM_JOINTS*3+3:NUM_JOINTS*3+6]
    
    global_velocity = np.zeros((num_frames, 3))
    global_velocity[:] = motion_vector[:, NUM_JOINTS*3+6:NUM_JOINTS*3+9]
    
    rotational_velocity = np.zeros((num_frames, 1))
    rotational_velocity[:] = motion_vector[:, NUM_JOINTS*3+9:]
    
    return poses, translation, forward_direction, global_velocity, rotational_velocity


def get_abs_poses(joints):
    def find_children_idx(joint_idx):
        parent_idx, children_idx = None, None
        # find parent node
        for key, value in JOINT_HIERARHCY.items():
            if joint_idx in value:
                parent_idx = key
                break
        for key, value in JOINT_HIERARHCY.items():
            if joint_idx == key:
                children_idx = []
                for v in value:
                    children_idx.append(v)
                break
        return parent_idx, children_idx
    ''''
    Make pose using Joint
    '''
    abs_joints = np.copy(joints)
    
    # get absolute joint value by adding parent
    for idx in range(abs_joints.shape[1]):
        
        # find parent and child node of current node
        parent_idx, children_idx = find_children_idx(idx)
        
        if children_idx is not None:
            for child_idx in children_idx:
                abs_joints[:, child_idx, :] += abs_joints[:, idx, :]
            
        # if parent_idx is not None:
        #     print(f"current idx: {idx}, parent idx: {parent_idx}")
        #     abs_joints[:, idx, :] += abs_joints[:, parent_idx, :]
        # else:
        #     print(f"current idx: {idx}, this is root")

    return abs_joints


def rotate_vector(vector, angle, axis):
    """
    Rotate vector by angle around axis using Rodrigues' rotation formula
    """
    axis = axis / np.linalg.norm(axis)
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)
    cross_prod_matrix = np.array([[0, -axis[2], axis[1]],
                                  [axis[2], 0, -axis[0]],
                                  [-axis[1], axis[0], 0]])
    rotation_matrix = (cos_theta * np.eye(3) + 
                       sin_theta * cross_prod_matrix + 
                       (1 - cos_theta) * np.outer(axis, axis))
    return np.dot(rotation_matrix, vector)


# Define a function to plot the camera
def plot_camera(ax, R, t, scale=0.5):
    '''
    cam2world
    R_c C
     0  1
    C: camera position in world coordinate
    R_c: camera direction
    '''
    # Camera origin
    origin = t

    # Camera basis vectors
    x_axis = origin + R[:, 0] * scale
    y_axis = origin + R[:, 1] * scale
    z_axis = origin + R[:, 2] * scale

    # Plot the camera origin
    ax.scatter(*origin, color='r')

    # Plot the camera basis vectors
    ax.quiver(*origin, *(x_axis - origin), color='r', length=scale, normalize=True)
    ax.quiver(*origin, *(y_axis - origin), color='g', length=scale, normalize=True)
    ax.quiver(*origin, *(z_axis - origin), color='b', length=scale, normalize=True)

    # Add labels for the camera axes
    ax.text(*x_axis, 'X_c', color='r')
    ax.text(*y_axis, 'Y_c', color='g')
    ax.text(*z_axis, 'Z_c', color='b')


def viz_motion(motion, corners):
    poses, translation, forward_direction, global_velocity, rotational_velocity = motion
    num_frames = len(poses)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.cla()
    
    def update(frame):
        ax.clear()
        
        # Calculate the current translation and forward direction based on velocities
        current_translation = translation[frame]
        
        # if frame == 0:
        #     forward_direction = np.array([1, 0, 0])
        # else:
        #     # current_translation = translation[frame] #+ global_velocity[frame - 1]
        #     current_forward_direction = forward_direction[frame - 1]
        #     # current_forward_direction = rotate_vector(forward_direction[frame - 1], 
        #     #                                           rotational_velocity[frame - 1], 
        #     #                                           np.array([0, 1, 0]))  # Assuming Y-axis rotation
        current_pose = poses[frame,:,:]
        
        # if frame < num_frames:
        #     current_pose = poses[frame,:,:]
        # else:
        #     current_pose = poses[-1,:,:]
        
        # Translate the current pose to ensure it's attached to the ground
        # min_x = np.min(gt_current_pose[:, 0])
        # min_z = np.min(gt_current_pose[:, 2])
        # ground_translation = np.array([-min_x, 0, -min_z])
        
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # Update scatter plot
        # scatter = ax.scatter(current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
        
        # Update the human body position
        human_body = current_pose + current_translation
        ax.scatter(human_body[:, 0], human_body[:, 1], human_body[:, 2], c='r')
        
        # Draw the forward direction vector
        # ax.quiver(current_translation[0], current_translation[1], current_translation[2], 
        #           current_forward_direction[0], current_forward_direction[1], current_forward_direction[2], color='g')
        
        # Draw connections between joints (assuming some SKELETON structure, adjust as needed)
        for joint_start, joint_end in SKELETON:
            ax.plot([human_body[joint_start, 0], human_body[joint_end, 0]],
                    [human_body[joint_start, 1], human_body[joint_end, 1]],
                    [human_body[joint_start, 2], human_body[joint_end, 2]], 'k-')
        
        
        # Plot bounding boxes
        for i in range(len(corners)):
            # Define the 12 lines connecting the corners of the bounding box
            edges = [
                [corners[i][j] for j in [0, 1, 3, 2, 0]],
                [corners[i][j] for j in [4, 5, 7, 6, 4]],
                [corners[i][j] for j in [0, 4]],
                [corners[i][j] for j in [1, 5]],
                [corners[i][j] for j in [2, 6]],
                [corners[i][j] for j in [3, 7]]
            ]
            # poly3d = [[bbox_corners[vert] for vert in face] for face in [
            #     [0, 1, 5, 4], [7, 6, 2, 3], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]]
            # ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
            
            for edge in edges:
                ax.plot([point[0] for point in edge], [point[1] for point in edge], [point[2] for point in edge], 'g-')
        
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100)
    
    return ani


def compare_motion(gt_motion, pred_motion, corners):
    gt_poses = gt_motion[0]
    gt_translation = gt_motion[1]
    gt_forward_direction = gt_motion[2]
    gt_global_velocity = gt_motion[3]
    gt_rotational_velocity = gt_motion[4]

    pred_poses = pred_motion[0]
    pred_translation = pred_motion[1]
    pred_forward_direction = pred_motion[2]
    pred_global_velocity = pred_motion[3]
    pred_rotational_velocity = pred_motion[4]

    # poses, translation, forward_direction, global_velocity, rotational_velocity = parse_motion_vector(motion_vector, NUM_JOINTS)
    num_frames = len(gt_poses)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.cla()
        
    # Initialize scatter plot
    # scatter = ax.scatter([], [], [])
    
    def update(frame):
        ax.clear()
        
        # Calculate the current translation and forward direction based on velocities
        gt_current_translation = gt_translation[frame]
        pred_current_translation = pred_translation[frame]
        
        if frame == 0:
            current_forward_direction = np.array([1, 0, 0])
        else:
            # current_translation = translation[frame] #+ global_velocity[frame - 1]
            current_forward_direction = gt_forward_direction[frame - 1]
            # current_forward_direction = rotate_vector(forward_direction[frame - 1], 
            #                                           rotational_velocity[frame - 1], 
            #                                           np.array([0, 1, 0]))  # Assuming Y-axis rotation
        gt_current_pose = gt_poses[frame,:,:]
        pred_current_pose = pred_poses[frame,:,:]
        # if frame < num_frames:
        #     current_pose = poses[frame,:,:]
        # else:
        #     current_pose = poses[-1,:,:]
        
        # Translate the current pose to ensure it's attached to the ground
        # min_x = np.min(gt_current_pose[:, 0])
        # min_z = np.min(gt_current_pose[:, 2])
        # ground_translation = np.array([-min_x, 0, -min_z])
        
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        
        # Update scatter plot
        # scatter = ax.scatter(current_pose[:, 0], current_pose[:, 1], current_pose[:, 2])
        
        # Update the human body position
        human_body = gt_current_pose + gt_current_translation
        ax.scatter(human_body[:, 0], human_body[:, 1], human_body[:, 2], c='r')
        
        # Draw the forward direction vector
        # ax.quiver(current_translation[0], current_translation[1], current_translation[2], 
        #           current_forward_direction[0], current_forward_direction[1], current_forward_direction[2], color='g')
        
        # Draw connections between joints (assuming some SKELETON structure, adjust as needed)
        for joint_start, joint_end in SKELETON:
            ax.plot([human_body[joint_start, 0], human_body[joint_end, 0]],
                    [human_body[joint_start, 1], human_body[joint_end, 1]],
                    [human_body[joint_start, 2], human_body[joint_end, 2]], 'k-')
        
        human_body = pred_current_pose + pred_current_translation
        ax.scatter(human_body[:, 0], human_body[:, 1], human_body[:, 2], c='b')
        
        # Draw the forward direction vector
        # ax.quiver(current_translation[0], current_translation[1], current_translation[2], 
        #           current_forward_direction[0], current_forward_direction[1], current_forward_direction[2], color='g')
        
        # Draw connections between joints (assuming some SKELETON structure, adjust as needed)
        for joint_start, joint_end in SKELETON:
            ax.plot([human_body[joint_start, 0], human_body[joint_end, 0]],
                    [human_body[joint_start, 1], human_body[joint_end, 1]],
                    [human_body[joint_start, 2], human_body[joint_end, 2]], 'k-')
        
        # Plot bounding boxes
        for i in range(len(corners)):
            # Define the 12 lines connecting the corners of the bounding box
            edges = [
                [corners[i][j] for j in [0, 1, 3, 2, 0]],
                [corners[i][j] for j in [4, 5, 7, 6, 4]],
                [corners[i][j] for j in [0, 4]],
                [corners[i][j] for j in [1, 5]],
                [corners[i][j] for j in [2, 6]],
                [corners[i][j] for j in [3, 7]]
            ]
            # poly3d = [[bbox_corners[vert] for vert in face] for face in [
            #     [0, 1, 5, 4], [7, 6, 2, 3], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]]
            # ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
            
            for edge in edges:
                ax.plot([point[0] for point in edge], [point[1] for point in edge], [point[2] for point in edge], 'g-')
        
    ani = FuncAnimation(fig, update, frames=num_frames, interval=100)
    
    return ani


def main(args):
    camt_dir = '/home/gahyeon/Desktop/data/camt'
    data_path = args.data_path
    idx = args.idx
    output_dir = args.output_dir
    # prox_dir = args.prox_dir
    view = args.view
    save = args.save

    with open(os.path.join(data_path, f"prediction/{idx}.pkl"), 'rb') as file:
        data = pickle.load(file)

    with open(os.path.join(camt_dir, "scene.pkl"), 'rb') as input_file:
        scene_data_list = pickle.load(input_file)
    print("Scene data path:", os.path.join(camt_dir, "scene.pkl"))

    
    data_name = data['name'][0]
    frame_list = data['frame_list']
    label = data['label']
    object_box = data['object_box'][0]
    prediction = data['prediction'][0]
    gt = data['gt'][0]

    # print(object_box.shape)
    
    '''Load scene'''
    scene_dict = {}
    for scene_name in scene_data_list:
        obj_dict = scene_data_list[scene_name]
        obj_names = []; corners = []

        for obj_name, obj_info in obj_dict.items():
            obj_names.append(obj_name)
            corners.append(obj_info['corners'])
                    
        scene_dict[scene_name] = {'obj_names': obj_names, 'corners': corners}

    scene_name = data_name.split("_")[0]
    scene_dict[scene_name]['corners']


    print(data_name)
    print(label)
    # print(frame_list)
    gt_motion, gt_transl, gt_forward_dir, gt_global_vel, gt_rot_vel = parse_motion_vector(gt)
    gt_motion = get_abs_poses(gt_motion)
    # print(gt_motion[0])

    pred_motion, pred_transl, pred_forward_dir, pred_global_vel, pred_rot_vel = parse_motion_vector(prediction)
    pred_motion = np.concatenate([gt_motion[:1,:,:], pred_motion], axis=0)
    pred_transl = np.concatenate([gt_transl[:1,:], pred_transl], axis=0)
    pred_motion = get_abs_poses(pred_motion)
    # print(pred_motion[0])
    
    # centers, sizes = parse_object_tokens(object_box)
    
    ani = compare_motion((gt_motion, gt_transl, gt_forward_dir, gt_global_vel, gt_rot_vel),
                           (pred_motion, pred_transl, pred_forward_dir, pred_global_vel, pred_rot_vel),
                            corners)
    if view:
        plt.show()   

    if save:
        # save animation
        os.makedirs(os.path.join(data_path, 'result'), exist_ok=True)
        output_path = os.path.join(data_path, 'result', f"{idx}.mp4")
        writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(output_path, writer=writer)
        print('Saved in', output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # data_path = '/home/gahyeon/Desktop/projects/camt/result/pred_result.pkl'
    parser.add_argument('--data_path', type=str, default='/home/gahyeon/Desktop/data/camt/prox_sentence_encoded.pkl', help='the directory of data')
    parser.add_argument('--idx', type=int)
    parser.add_argument('--output_dir', type=str, default='./result/output', help='the output directory of data')
    parser.add_argument('--mode', type=str, default='test', help='the directory of data')
    parser.add_argument('--view', action="store_true")
    parser.add_argument('--save', action="store_true")
    args = parser.parse_args()

    main(args)