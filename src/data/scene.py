import trimesh
import os
import argparse
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
from src.utils.data_utils import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_center_and_size(min_corner, max_corner):
    # Compute center
    center = ((min_corner[0] + max_corner[0]) / 2,
              (min_corner[1] + max_corner[1]) / 2,
              (min_corner[2] + max_corner[2]) / 2)
    
    # Compute size
    size = (max_corner[0] - min_corner[0],
            max_corner[1] - min_corner[1],
            max_corner[2] - min_corner[2])
    
    return center, size


def find_min_max_corners(corners):
    '''
    Get min and max corners from 3D object bounding boxes
    '''
    min_index = np.argmin(np.sum(corners, axis=1))
    min_corner = corners[min_index]
    max_index = np.argmax(np.sum(corners, axis=1))
    max_corner = corners[max_index]
    return min_corner, max_corner


def main(args):
    scene_dir = args.scene_dir
    prox_dir = args.prox_dir
    output_dir = args.output_dir
    cam2world_dir = os.path.join(prox_dir, 'cam2world')

    mesh_list = os.listdir(scene_dir)
    scenes = {}
    for mesh in sorted(mesh_list):
        filename, ext = os.path.splitext(mesh)
        if ext == '.ply': continue

        filename_split = filename.split("_")
        scene_name = filename_split[0]
        
        if len(filename_split[-1]) == 1:
            obj_name = filename_split[-2] + filename_split[-1]
        else:
            obj_name = filename_split[-1]

        # add scene in scenes dict
        if scene_name not in scenes.keys():
            scenes[scene_name] = []

        # add objects in the scene
        scenes[scene_name].append((obj_name, os.path.join(scene_dir, mesh)))
    
    
    output = {}   
    for scene_name, objects in scenes.items():
        with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
            trans = np.array(json.load(f))
        print(scene_name)

        obj_names = []
        centers = []
        sizes = []
        corners = []
        output[scene_name] = {}
        for obj in objects:
            obj_name = obj[0]
            obj_path = obj[1]
            
            obj_mesh = pickle.load(open(obj_path, 'rb'))
            vertices = obj_mesh['vertices']
            # vertices = c2w_transform(trans, obj_mesh['vertices'])
            
            min_corner, max_corner = find_min_max_corners(vertices)
            
            center, size = compute_center_and_size(min_corner, max_corner)
           
            obj_names.append(obj_name)
            centers.append(center)
            sizes.append(size)
            corners.append(vertices)

            
            # output[scene_name][obj_name] = {}

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim([-2, 2])
        # ax.set_ylim([-2, 2])
        # ax.set_zlim([-2, 2])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Z')
        # ax.set_zlabel('Y')
        # for i in range(len(corners)):
        #     edges = [
        #         [corners[i][j] for j in [0, 1, 3, 2, 0]],
        #         [corners[i][j] for j in [4, 5, 7, 6, 4]],
        #         [corners[i][j] for j in [0, 4]],
        #         [corners[i][j] for j in [1, 5]],
        #         [corners[i][j] for j in [2, 6]],
        #         [corners[i][j] for j in [3, 7]]
        #     ]
        #     # poly3d = [[corners[i][vert] for vert in face] for face in [
        #     #     [0, 1, 5, 4], [7, 6, 2, 3], [0, 3, 7, 4], [1, 2, 6, 5], [0, 1, 2, 3], [4, 5, 6, 7]]]
        #     # ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
            
        #     for edge in edges:
        #         ax.plot([point[0] for point in edge], [point[1] for point in edge], [point[2] for point in edge], 'b-')

        #     # Plot the min_corner, max_corner, and center
        #     # ax.scatter(*min_corner, color='red', s=100, label='Min Corner')
        #     # ax.scatter(*max_corner, color='red', s=100, label='Max Corner')
        #     ax.scatter(*center, color='red', s=100, label='Center')

        #     # Plot the size along each axis
        #     ax.plot([center[0], center[0] + size[0] / 2], [center[1], center[1]], [center[2], center[2]], 'b-')
        #     # ax.plot([center[0], center[0] - size[0] / 2], [center[1], center[1]], [center[2], center[2]], 'g-')
        #     ax.plot([center[0], center[0]], [center[1], center[1] + size[1] / 2], [center[2], center[2]], 'g-')
        #     # ax.plot([center[0], center[0]], [center[1], center[1] - size[1] / 2], [center[2], center[2]], 'g-')
        #     ax.plot([center[0], center[0]], [center[1], center[1]], [center[2], center[2] + size[2] / 2], 'r-')
        #     # ax.plot([center[0], center[0]], [center[1], center[1]], [center[2], center[2] - size[2] / 2], 'g-')
    
        # plt.show()
        
        
        # normalized_centers = normalize(np.array(centers))
        # normalized_sizes = normalize(np.array(sizes))

        # print("centers:", centers)
        # print("normalized centers:", normalized_centers)
        # print("-----------------------------------")
        # print("sizes:", sizes)
        # print("normalized sizes:", normalized_sizes)
        # print("-----------------------------------\n")

        
        for i in range(len(obj_names)):
            # print(obj_name)
            output[scene_name][obj_names[i]] = {'center': centers[i], 'size': sizes[i], 'corners': corners[i]}
    
    with open(os.path.join(output_dir, 'scene.pkl'), 'wb') as output_file:
        pickle.dump(output, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_dir', type=str, default='/home/gahyeon/Desktop/data/mover/PROX_cropped_meshes_bboxes/PROX_cropped_meshes_bboxes/qualitative_dataset/obb', help='the directory of 3d bounding boxes')
    parser.add_argument('--prox_dir', type=str, default='/home/gahyeon/Desktop/data/prox', help='the directory of prox dataset')
    # parser.add_argument('--label_dir', type=str, default='/home/gahyeon/Desktop/data/prox/interaction_labels_1', help='the directory of prox dataset')
    # parser.add_argument('--sampling', default=False)
    # parser.add_argument('--time', default=7)
    # parser.add_argument('--saving', default=False)
    # parser.add_argument('--ext', type=str, default='jpg', help='the image file extension')
    
    parser.add_argument('--output_dir', type=str, default='/home/gahyeon/Desktop/data/camt/', help='output dir')
    args = parser.parse_args()
    
    main(args)

    # with open(os.path.join(args.output_dir, "scene.pkl"), 'rb') as input_file:
    #     data = pickle.load(input_file)

    # for scene, objects in data.items():
    #     print(scene)
    #     print(objects)
    #     print(data[scene].keys())