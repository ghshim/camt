import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.utils.viz_utils import viz_motion

def main(args):
    data_dir = args.data_dir

    with open(os.path.join(data_dir, "scene.pkl"), 'rb') as input_file:
        scene_data_list = pickle.load(input_file)
    print("Scene data path:", os.path.join(data_dir, "scene.pkl"))

    '''Load scene'''
    scene_dict = {}
    for scene_name in scene_data_list:
        obj_dict = scene_data_list[scene_name]
        obj_names = []; corners = []

        for obj_name, obj_info in obj_dict.items():
            obj_names.append(obj_name)
            corners.append(obj_info['corners'])
                    
        scene_dict[scene_name] = {'obj_names': obj_names, 'corners': corners}


    with open(os.path.join(data_dir, "prox_xform.pkl"), 'rb') as input_file:
        data = pickle.load(input_file)

    for data_dict in data:
        data_name = data_dict['name']
        motion_list = data_dict['motions']

        scene_name = data_name.split("_")[0]
        corners = scene_dict[scene_name]['corners']   
        
        for motion_dict in motion_list:
            # print(motion_dict.keys())
            label = motion_dict.pop('label')
            print(label)
            # sentence_embedding = motion_dict.pop('sentence_embedding')
            
            
            frame_list = list(motion_dict.keys())
            # print(len(frame_lists))
            # count += 1
            # continue
            poses = []
            transl = []
            for frame in frame_list:
                poses.append(motion_dict[frame]['joints'])
                transl.append(motion_dict[frame]['transl'])
            poses = np.array(poses)
            transl = np.array(transl)
            ani = viz_motion((poses, transl, None, None, None), corners)
            plt.show()
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
            break        
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # data_path = '/home/gahyeon/Desktop/projects/camt/result/pred_result.pkl'
    parser.add_argument('--data_dir', type=str, default='/home/gahyeon/Desktop/data/camt', help='the directory of data')
    parser.add_argument('--view', action="store_true")
    parser.add_argument('--save', action="store_true")
    args = parser.parse_args()

    main(args)