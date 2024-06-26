import pickle
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.utils.viz_utils import viz_motion

def main(args):
    with open(os.path.join(args.output_dir, "prox_xform.pkl"), 'rb') as input_file:
        data = pickle.load(input_file)

    for data_dict in data:
        data_name = data_dict['name']
        motion_list = data_dict['motions']
        
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