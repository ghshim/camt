import pickle
import json
import os
import argparse
import cv2
import numpy as np
import open3d as o3d
import torch
import smplx

def load_scene(scene_dir):
    '''Load scene'''
    mesh_list = os.listdir(scene_dir)
    scenes = {}
    for mesh in sorted(mesh_list):
        filename, ext = os.path.splitext(mesh)
        if ext == '.pkl': continue

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
    
    return scenes


def viz_prox(data_name, objs_mesh_list, frame_list, smplx_model_dir, num_pca_comps=6):
    mover_dir = os.path.join('/home/gahyeon/Desktop/data/mover/MOVER_results', data_name)
    prox_dir = '/home/gahyeon/Desktop/data/prox'
    fitting_dir = os.path.join(prox_dir, "fittings", data_name, 'results')
    # recording_name = os.path.abspath(fitting_dir).split("/")[-1]
    color_dir = os.path.join(prox_dir, 'recordings', data_name, 'Color')
    scene_name = data_name.split("_")[0]
    cam2world_dir = os.path.join(prox_dir, 'cam2world')

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # with open(os.path.join(cam2world_dir, scene_name + '.json'), 'r') as f:
    #     trans = np.array(json.load(f))
    # with open(os.path.join(mover_dir, 'model_scene_1_lr0.002_end.json'), 'r') as f:
    #     trans = np.array(json.load(f))
    world2cam_dir = '/home/gahyeon/Desktop/data/mover/PROX_cropped_meshes_bboxes/PROX_cropped_meshes_bboxes/world2cam/qualitative_dataset'
    with open(os.path.join(world2cam_dir, scene_name, 'cam_gp.json'), 'r') as f:
        inverse_mat = json.load(f)
        inverse_mat = inverse_mat['inverse_mat']
    
    # 평행 이동 벡터 (없는 경우 [0, 0, 0]으로 가정)
    t_vec = np.array([0.0, 0.0, 0.0])

    # 3x3 회전 행렬을 4x4 변환 행렬로 확장
    trans = np.eye(4)
    trans[:3, :3] = inverse_mat
    trans[:3, 3] = t_vec

        
    gp_mesh = o3d.io.read_triangle_mesh(os.path.join(mover_dir, 'model_scene_1_lr0.002_end/gp_mesh.obj'))
    vis.add_geometry(gp_mesh)
    '''Load objects in the scene'''
    for obj in objs_mesh_list:
        obj_mesh = o3d.io.read_triangle_mesh(obj[1])
        # print("obj_mesh", np.asarray(obj_mesh.vertices))
        obj_mesh.transform(trans)
        vis.add_geometry(obj_mesh)
    
    '''Load body'''
    
    # smplx model
    female_subjects_ids = [162, 3452, 159, 3403]
    subject_id = int(data_name.split('_')[1])
    if subject_id in female_subjects_ids:
        gender = 'female'
    else:
        gender = 'male'

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    
    body_model_dict = {
        'male': smplx.create(os.path.join(smplx_model_dir,'SMPLX_MALE.npz'), model_type='smplx',
                             gender='male', ext='npz', num_betas = 10,
                             num_pca_comps=num_pca_comps),
        'female': smplx.create(os.path.join(smplx_model_dir,'SMPLX_FEMALE.npz'), model_type='smplx',
                               gender='female', ext='npz', num_betas = 10,
                               num_pca_comps=num_pca_comps),
        'neutral': smplx.create(os.path.join(smplx_model_dir,'SMPLX_NEUTRAL.npz'), model_type='smplx',
                               gender='neutral', ext='npz', num_betas = 10,
                               num_pca_comps=num_pca_comps)
    }

    
    # print(smpl_params.keys())
    if gender is not None:
        body_model = body_model_dict[gender]
    else:
        body_model = body_model_dict['neutral']
    
    count = 0

    for frame_name in sorted(frame_list):
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
        vertices = output.vertices.detach().cpu().numpy().squeeze()

        if count == 0:
            body = o3d.geometry.TriangleMesh()
            vis.add_geometry(body)

        body.vertices = o3d.utility.Vector3dVector(vertices)
        body.triangles = o3d.utility.Vector3iVector(body_model.faces)
        body.vertex_normals = o3d.utility.Vector3dVector([])
        body.triangle_normals = o3d.utility.Vector3dVector([])
        body.compute_vertex_normals()
        # body.transform(trans)

        color_img = cv2.imread(os.path.join(color_dir, frame_name + '.jpg'))
        color_img = cv2.flip(color_img, 1)

        vis.update_geometry(body)

        while True:
            cv2.imshow('frame', color_img)
            vis.poll_events()
            vis.update_renderer()
            key = cv2.waitKey(30)
            if key == 27:
                break

        count += 1


def main(args):
    camt_dir = args.camt_dir
    scene_dir = args.scene_dir
    smplx_model_dir = args.smplx_model_dir

    scenes = load_scene(scene_dir)
    data_path = os.path.join(camt_dir, 'prox.pkl')

    with open(data_path, 'rb') as input_file:
        data = pickle.load(input_file)

    for data_dict in data:
        data_name = data_dict['name']
        scene_name = data_name.split('_')[0]
        objs_mesh_list = scenes[scene_name]

        motion_list = data_dict['motions']

        print(data_name)
        
        for motion_dict in motion_list:
            # print(motion_dict.keys())
            label = motion_dict.pop('label')
            print(label)
            frame_list = list(motion_dict.keys())
            
            viz_prox(data_name, objs_mesh_list, frame_list, smplx_model_dir)


if __name__ == '__main__':
    '''
    python ./src/data/motion_sampling.py --sampling True --sampling_time 2 --xform True --saving True
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--prox_dir', type=str, help='the directory of prox dataset', default='/home/gahyeon/Desktop/data/prox')
    parser.add_argument('--camt_dir', type=str, default='/home/gahyeon/Desktop/data/camt/', help='outptu dir to save motion')
    parser.add_argument('--scene_dir', type=str, default='/home/gahyeon/Desktop/data//mover/PROX_cropped_meshes_bboxes/PROX_cropped_meshes_bboxes/qualitative_dataset/all_cropped_mesh', help='outptu dir to save motion')
    parser.add_argument('--smplx_model_dir', type=str, help='The directory of smplx body mdoel', default="/home/gahyeon/Desktop/models/body_models/smplx")
    
    args = parser.parse_args()
    
    main(args)