import numpy as np
import json
import torch
from torch.utils.data import random_split, Dataset, DataLoader
from src.utils.data_utils import load_data
from src.utils.constants import NUM_JOINTS
from src.data.sentence_transformer import encode


class MotionDataset(Dataset):
    def __init__(
            self, 
            data_list,
            frame_lists, 
            object_boxes, 
            labels,
            motions,
            translations,
            forward_directions,
            global_velocities,
            rotational_velocities,
            device=None
        ):
        self.data_list = data_list
        self.frame_lists = frame_lists
        self.labels = labels
        self.labels_embeddings = self.encode_labels() # sentence encoding using SentenceTransformer
        self.object_boxes = object_boxes   
        self.device = device     

        self.initial_poses, self.next_motions = self.create_pose_vector(motions,
                                                                        translations,
                                                                        forward_directions,
                                                                        global_velocities,
                                                                        rotational_velocities)

        # if normalize:
        #     self.mean_pose, self.std_pose = None, None
        #     self.mean_gvel, self.std_gvel = None, None
        #     self.mean_rvel, self.std_rvel = None, None
        #     standardized_poses, standardized_global_velocity, standardized_rotational_velocity = self.standardize_data(self.initial_poses)
    
    def __getitem__(self, index):
        """Returns one data pair (input data, output data)."""
        data_name = self.data_list[index]
        frame_list = self.frame_lists[index]
        label = self.labels[index]

        initial_pose = torch.Tensor(self.initial_poses[index]).to(self.device) # (1, 72)
        next_motion = torch.Tensor(self.next_motions[index]).to(self.device) # (NUM_SEQ, 76)
        motion_descr = torch.Tensor(self.labels_embeddings[index]).to(self.device) # (384,)
        object_box = torch.Tensor(self.object_boxes[index]).to(self.device)
            
        # print('initial_pose.shape', initial_pose.shape)
        # print('object_box.shape', object_box.shape)
        # print('motion_description.shape', motion_description.shape)
        # print('next_motion.shape:', next_motion.shape)
        # print('------------------------')
        return data_name, frame_list, label, initial_pose, next_motion, motion_descr, object_box
    
    def __len__(self):
        return len(self.initial_poses)

    def encode_labels(self):
        embeddings = encode(self.labels)
        return embeddings
    
    def compute_stats(self, data):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return mean, std

    def normalize_data(self, data):
        # print(data.shape)
        mean = np.mean(data, axis=(0))
        std = np.std(data, axis=(0))
        normalized_data = (data - mean) / (std + 1e-8)  # Avoid division by zero
        return normalized_data, mean.tolist(), std.tolist()

    def save_stats_to_json(self, stats_dict, save_path):
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, indent=4)

    def create_pose_vector(self, motions, translations, forward_directions, global_velocities, rotational_velocities, JOINT_NUM=22, save=True):
        '''
        Create pose vector

        Args:
            motions: (num_data, seq_len, num_joints, 3)
            translation: (num_data, seq_len, 3)
            forward_direction: (num_data, seq_len, 3)
            global_velocity: (num_data, seq_len-1, 3)
            rotational_velocity: (num_data, seq_len-1)

        Returns:
            init_pose_vector: (num_data, 1, 72)
            next_pose_vector: (num_data, seq_len, 76)
        '''
        num_data = len(motions)
        init_poses = []
        next_motions = []

        # Normalize each type of data
        # normalized_motions, motions_mean, motions_std = self.normalize_data(motions)
        # normalized_translations, transl_mean, transl_std = self.normalize_data(translations)
        # normalized_forward_directions, forward_dir_mean, forward_dir_std = self.normalize_data(forward_directions)
        # normalized_global_velocities, global_vel_mean, global_vel_std = self.normalize_data(global_velocities)
        # normalized_rotational_velocities, rot_vel_mean, rot_vel_std = self.normalize_data(rotational_velocities)

        # Save means and stds to JSON
        # if save:
        #     stats_dict = {
        #         'motions': {'mean': motions_mean, 'std': motions_std},
        #         'translations': {'mean': transl_mean, 'std': transl_std},
        #         'forward_directions': {'mean': forward_dir_mean, 'std': forward_dir_std},
        #         'global_velocities': {'mean': global_vel_mean, 'std': global_vel_std},
        #         'rotational_velocities': {'mean': rot_vel_mean, 'std': rot_vel_std},
        #     }
        #     self.save_stats_to_json(stats_dict, 'normalization_stats.json')

        for i in range(num_data):
            # normalized_motion, _, _ = self.normalize_data(motions[i])
            # normalized_translation, _, _ = self.normalize_data(translations[i])
            # normalized_forward_direction, _, _ = self.normalize_data(forward_directions[i])
            # normalized_global_velocity, _, _=  self.normalize_data(global_velocities[i])
            # normalized_rotational_velocity, _, _ = self.normalize_data(rotational_velocities[i])

            # # Converting inputs to tensors
            # motion = torch.tensor(normalized_motion, dtype=torch.float32) # (seq_len, num_joints, 3)
            # transl = torch.unsqueeze(torch.tensor(normalized_translation, dtype=torch.float32), dim=2)  # (seq_len, 3) -> (seq_len, 1, 3)
            # forward_dir = torch.unsqueeze(torch.tensor(normalized_forward_direction, dtype=torch.float32), dim=2)  # (seq_len, 3) -> (seq_len, 1, 3)
            # global_vel = torch.unsqueeze(torch.tensor(normalized_global_velocity, dtype=torch.float32), dim=2)  # (seq_len-1, 3) -> ( seq_len-1, 1, 3)
            # rot_vel = torch.unsqueeze(torch.unsqueeze(torch.tensor(normalized_rotational_velocity, dtype=torch.float32), dim=1), dim=1)  # (seq_len-1) -> (seq_len-1, 1, 1)
            
            # Converting inputs to tensors
            motion = torch.tensor(motions[i], dtype=torch.float32) # (seq_len, num_joints, 3)
            transl = torch.unsqueeze(torch.tensor(translations[i], dtype=torch.float32), dim=2)  # (seq_len, 3) -> (seq_len, 1, 3)
            forward_dir = torch.unsqueeze(torch.tensor(forward_directions[i], dtype=torch.float32), dim=2)  # (seq_len, 3) -> (seq_len, 1, 3)
            global_vel = torch.unsqueeze(torch.tensor(global_velocities[i], dtype=torch.float32), dim=2)  # (seq_len-1, 3) -> ( seq_len-1, 1, 3)
            rot_vel = torch.unsqueeze(torch.unsqueeze(torch.tensor(rotational_velocities[i], dtype=torch.float32), dim=1), dim=1)  # (seq_len-1) -> (seq_len-1, 1, 1)
            
            # Flatten data
            motion_flat = motion.view(-1, 22 * 3)  # (NUM_FRAMES, 22*3)
            transl_flat = transl.view(-1, 3)  # (NUM_FRAMES, 3)
            forward_dir_flat = forward_dir.view(-1, 3)  # (NUM_FRAMES, 3)
            global_vel_flat = global_vel.view(-1, 3)  # ((NUM_FRAMES-1), 3)
            rot_vel_flat = rot_vel.view(-1, 1)  # ((NUM_FRAMES-1), 1)

            init_pose_vector = torch.cat([motion_flat[:1],
                                          transl_flat[:1],
                                          forward_dir_flat[:1]], dim=1)
    
            next_pose_vector = torch.cat([motion_flat[1:],
                                          transl_flat[1:],
                                          forward_dir_flat[1:],
                                          global_vel_flat,
                                          rot_vel_flat], dim=1)
            
            # add EoS token: torch.zeros((1, 76))
            next_pose_vector = torch.cat([next_pose_vector, torch.zeros((1, 76))], dim=0)
            
            init_poses.append(init_pose_vector)
            next_motions.append(next_pose_vector)
        
        return init_poses, next_motions

    def normalize(self, data, mean, std):
        return (data - mean) / std

    def parse_motion_vector(motion_vector):
        num_frames_minus_1, _ = motion_vector.shape
        num_frames = num_frames_minus_1 + 1
        
        poses = np.zeros((num_frames, NUM_JOINTS, 3))
        poses[1:] = motion_vector[:, :NUM_JOINTS*3].reshape(num_frames_minus_1, NUM_JOINTS, 3)
        
        translation = np.zeros((num_frames, 3))
        translation[1:] = motion_vector[:, NUM_JOINTS*3:NUM_JOINTS*3+3]
        
        forward_direction = np.zeros((num_frames, 3))
        forward_direction[1:] = motion_vector[:, NUM_JOINTS*3+3:NUM_JOINTS*3+6]
        
        global_velocity = np.zeros((num_frames_minus_1, 3))
        global_velocity[:] = motion_vector[:, NUM_JOINTS*3+6:NUM_JOINTS*3+9]
        
        rotational_velocity = np.zeros((num_frames_minus_1, 1))
        rotational_velocity[:] = motion_vector[:, NUM_JOINTS*3+9:]
        
        return poses, translation, forward_direction, global_velocity, rotational_velocity
    
    

def create_datasets(data_dict, device, train_ratio=0.8, val_ratio=0.0, test_ratio=0.2):
    data_list = data_dict['name']
    frame_lists = data_dict['frame_lists']
    labels = data_dict['labels']
    object_boxes = data_dict['object_boxes']
    motions = np.array(data_dict['motions'], dtype=object)
    translations = np.array(data_dict['translations'], dtype=object)
    forward_directions = np.array(data_dict['forward_directions'], dtype=object)
    global_velocities = np.array(data_dict['global_velocities'], dtype=object)
    rotational_velocities = np.array(data_dict['rotational_velocities'], dtype=object)

    dataset = MotionDataset(data_list,
                            frame_lists, 
                            object_boxes, 
                            labels, 
                            motions,
                            translations,
                            forward_directions,
                            global_velocities,
                            rotational_velocities,
                            device)
    
    dataset_size = len(dataset)

    # split train, val, test
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = int(test_ratio * dataset_size)

    generator = torch.Generator()
    generator.manual_seed(0)    
    train_dataset, val_dataset, test_dataset = random_split(dataset, 
                                                            [train_size, val_size, test_size],
                                                            generator=generator)

    return train_dataset, val_dataset, test_dataset


def get_loader(
    dataset_path,
    batch_size=1,
    device=None
):
    data_dict = load_data(dataset_path, relative=True)
    # print("initial_poses shape:", len(initial_poses))
    # print("object_boxes shape:", len(object_boxes))
    # print("next_motion_descriptions shape:", len(next_motion_descriptions))
    # print("next_motions shape:", len(next_motions))

    # print("initial_poses shape:", initial_poses[0].shape)
    # print("object_boxes shape:", object_boxes[0].shape)
    # print("next_motion_descriptions shape:", next_motion_descriptions[0].shape)
    # print("next_motions shape:", next_motions[0].shape)

    train_dataset, val_dataset, test_dataset = create_datasets(data_dict, device)

    # create dataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader
